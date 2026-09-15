"""
Hour 6.5-8.5: build the golden-set candidate file for hand-labeling.

Stratified sample (160, across the 8 intents per the baseline classifier's
predictions) + oversampled hard cases (~40: fraud keywords, near-verbatim
retrieval stress, pay/paid-for vocabulary overlap, low retrieval similarity)
from the held-out pool (amazonhelp_heldout.csv, never touched before now).
Runs the full pipeline (classify_intent, check_fraud_override, draft_reply,
validate_draft, escalation_policy.decide) so the candidate file already
carries AI-assisted pre-labels -- the true_intent/true_escalation_decision/
notes columns are left blank for hand review.

Usage: .venv/Scripts/python.exe src/build_golden_set.py
"""
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classify_intent import check_fraud_override, classify_intent
from draft_reply import QUERY_PREFIX, _draft_and_retrieve, _get_resources, validate_draft
from escalation_policy import decide
from sampling import stratified_sample

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_HELDOUT = Path("data/processed/amazonhelp_heldout.csv")
BASELINE_MODEL = Path("data/processed/baseline_intent_classifier_balanced.joblib")
OUT_CANDIDATES = Path("reports/golden_set_candidates.csv")

N_STRATIFIED = 160
N_HARD_PER_CATEGORY = 10
TARGET_HARD_TOTAL = 40
STRATIFY_SEED = 21
HARD_CASE_SEED = 22
HIGH_SIM_STRESS_THRESHOLD = 0.9
LOW_SIM_CANDIDATE_POOL = 50  # widen before exclusion-filtering down to N_HARD_PER_CATEGORY
CHECKPOINT_EVERY = 25

PAY_FOR_RE = re.compile(r"\bpay(?:ing|s)?\s+for\b|\bpaid\s+for\b", flags=re.IGNORECASE)


def build_sampling_pool() -> pd.DataFrame:
    df = pd.read_csv(IN_HELDOUT)
    print(f"Held-out pool: {len(df):,} rows")

    baseline = joblib.load(BASELINE_MODEL)
    df["predicted_intent_baseline"] = baseline.predict(df["customer_tweet_clean"])
    df["fraud_override"] = df["customer_tweet_clean"].map(check_fraud_override)
    df["pay_for_match"] = df["customer_tweet_clean"].str.contains(PAY_FOR_RE, na=False)

    resources = _get_resources()
    embed_model = resources["embed_model"]
    corpus_embeddings = resources["corpus_embeddings"]

    print("Embedding held-out tweets to compute max RAG-corpus similarity ...")
    queries = [QUERY_PREFIX + t for t in df["customer_tweet_clean"]]
    query_vecs = embed_model.encode(
        queries, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    ).astype(np.float32)
    sims = query_vecs @ corpus_embeddings.T
    df["max_rag_similarity"] = sims.max(axis=1)

    return df


def select_golden_pool(df: pd.DataFrame) -> pd.DataFrame:
    stratified = stratified_sample(df, "predicted_intent_baseline", N_STRATIFIED, random_state=STRATIFY_SEED)
    remaining = df[~df["customer_tweet_id"].isin(stratified["customer_tweet_id"])]

    selected_ids = set()
    hard_parts = []

    def take(pool: pd.DataFrame, label: str, n_target: int) -> int:
        pool = pool[~pool["customer_tweet_id"].isin(selected_ids)]
        n = min(n_target, len(pool))
        sampled = pool.sample(n=n, random_state=HARD_CASE_SEED) if n else pool.iloc[0:0].copy()
        print(f"  {label}: {n}/{n_target} selected (pool had {len(pool)} candidates)")
        selected_ids.update(sampled["customer_tweet_id"])
        hard_parts.append(sampled.assign(hard_case_category=label))
        return n

    print("\nHard-case oversampling:")
    take(remaining[remaining["fraud_override"]], "fraud_keyword", N_HARD_PER_CATEGORY)
    take(remaining[remaining["max_rag_similarity"] > HIGH_SIM_STRESS_THRESHOLD], "near_verbatim_stress", N_HARD_PER_CATEGORY)
    take(remaining[remaining["pay_for_match"]], "pay_for_vocab_overlap", N_HARD_PER_CATEGORY)
    take(remaining.sort_values("max_rag_similarity").head(LOW_SIM_CANDIDATE_POOL), "low_retrieval_similarity", N_HARD_PER_CATEGORY)

    n_so_far = sum(len(p) for p in hard_parts)
    shortfall = TARGET_HARD_TOTAL - n_so_far
    if shortfall > 0:
        print(f"  shortfall of {shortfall} (near_verbatim_stress is naturally scarce) -- "
              f"backfilling from low_retrieval_similarity to reach {TARGET_HARD_TOTAL} hard cases total")
        take(
            remaining.sort_values("max_rag_similarity").head(LOW_SIM_CANDIDATE_POOL + shortfall),
            "low_retrieval_similarity_backfill",
            shortfall,
        )

    hard = pd.concat(hard_parts)
    stratified = stratified.assign(hard_case_category="")

    golden = pd.concat([stratified, hard]).drop_duplicates(subset="customer_tweet_id").reset_index(drop=True)
    print(f"\nGolden pool: {len(stratified)} stratified + {len(hard)} hard-case rows -> {len(golden)} unique total")
    return golden


def run_pipeline(golden: pd.DataFrame) -> pd.DataFrame:
    resources = _get_resources()
    phi3_model, phi3_tokenizer = resources["phi3_model"], resources["phi3_tokenizer"]

    records = []
    done_ids = set()
    if OUT_CANDIDATES.exists():
        prev = pd.read_csv(OUT_CANDIDATES)
        records = prev.to_dict("records")
        done_ids = set(prev["customer_tweet_id"])
        print(f"Resuming: {len(done_ids)}/{len(golden)} already processed in {OUT_CANDIDATES}")

    start = time.time()
    n_failed = 0
    for _, row in golden.iterrows():
        if row["customer_tweet_id"] in done_ids:
            continue
        text = row["customer_tweet_clean"]

        try:
            cls = classify_intent(text, phi3_model, phi3_tokenizer)
        except ValueError as exc:
            n_failed += 1
            print(f"  [CLASSIFY FAILURE] {exc}")
            continue

        fraud = check_fraud_override(text)
        reply, examples = _draft_and_retrieve(text, cls["intent"], fraud, resources)
        top1_sim = float(examples.iloc[0]["similarity"]) if examples is not None else 0.0
        flags = [] if fraud else validate_draft(reply, text, examples)

        decision = decide(
            intent=cls["intent"],
            confidence=cls["confidence"],
            fraud_override=fraud,
            retrieval_top1_similarity=top1_sim,
            validate_draft_flags=flags,
        )

        records.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "hard_case_category": row.get("hard_case_category", ""),
                "tweet_text": text,
                "true_intent": "",
                "true_escalation_decision": "",
                "predicted_intent": cls["intent"],
                "predicted_confidence": cls["confidence"],
                "drafted_reply": reply,
                "escalation_decision": decision["decision"],
                "escalation_reason": decision["reason"],
                "validate_draft_flags": flags,
                "notes": "",
            }
        )

        if len(records) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(OUT_CANDIDATES, index=False)
            elapsed = time.time() - start
            print(f"  processed {len(records)}/{len(golden)} (checkpointed, {elapsed:.0f}s elapsed)")

    result = pd.DataFrame(records)
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CANDIDATES, index=False)
    print(f"\nSaved {len(result)} golden-set candidates to {OUT_CANDIDATES} ({n_failed} classify failures)")
    return result


def main() -> None:
    df = build_sampling_pool()
    golden = select_golden_pool(df)
    run_pipeline(golden)


if __name__ == "__main__":
    main()
