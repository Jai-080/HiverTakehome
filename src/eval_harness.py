"""
Hour 9.5-10.5: LLM-as-judge reply-quality evaluation + human calibration
sample.

Part 1 (--stage judge): score the full system's drafted_reply for all 200
golden-set rows on 5 dimensions (1-5 + rationale) using Phi-3-mini as a
judge. Deliberately distinct system prompt (skeptical external reviewer,
not the generation persona) and decoding (temperature=0.3, actual
sampling) from draft_reply.py's greedy generation, so this isn't just the
generator run backwards. Judge and generator share a model (Phi-3-mini)
since no second model was available within budget/hardware -- a known
self-preference-bias risk this script documents, not fixes. See
DECISIONS.md.

Part 2 (--stage sample): stratified-sample 45 rows across
hard_case_category for a human calibration set, blank score columns.

Part 3 lives in src/compute_judge_agreement.py -- run only after the
human scores in reports/human_judge_sample.csv are filled in.

Usage:
  .venv/Scripts/python.exe src/eval_harness.py --stage judge    # Part 1
  .venv/Scripts/python.exe src/eval_harness.py --stage sample   # Part 2
  .venv/Scripts/python.exe src/eval_harness.py --stage all      # both (default)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from classify_intent import _extract_json, _generate_raw, load_model
from sampling import stratified_sample

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_GOLDEN = Path("reports/golden_set_candidates_labeled.csv")
OUT_JUDGE_SCORES = Path("reports/llm_judge_scores.csv")
OUT_HUMAN_SAMPLE = Path("reports/human_judge_sample.csv")

JUDGE_MAX_NEW_TOKENS = 400
JUDGE_TEMPERATURE = 0.3  # actual sampling -- distinct from draft_reply.py's greedy (do_sample=False)
CHECKPOINT_EVERY = 25

N_HUMAN_SAMPLE = 45
HUMAN_SAMPLE_SEED = 33

DIMENSIONS = ["relevance", "groundedness", "tone_match", "no_hallucinated_specifics", "actionability"]

# Deliberately distinct wording AND framing from draft_reply.py's DRAFT_INSTRUCTION
# -- an external skeptical auditor, not the assistant that wrote the reply.
JUDGE_INSTRUCTION = (
    "You are a skeptical quality auditor reviewing a customer support reply. You did "
    "NOT write this reply and have no stake in it looking good -- your job is to find "
    "problems, not to be charitable.\n\n"
    "Score the AGENT REPLY below on five dimensions, each on a 1-5 scale "
    "(1 = fails badly, 5 = excellent), with a one-sentence rationale for each:\n"
    "- relevance: does the reply address what the customer actually asked, rather "
    "than a generic non-answer?\n"
    "- groundedness: does the reply avoid inventing account-specific facts (order "
    "numbers, ship dates, refund amounts, tracking numbers) not present in the "
    "customer's message?\n"
    "- tone_match: does it read like a real brand support reply -- warm but "
    "professional -- rather than generic filler or a form letter?\n"
    "- no_hallucinated_specifics: distinct from groundedness -- does it avoid "
    "fabricated claims even if they sound plausible (a specific policy, timeline, "
    "or promise that was never actually made)?\n"
    "- actionability: does it tell the customer what happens next, rather than just "
    "acknowledging the problem?"
)
JUDGE_JSON_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object of the form: "
    '{"relevance": {"score": <1-5 int>, "rationale": "<one sentence>"}, '
    '"groundedness": {"score": <1-5 int>, "rationale": "<one sentence>"}, '
    '"tone_match": {"score": <1-5 int>, "rationale": "<one sentence>"}, '
    '"no_hallucinated_specifics": {"score": <1-5 int>, "rationale": "<one sentence>"}, '
    '"actionability": {"score": <1-5 int>, "rationale": "<one sentence>"}}. '
    "No other text, no markdown code fences."
)


def build_judge_prompt(tweet_text: str, drafted_reply: str) -> str:
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f'CUSTOMER TWEET: "{tweet_text}"\n\n'
        f'AGENT REPLY: "{drafted_reply}"\n\n'
        f"{JUDGE_JSON_INSTRUCTION}"
    )


def _valid_judge_json(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    for dim in DIMENSIONS:
        entry = parsed.get(dim)
        if not isinstance(entry, dict):
            return False
        score = entry.get("score")
        if not isinstance(score, (int, float)) or not (1 <= score <= 5):
            return False
        if not entry.get("rationale"):
            return False
    return True


JUDGE_STATS = {"calls": 0, "retries": 0, "failures": 0}


def judge_reply(tweet_text: str, drafted_reply: str, model, tokenizer) -> dict:
    JUDGE_STATS["calls"] += 1
    prompt = build_judge_prompt(tweet_text, drafted_reply)
    raw = _generate_raw(model, tokenizer, prompt, max_new_tokens=JUDGE_MAX_NEW_TOKENS, temperature=JUDGE_TEMPERATURE)
    parsed = _extract_json(raw)

    if not _valid_judge_json(parsed):
        JUDGE_STATS["retries"] += 1
        strict_prompt = (
            prompt + "\n\nYour previous response was not valid JSON matching the required "
            "shape. Respond with ONLY valid JSON, no other text, no explanation, no markdown."
        )
        raw = _generate_raw(
            model, tokenizer, strict_prompt, max_new_tokens=JUDGE_MAX_NEW_TOKENS, temperature=JUDGE_TEMPERATURE
        )
        parsed = _extract_json(raw)

    if not _valid_judge_json(parsed):
        JUDGE_STATS["failures"] += 1
        raise ValueError(f"Could not parse a valid judge response after retry: {raw!r}")

    return parsed


def run_judge() -> None:
    golden = pd.read_csv(IN_GOLDEN)
    print(f"Judging {len(golden)} drafted replies from {IN_GOLDEN}\n")

    model, tokenizer = load_model()

    records = []
    done_ids = set()
    if OUT_JUDGE_SCORES.exists():
        prev = pd.read_csv(OUT_JUDGE_SCORES)
        records = prev.to_dict("records")
        done_ids = set(prev["customer_tweet_id"])
        print(f"Resuming: {len(done_ids)}/{len(golden)} already judged in {OUT_JUDGE_SCORES}")

    start = time.time()
    for _, row in golden.iterrows():
        if row["customer_tweet_id"] in done_ids:
            continue

        try:
            scores = judge_reply(row["tweet_text"], row["drafted_reply"], model, tokenizer)
        except ValueError as exc:
            print(f"  [JUDGE FAILURE] {exc}")
            continue

        record = {"customer_tweet_id": row["customer_tweet_id"]}
        for dim in DIMENSIONS:
            record[f"{dim}_score"] = scores[dim]["score"]
            record[f"{dim}_rationale"] = scores[dim]["rationale"]
        records.append(record)

        if len(records) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(OUT_JUDGE_SCORES, index=False)
            elapsed = time.time() - start
            print(f"  judged {len(records)}/{len(golden)} (checkpointed, {elapsed:.0f}s elapsed)")

    result = pd.DataFrame(records)
    OUT_JUDGE_SCORES.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_JUDGE_SCORES, index=False)
    print(
        f"\nSaved {len(result)} judge scores to {OUT_JUDGE_SCORES} "
        f"(retries: {JUDGE_STATS['retries']}, failures: {JUDGE_STATS['failures']}, of {JUDGE_STATS['calls']} calls)"
    )


def run_sample() -> None:
    golden = pd.read_csv(IN_GOLDEN)
    golden["hard_case_category"] = golden["hard_case_category"].fillna("")

    sample = stratified_sample(golden, "hard_case_category", N_HUMAN_SAMPLE, random_state=HUMAN_SAMPLE_SEED)
    print(f"Stratified {len(sample)}/{N_HUMAN_SAMPLE} rows across hard_case_category:")
    print(sample["hard_case_category"].value_counts())

    out = sample[["customer_tweet_id", "hard_case_category", "tweet_text", "drafted_reply"]].copy()
    for dim in DIMENSIONS:
        out[f"{dim}_score"] = ""

    OUT_HUMAN_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_HUMAN_SAMPLE, index=False)
    print(f"\nSaved human calibration sample to {OUT_HUMAN_SAMPLE} (score columns left blank)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["judge", "sample", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("judge", "all"):
        run_judge()
    if args.stage in ("sample", "all"):
        run_sample()


if __name__ == "__main__":
    main()
