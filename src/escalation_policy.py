"""
Hour 5-6: escalation policy -- deterministic function on intent, classifier
confidence, fraud override, and retrieval quality that decides auto-handle
vs. escalate-to-human, with a one-line stated reason.

Threshold selection: chosen from actual observed score distributions, not
defaulted to round numbers -- see print_score_distributions() and
DECISIONS.md.

Usage:
  .venv/Scripts/python.exe src/escalation_policy.py --distributions   # print the confidence/similarity distributions that justified the thresholds
  .venv/Scripts/python.exe src/escalation_policy.py --test            # run decide() over draft_reply.py's saved 15-tweet test results
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_PSEUDO = Path("data/processed/pseudo_labels.csv")
IN_CORPUS_EMBEDDINGS = Path("data/processed/rag_corpus_embeddings.npy")
IN_TEST_RESULTS = Path("data/processed/draft_reply_test_results.csv")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SIM_SAMPLE_N = 200
SIM_SAMPLE_SEED = 99

# High-risk intents always escalate regardless of confidence -- each
# requires a human for account access, money movement, or seller-trust
# action, none of which this system should auto-resolve.
HIGH_RISK_INTENTS = {
    "account_access",
    "billing_subscription_dispute",
    "refund_return_dispute",
    "marketplace_seller_trust",
}

# Chosen from the actual observed distributions (run --distributions to
# reproduce), roughly the 25th percentile of each -- "the bottom quarter of
# observed scores is unreliable" -- not a round-number default.
# Confidence (pseudo_labels.csv, n=700): min=0.70 25%=0.90 median=0.90 75%=0.95 max=0.98
# Self-reported confidence from a small local model is heavily clustered
# near the top (median IS the 25th percentile, both 0.90) -- there's no
# gentle slope to pick a "natural" cutoff from, so the 25th percentile is
# a blunt but data-grounded choice: anything below the bottom quarter of
# what the model actually reports gets treated as unreliable.
CONF_THRESHOLD = 0.90
# Retrieval top-1 similarity (200-query sample against the 4,228-row corpus):
# min=0.61 25%=0.74 median=0.81 75%=0.92 max=0.98
SIM_THRESHOLD = 0.74


def decide(
    intent: str,
    confidence: float,
    fraud_override: bool,
    retrieval_top1_similarity: float,
    validate_draft_flags: list,
) -> dict:
    if fraud_override:
        return {"decision": "escalate", "reason": "security/fraud keyword match"}

    if intent in HIGH_RISK_INTENTS:
        return {
            "decision": "escalate",
            "reason": (
                f"high-risk intent '{intent}': requires human for account/money/trust "
                "action, regardless of confidence"
            ),
        }

    if validate_draft_flags:
        return {
            "decision": "escalate",
            "reason": f"generated reply failed automated safety validation: {validate_draft_flags}",
        }

    failed = []
    if confidence < CONF_THRESHOLD:
        failed.append(f"confidence {confidence:.2f} < {CONF_THRESHOLD}")
    if retrieval_top1_similarity < SIM_THRESHOLD:
        failed.append(f"retrieval_top1_similarity {retrieval_top1_similarity:.2f} < {SIM_THRESHOLD}")

    if failed:
        return {"decision": "escalate", "reason": "; ".join(failed)}

    return {
        "decision": "auto",
        "reason": (
            f"confidence {confidence:.2f} >= {CONF_THRESHOLD} and "
            f"similarity {retrieval_top1_similarity:.2f} >= {SIM_THRESHOLD}"
        ),
    }


def print_score_distributions() -> None:
    pseudo = pd.read_csv(IN_PSEUDO)
    print(f"Confidence distribution (pseudo_labels.csv, n={len(pseudo)}):")
    print(pseudo["confidence"].describe(percentiles=[0.25, 0.5, 0.75]))

    corpus_embeddings = np.load(IN_CORPUS_EMBEDDINGS)
    sample = pseudo.sample(n=SIM_SAMPLE_N, random_state=SIM_SAMPLE_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {EMBED_MODEL} for similarity sample (device={device}) ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    queries = [QUERY_PREFIX + t for t in sample["customer_tweet_clean"]]
    query_vecs = model.encode(queries, normalize_embeddings=True, batch_size=32).astype(np.float32)

    sims = query_vecs @ corpus_embeddings.T
    top1 = pd.Series(sims.max(axis=1))
    print(
        f"\nTop-1 similarity distribution ({SIM_SAMPLE_N} queries against "
        f"{len(corpus_embeddings)}-row corpus):"
    )
    print(top1.describe(percentiles=[0.25, 0.5, 0.75]))


def run_test() -> None:
    df = pd.read_csv(IN_TEST_RESULTS)

    print(f"Running decide() over {len(df)} saved test results from {IN_TEST_RESULTS}\n")
    n_leaked = 0
    for _, row in df.iterrows():
        flags = ast.literal_eval(row["validate_draft_flags"]) if pd.notna(row["validate_draft_flags"]) else []
        sim = row["retrieval_top1_similarity"] if pd.notna(row["retrieval_top1_similarity"]) else 0.0

        result = decide(
            intent=row["predicted_intent"],
            confidence=row["confidence"],
            fraud_override=bool(row["fraud_override"]),
            retrieval_top1_similarity=sim,
            validate_draft_flags=flags,
        )

        print(f"[{result['decision']:8s}] intent={row['predicted_intent']:<32s} {result['reason']}")

        if result["decision"] == "auto" and (row["fraud_override"] or row["predicted_intent"] in HIGH_RISK_INTENTS):
            n_leaked += 1
            print("  !! SANITY CHECK FAILED: fraud/high-risk case auto-handled !!")

    print(
        f"\nSanity check: {n_leaked} fraud-override or high-risk-intent case(s) "
        f"slipped into auto-handle (expect 0)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributions", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.distributions:
        print_score_distributions()
    if args.test:
        run_test()
    if not args.distributions and not args.test:
        parser.print_help()


if __name__ == "__main__":
    main()
