"""
Hour 3.5-5: retrieval-grounded reply drafting.

draft_reply(tweet_text, predicted_intent, fraud_override) drafts a reply
mirroring AmazonHelp's actual resolution pattern from similar historical
(customer_tweet, brand_reply) pairs, without inventing account-specific
facts. Fraud-flagged messages short-circuit to a fixed holding message --
no retrieval, no LLM call.

Note: rag_corpus.csv (built by build_rag_index.py) has no intent labels of
its own -- it's a substance-filtered slice of the working set that
classify_intent.py never ran over (classifying all 4,228 rows with Phi-3
would cost ~6+ hours of GPU time). Backfilled with predicted_intent here
via the FAST local baseline classifier (TF-IDF+LogReg, balanced) instead
of the LLM -- these labels only bucket retrieval candidates, never feed
evaluation, so the baseline's own accuracy limits are an acceptable
trade-off for this internal use. See DECISIONS.md.

Usage: .venv/Scripts/python.exe src/draft_reply.py   # test on ~15 tweets spanning different intents, print grounding + drafts
"""
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from classify_intent import _generate_raw, load_model as load_phi3
from sampling import stratified_sample

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_PSEUDO = Path("data/processed/pseudo_labels.csv")
IN_CORPUS = Path("data/processed/rag_corpus.csv")
IN_CORPUS_EMBEDDINGS = Path("data/processed/rag_corpus_embeddings.npy")
BASELINE_MODEL = Path("data/processed/baseline_intent_classifier_balanced.joblib")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_NEW_TOKENS = 120  # classify_intent.py's 60 is fine for a JSON label; drafting needs more room
TOP_K_SIMILARITY = 20
TOP_K_FEWSHOT = 3
MIN_INTENT_MATCHES = 3
FRAUD_HOLDING_MESSAGE = "We've flagged this for immediate review by our security team."
N_TEST = 15
TEST_SEED = 13

DRAFT_INSTRUCTION = (
    "You are AmazonHelp, Amazon's customer support account on Twitter. Draft a "
    "reply to the customer's tweet below, in AmazonHelp's voice, mirroring the "
    "resolution pattern shown in the similar past examples.\n\n"
    "Do NOT invent any account-specific facts not present in the current tweet -- "
    "no order numbers, ship dates, tracking numbers, refund amounts, or other "
    "specifics the customer didn't actually provide. If a past example mentions "
    "such details, use it only as a pattern for HOW to respond, not as a fact "
    "about this customer's situation.\n\n"
    "Never use any person's name in your draft -- not the customer's, not an "
    "agent's -- whether it comes from a retrieved example or you'd otherwise be "
    "tempted to invent one. Address the customer generically ('Hi,' or 'Hi "
    "there,' or no name at all).\n\n"
    "Paraphrase in your own words -- do not copy a retrieved reply verbatim "
    "even when the situation is very similar."
)

# Agent-signature leakage ("^LR", "^MN") from retrieved historical replies --
# this is text cleanup, not a generation-policy question, so still stripped
# from the few-shot examples before they enter the prompt.
SIGNATURE_RE = re.compile(r"\^[A-Z]{2,3}\b")

# Small allowlist of capitalized brand/product terms that are NOT person names
# -- used by the generic name-leak check below.
COMMON_WORDS = {
    "Amazon", "AmazonHelp", "AmazonIndia", "AmazonPrime", "Prime", "Kindle",
    "Alexa", "Echo", "Fire", "Visa", "MRP", "AWB", "DM", "US", "UK",
}

NEAR_VERBATIM_THRESHOLD = 0.85

_RESOURCES = None


def clean_retrieved_reply(text: str) -> str:
    text = SIGNATURE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_name_leaks(text: str) -> list[str]:
    """Generic check: any capitalized word, not sentence-initial, not a known
    brand/entity, sitting in a greeting-adjacent position (immediately before
    '!'/',' , or immediately after Hi/Hello/Hey/Dear). Since the policy now
    bans ALL person names regardless of source, any match is flagged -- no
    need to check whether it's traceable to a retrieved example."""
    leaks = set()

    greeting_re = re.compile(r"\b(?:Hi|Hello|Hey|Dear)\b,?\s+([A-Z][a-zA-Z]+)")
    for m in greeting_re.finditer(text):
        word = m.group(1)
        if word not in COMMON_WORDS:
            leaks.add(word)

    for m in re.finditer(r"\b([A-Z][a-zA-Z]+)[,!]", text):
        word = m.group(1)
        if word in COMMON_WORDS:
            continue
        prefix = text[: m.start(1)].rstrip()
        is_sentence_start = (not prefix) or prefix[-1] in '.!?"\''
        if not is_sentence_start:
            leaks.add(word)

    return sorted(leaks)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a, tokens_b = _tokenize(a), _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def validate_draft(reply: str, original_tweet: str, examples: pd.DataFrame = None) -> list[str]:
    warnings = []

    for m in SIGNATURE_RE.finditer(reply):
        warnings.append(f"leftover agent signature {m.group(0)!r}")

    for name in _find_name_leaks(reply):
        warnings.append(f"name used in draft: {name!r} (names are banned by policy)")

    if examples is not None and len(examples) > 0:
        top1_reply = clean_retrieved_reply(examples.iloc[0]["brand_reply_clean"])
        overlap = _token_overlap_ratio(reply, top1_reply)
        if overlap >= NEAR_VERBATIM_THRESHOLD:
            warnings.append(f"near-verbatim: {overlap:.0%} token overlap with top-1 retrieved reply")

    return warnings


def load_resources() -> dict:
    print(f"Loading {IN_CORPUS} + {IN_CORPUS_EMBEDDINGS} ...")
    corpus = pd.read_csv(IN_CORPUS)
    corpus_embeddings = np.load(IN_CORPUS_EMBEDDINGS)

    print(f"Backfilling corpus intents via {BASELINE_MODEL} (fast local baseline, not the LLM)")
    baseline = joblib.load(BASELINE_MODEL)
    corpus["predicted_intent"] = baseline.predict(corpus["customer_tweet_clean"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBED_MODEL} for query embedding (device={device}) ...")
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    phi3_model, phi3_tokenizer = load_phi3()

    return {
        "corpus": corpus,
        "corpus_embeddings": corpus_embeddings,
        "embed_model": embed_model,
        "phi3_model": phi3_model,
        "phi3_tokenizer": phi3_tokenizer,
    }


def _get_resources() -> dict:
    global _RESOURCES
    if _RESOURCES is None:
        _RESOURCES = load_resources()
    return _RESOURCES


def retrieve_examples(tweet_text: str, predicted_intent: str, resources: dict) -> pd.DataFrame:
    corpus = resources["corpus"]
    corpus_embeddings = resources["corpus_embeddings"]
    embed_model = resources["embed_model"]

    query_vec = embed_model.encode([QUERY_PREFIX + tweet_text], normalize_embeddings=True).astype(np.float32)[0]

    # Both sides are L2-normalized (query here, corpus in build_rag_index.py) -> dot product == cosine similarity.
    sims = corpus_embeddings @ query_vec
    top_idx = np.argsort(sims)[::-1][:TOP_K_SIMILARITY]
    top = corpus.iloc[top_idx].copy()
    top["similarity"] = sims[top_idx]

    matching = top[top["predicted_intent"] == predicted_intent]
    if len(matching) < MIN_INTENT_MATCHES:
        print(
            f"  [fallback] only {len(matching)} of top-{TOP_K_SIMILARITY} match intent "
            f"'{predicted_intent}' -- falling back to top-{TOP_K_FEWSHOT} by pure similarity"
        )
        return top.sort_values("similarity", ascending=False).head(TOP_K_FEWSHOT)

    return matching.sort_values("similarity", ascending=False).head(TOP_K_FEWSHOT)


def build_draft_prompt(tweet_text: str, examples: pd.DataFrame) -> str:
    lines = [DRAFT_INSTRUCTION, "\nSimilar past examples (customer tweet -> AmazonHelp's reply):"]
    for _, row in examples.iterrows():
        lines.append(f'- Customer: "{row["customer_tweet_clean"]}"')
        lines.append(f'  AmazonHelp: "{clean_retrieved_reply(row["brand_reply_clean"])}"')
    lines.append(f'\nNow draft a reply to this customer tweet:\n"{tweet_text}"')
    lines.append("\nRespond with ONLY the reply text. No quotes, no explanation.")
    return "\n".join(lines)


def _draft_and_retrieve(tweet_text: str, predicted_intent: str, fraud_override: bool, resources: dict):
    """Internal helper shared by draft_reply() and the test harness -- returns (reply, examples_or_None)."""
    if fraud_override:
        return FRAUD_HOLDING_MESSAGE, None

    examples = retrieve_examples(tweet_text, predicted_intent, resources)
    prompt = build_draft_prompt(tweet_text, examples)
    raw = _generate_raw(resources["phi3_model"], resources["phi3_tokenizer"], prompt, max_new_tokens=MAX_NEW_TOKENS)
    return raw.strip(), examples


def draft_reply(tweet_text: str, predicted_intent: str, fraud_override: bool) -> str:
    resources = _get_resources()
    reply, _ = _draft_and_retrieve(tweet_text, predicted_intent, fraud_override, resources)
    return reply


# The two near-duplicate cases that showed name/signature leakage in the
# first test run -- forced back in so this run confirms the fix actually
# resolves them, not just that new random cases happen to look clean.
FORCED_TEST_TWEETS = [
    "Are you freaking kidding me ?? This is how you deliver to businesses now?? <URL>",
    "i have a free trial of Amzon prime but now bank statement shows that i have to pay 2 £. Will send screens in DM",
]


OUT_TEST_RESULTS = Path("data/processed/draft_reply_test_results.csv")


def run_test() -> None:
    resources = _get_resources()

    pseudo = pd.read_csv(IN_PSEUDO)
    candidates = pseudo[~pseudo["fraud_override"]].copy()

    forced = candidates[candidates["customer_tweet_clean"].isin(FORCED_TEST_TWEETS)]
    remaining_pool = candidates[~candidates["customer_tweet_clean"].isin(FORCED_TEST_TWEETS)]
    sample = stratified_sample(remaining_pool, "predicted_intent", N_TEST - len(forced), random_state=TEST_SEED)
    sample = pd.concat([forced, sample]).reset_index(drop=True)

    print(f"\nTesting draft_reply on {len(sample)} tweets across {sample['predicted_intent'].nunique()} intents\n")
    n_checked = 0
    n_near_verbatim = 0
    results = []
    for _, row in sample.iterrows():
        text = row["customer_tweet_clean"]
        intent = row["predicted_intent"]
        confidence = row["confidence"]
        fraud = bool(row["fraud_override"])

        print("=" * 100)
        print(f"TWEET [{intent}]: {text}")

        reply, examples = _draft_and_retrieve(text, intent, fraud, resources)

        top1_similarity = None
        if examples is not None:
            top1_similarity = float(examples.iloc[0]["similarity"])
            print("\nRetrieved examples:")
            for _, ex in examples.iterrows():
                print(f'  - (sim={ex["similarity"]:.3f}, intent={ex["predicted_intent"]}) "{ex["customer_tweet_clean"][:80]}"')
                print(f'    -> "{ex["brand_reply_clean"][:120]}"')

        print(f"\nDRAFTED REPLY: {reply}")

        warnings = []
        if not fraud:
            n_checked += 1
            warnings = validate_draft(reply, text, examples)
            if warnings:
                for w in warnings:
                    print(f"  [VALIDATION WARNING] {w}")
                    if "near-verbatim" in w:
                        n_near_verbatim += 1
            else:
                print("  [validate_draft: clean]")
        print()

        results.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_tweet_clean": text,
                "predicted_intent": intent,
                "confidence": confidence,
                "fraud_override": fraud,
                "retrieval_top1_similarity": top1_similarity,
                "drafted_reply": reply,
                "validate_draft_flags": warnings,
            }
        )

    if n_checked:
        print(f"Near-verbatim rate: {n_near_verbatim}/{n_checked} drafts ({n_near_verbatim / n_checked:.0%})")

    OUT_TEST_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_TEST_RESULTS, index=False)
    print(f"Saved test results to {OUT_TEST_RESULTS}")


if __name__ == "__main__":
    run_test()
