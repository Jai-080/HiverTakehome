"""
Hour 2.5-3.5: intent classifier (local Phi-3-mini, few-shot JSON prompting) +
fraud/security escalation override (pure regex) + simple TF-IDF/LogReg
baseline trained on the classifier's own pseudo-labels.

Pivoted from Gemini generate_content to a local model after discovering
gemini-2.5-flash's free tier is capped at 20 requests/day on this project
(via the AI Studio rate-limit dashboard) -- unusable at pipeline scale.
See DECISIONS.md.

Usage:
  .venv/Scripts/python.exe src/classify_intent.py --stage pilot   # 20 tweets, print results, sanity-check before scaling up (default)
  .venv/Scripts/python.exe src/classify_intent.py --stage full    # 700-row sample from amazonhelp_working.csv -> data/processed/pseudo_labels.csv
  .venv/Scripts/python.exe src/classify_intent.py --stage train   # fit TF-IDF+LogReg baseline on pseudo_labels.csv, save + report
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_WORKING = Path("data/processed/amazonhelp_working.csv")
OUT_PSEUDO = Path("data/processed/pseudo_labels.csv")
OUT_MODEL = Path("data/processed/baseline_intent_classifier.joblib")

CLASSIFY_MODEL = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 60
N_PILOT = 20
PILOT_SEED = 7
N_FULL = 700
FULL_SEED = 42
CHECKPOINT_EVERY = 100

# Transcribed from reports/intent_taxonomy.md -- do not edit the taxonomy file
# itself, edit here if the definitions change.
INTENT_DEFS = {
    "delivery_shipping_issue": {
        "label": "Delivery/Shipping Issue",
        "definition": (
            "Late, damaged, missing, or wrong item; or a tracking status "
            '("delivered") that doesn\'t match what the customer actually received.'
        ),
        "examples": [
            "An iron box bought from Amazon was a damaged item because of less "
            "precautions in packing. So Replaced but no pickup. #Disappointed",
            "Several parcels delayed lately despite choosing next day and being "
            "a prime customer. What's going on?",
            # sarcasm trap: lexically positive wording, actually a delivery-handling complaint
            "Thanks for leaving my parcel outside in the rain - very clever!!!",
            # vocabulary-overlap trap found in pilot testing: mentions "pay"/payment
            # but the complaint is about a service level not being met, not the
            # charge itself -- this is delivery_shipping_issue, not billing_subscription_dispute
            "how you gonna give me a shipping estimate of next week when I pay for 1 day delivery",
        ],
    },
    "refund_return_dispute": {
        "label": "Refund/Return & Dispute Resolution",
        "definition": "Refund requests, balance/money disputes, subscribe-and-save issues.",
        "examples": [
            "will I be getting my postage refunded?",
            "A order placed by mistakenly. I cancelled the order. My amazon pay "
            "balance showing 0. My balance was 1299. When the amount will show "
            "on my balance?",
            "very poor experience with subscribe and save. Order never delivered. "
            "Refund messed up. Very poor show.",
        ],
    },
    "account_access": {
        "label": "Account & Access Issues",
        "definition": "Login lockout, password reset failure, checkout/exchange technical friction.",
        "examples": [
            "Hey you locked me out of my account after I bought something and "
            "now won't let me log back in, what gives?!?",
            "sorry let me say that again in english, I cannot log in into my "
            "account even after reseting the password multiple times",
            "not able to check out with exchange despite the product page saying so. Help",
        ],
    },
    "billing_subscription_dispute": {
        "label": "Billing/Subscription Charge Dispute",
        "definition": (
            "Unwanted Prime membership charges, trial charges, cancellation "
            "difficulty -- being charged the wrong amount or an unwanted charge. "
            "NOT about a paid-for service level (e.g. delivery speed) not being "
            "met -- that's delivery_shipping_issue even if payment is mentioned."
        ),
        "examples": [
            "second time in as many months that Amazon has gone into my account "
            "and stolen money for a Prime membership I do not want.",
            "I activated the Amazon Trial prime and £ 1 was deducted from my Visa card !",
            "Hi my account is meant to be closed but im still getting emails "
            "about prime payment issues?",
        ],
    },
    "digital_service_device_support": {
        "label": "Digital Service/Device Technical Support",
        "definition": "Echo, Kindle, Alexa, app bugs, cross-device sync issues.",
        "examples": [
            "Just as an aside, what the hell have done to the Kindle App?! "
            "It's bloody awful, give me back my carousel!!",
            "Since the most recent update my spot in my books will no longer "
            "sync between my devices. All settings are correct.",
            "Bought Minecraft from Amazon Store on previous device. Wiped & "
            "sold device, Xferred acct to new device, now I get this - help!",
        ],
    },
    "marketplace_seller_trust": {
        "label": "Marketplace/Seller Trust",
        "definition": "Counterfeit products, fake/misleading listings.",
        "examples": [
            "#PehleKaroPuriTayyari #FakeListings available at 699 without any "
            "deal, Lightning deal at 799. Same product with different MRP.",
            "Why does sell fake Maybelline products and even after many "
            "complaints, refuse to remove it and suspend the seller?",
        ],
    },
    "positive_feedback": {
        "label": "Positive Feedback / No Action Needed",
        "definition": (
            "Genuine gratitude or praise, requiring no support action. Lexical "
            "positivity alone is not sufficient -- watch for sarcasm (see the "
            "delivery-issue example)."
        ),
        "examples": [
            "I love online shopping. Christmas shopping is done. Thanks !",
            "So my will be arriving before 31st of October Thanks guys for speedy shipping.",
        ],
    },
    "general_complaint_other": {
        "label": "General Complaint / Other",
        "definition": "Venting or off-topic requests with no specific actionable resolution path.",
        "examples": [
            "Amazon's customer service is very nonsense its customer care "
            "service has become very nonsense, not helping people pockets are being stupid",
            "I got fed up with stubborn attitude. It seems Amazon team supports fraud.",
            "I'm almost positive and the have it out for me.",
        ],
    },
}

# Escalation override -- NOT a classifier intent, too rare to cluster reliably
# (see reports/intent_taxonomy.md). Pure regex, no model call, runs
# independently of classify_intent() and can force escalation regardless of
# predicted intent.
FRAUD_PATTERNS = [
    r"\bscam(med|mer|ming)?\b",
    r"\bfraud(ulent)?\b",
    r"\bphish(ing)?\b",
    r"\bhack(ed|er|ing)?\b",
    r"\bstole (my|our)\b",
    r"\bstolen (my|our)\b",
    r"\bunauthori[sz]ed\b",
    r"\bpolice\b",
    r"\bpretending to be you\b",
    r"\bidentity theft\b",
    r"\bsuspicious\b.{0,30}\b(email|account|message|link)\b",
    r"\baccount is locked\b.{0,25}\b(isn'?t|wasn'?t|not)\b",
    r"\bcc details\b",
    r"\bcredit card details\b",
]
FRAUD_RE = re.compile("|".join(FRAUD_PATTERNS), flags=re.IGNORECASE)


def check_fraud_override(text: str) -> bool:
    return bool(FRAUD_RE.search(text))


def build_system_instruction() -> str:
    lines = [
        "You are an intent classifier for AmazonHelp customer support tweets.",
        "Classify the customer's tweet into exactly one of these intents:",
        "",
    ]
    for key, spec in INTENT_DEFS.items():
        lines.append(f"- {key} ({spec['label']}): {spec['definition']}")
        for ex in spec["examples"]:
            lines.append(f'    e.g. "{ex}"')
    lines.append("")
    lines.append(
        "Return the single best-fitting intent key and your confidence (0-1) "
        "that this is the correct intent. Watch for sarcasm: lexically "
        "positive wording does not always mean positive_feedback."
    )
    return "\n".join(lines)


SYSTEM_INSTRUCTION = build_system_instruction()
JSON_INSTRUCTION = (
    'Respond with ONLY a single valid JSON object of the form '
    '{{"intent": "<one of the intent keys above>", "confidence": <0-1 number>}}. '
    "No other text, no markdown code fences, no explanation."
)


def build_prompt(text: str) -> str:
    return f'{SYSTEM_INSTRUCTION}\n\nClassify this customer tweet:\n"{text}"\n\n{JSON_INSTRUCTION}'


def load_model():
    print(f"Loading {CLASSIFY_MODEL} (4-bit, device_map=auto) ...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(
        CLASSIFY_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFY_MODEL)
    print(f"Loaded. device={model.device}")
    return model, tokenizer


def _generate_raw(
    model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float | None = None
) -> str:
    """temperature=None -> greedy decoding (do_sample=False), the default used for
    classification/drafting. Pass a float (e.g. 0.3) to enable actual sampling --
    used by eval_harness.py's judge so it isn't just the generator run backwards."""
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature is not None:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# Module-level counters so run_pilot/run_full can report how often the small
# model needed a retry or failed outright -- a real number worth reporting,
# since small models are less reliable at strict JSON than Gemini's
# structured-output mode.
STATS = {"calls": 0, "retries": 0, "failures": 0}


def classify_intent(text: str, model, tokenizer) -> dict:
    STATS["calls"] += 1
    prompt = build_prompt(text)
    raw = _generate_raw(model, tokenizer, prompt)
    parsed = _extract_json(raw)

    if parsed is None or parsed.get("intent") not in INTENT_DEFS or parsed.get("confidence") is None:
        STATS["retries"] += 1
        strict_prompt = (
            prompt + "\n\nYour previous response was not valid JSON. "
            "Respond with ONLY valid JSON, no other text, no explanation, no markdown."
        )
        raw = _generate_raw(model, tokenizer, strict_prompt)
        parsed = _extract_json(raw)

    if parsed is None or parsed.get("intent") not in INTENT_DEFS or parsed.get("confidence") is None:
        STATS["failures"] += 1
        raise ValueError(f"Could not parse a valid classification after retry: {raw!r}")

    return {"intent": parsed["intent"], "confidence": float(parsed["confidence"])}


def run_pilot(model, tokenizer) -> None:
    df = pd.read_csv(IN_WORKING)
    sample = df.sample(n=N_PILOT, random_state=PILOT_SEED).reset_index(drop=True)

    print(f"\nPilot: classifying {len(sample)} tweets (seed={PILOT_SEED}) ...\n")
    start = time.time()
    n_ok = 0
    for _, row in sample.iterrows():
        text = row["customer_tweet_clean"]
        try:
            result = classify_intent(text, model, tokenizer)
        except ValueError as exc:
            print(f"[PARSE FAILURE] {text[:100]}  -- {exc}")
            continue
        n_ok += 1
        fraud = check_fraud_override(text)
        flag = "  [FRAUD OVERRIDE]" if fraud else ""
        print(f"[{result['intent']:<32s} conf={result['confidence']:.2f}]{flag} {text[:100]}")

    elapsed = time.time() - start
    print(
        f"\nPilot done: {n_ok}/{len(sample)} classified OK in {elapsed:.1f}s "
        f"({elapsed / len(sample):.2f}s/tweet avg). "
        f"Retries: {STATS['retries']}, failures: {STATS['failures']} (of {STATS['calls']} calls)."
    )
    est_full_minutes = (elapsed / len(sample)) * N_FULL / 60
    print(f"Estimated time for the {N_FULL}-row full run at this rate: ~{est_full_minutes:.0f} min")


def _save_pseudo_labels(records: list) -> None:
    OUT_PSEUDO.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT_PSEUDO, index=False)


def run_full(model, tokenizer) -> None:
    df = pd.read_csv(IN_WORKING)
    sample = df.sample(n=N_FULL, random_state=FULL_SEED).reset_index(drop=True)

    records = []
    done_ids = set()
    if OUT_PSEUDO.exists():
        prev = pd.read_csv(OUT_PSEUDO)
        records = prev.to_dict("records")
        done_ids = set(prev["customer_tweet_id"])
        print(f"Resuming: {len(done_ids)}/{len(sample)} already classified in {OUT_PSEUDO}")

    start = time.time()
    n_failed = 0
    for i, row in sample.iterrows():
        if row["customer_tweet_id"] in done_ids:
            continue
        text = row["customer_tweet_clean"]
        try:
            result = classify_intent(text, model, tokenizer)
        except ValueError:
            n_failed += 1
            continue

        records.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_tweet_clean": text,
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
                "fraud_override": check_fraud_override(text),
            }
        )

        if len(records) % CHECKPOINT_EVERY == 0:
            _save_pseudo_labels(records)
            elapsed = time.time() - start
            print(f"  classified {len(records)}/{len(sample)} (checkpointed, {elapsed:.0f}s elapsed)")

    _save_pseudo_labels(records)
    print(
        f"Saved {len(records)} pseudo-labels to {OUT_PSEUDO} "
        f"({n_failed} parse failures, {STATS['retries']} retries total, of {STATS['calls']} calls)"
    )


def run_train(balanced: bool = False) -> None:
    df = pd.read_csv(OUT_PSEUDO)
    X = df["customer_tweet_clean"]
    y = df["predicted_intent"]

    class_weight = "balanced" if balanced else None
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    tag = "balanced" if balanced else "unbalanced"
    print(f"Training accuracy ({tag}, on pseudo-labels, n={len(df)}): {acc:.3f}")

    labels = sorted(y.unique())
    cm = confusion_matrix(y, preds, labels=labels)
    short = [label[:14] for label in labels]
    print(f"\nConfusion matrix ({tag}, rows=pseudo-label, cols=predicted):")
    print(" " * 16 + " ".join(f"{s:>14s}" for s in short))
    for label, row_counts in zip(short, cm):
        print(f"{label:<16s}" + " ".join(f"{v:>14d}" for v in row_counts))

    out_path = OUT_MODEL.with_name("baseline_intent_classifier_balanced.joblib") if balanced else OUT_MODEL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\nSaved baseline pipeline (TfidfVectorizer + LogisticRegression, {tag}) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "full", "train"], default="pilot")
    parser.add_argument("--balanced", action="store_true", help="train stage only: class_weight='balanced'")
    args = parser.parse_args()

    if args.stage == "train":
        run_train(balanced=args.balanced)
        return

    model, tokenizer = load_model()
    if args.stage == "pilot":
        run_pilot(model, tokenizer)
    else:
        run_full(model, tokenizer)


if __name__ == "__main__":
    main()
