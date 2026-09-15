"""
Hour 8.5-9.5: compare the trivial baseline, the simple baseline, and the
full system against the hand-labeled golden set. Intent + escalation
metrics only -- reply quality is Hour 9.5-10.5's LLM-judge job, not this
script's, and two of the three methods here don't draft grounded replies
at all, so scoring reply quality here would be apples-to-oranges by
construction.

Usage: .venv/Scripts/python.exe src/run_baselines.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from escalation_policy import HIGH_RISK_INTENTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_GOLDEN = Path("reports/golden_set_candidates_labeled.csv")
IN_PSEUDO = Path("data/processed/pseudo_labels.csv")
IN_SIMPLE_BASELINE_MODEL = Path("data/processed/baseline_intent_classifier.joblib")
OUT_REPORT = Path("reports/baseline_comparison.md")

# Not scored (see module docstring) -- kept only for completeness/documentation.
TRIVIAL_CANNED_REPLY = "Thanks for reaching out -- we're looking into this and will follow up shortly."


def compute_intent_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "intent_accuracy": report["accuracy"],
        "intent_macro_f1": report["macro avg"]["f1-score"],
    }


def compute_escalation_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, labels=["auto", "escalate"], output_dict=True, zero_division=0)
    return {
        "escalation_precision": report["escalate"]["precision"],
        "escalation_recall": report["escalate"]["recall"],
        "escalation_f1": report["escalate"]["f1-score"],
        "escalation_accuracy": accuracy_score(y_true, y_pred),
    }


def run_trivial(golden: pd.DataFrame) -> dict:
    # Majority class from the SIMPLE BASELINE'S OWN TRAINING DISTRIBUTION
    # (pseudo_labels.csv), never from the golden set's true_intent -- using
    # the golden set's own label distribution to pick the majority class
    # would leak test information into the baseline.
    pseudo = pd.read_csv(IN_PSEUDO)
    majority_intent = pseudo["predicted_intent"].value_counts().idxmax()
    print(f"Trivial baseline majority intent (from pseudo_labels.csv): {majority_intent!r}")

    intent_preds = [majority_intent] * len(golden)

    # Fixed "escalate everything": the safer default for a support system
    # when the alternative is a single dumb rule, and it gives a real
    # precision/recall floor to compare the real policy's over-escalation
    # behavior against.
    escalation_preds = ["escalate"] * len(golden)

    metrics = compute_intent_metrics(golden["true_intent"], intent_preds)
    metrics.update(compute_escalation_metrics(golden["true_escalation_decision"], escalation_preds))
    return metrics


def run_simple(golden: pd.DataFrame) -> dict:
    model = joblib.load(IN_SIMPLE_BASELINE_MODEL)
    print(f"Simple baseline model type: {type(model)}")
    # It's a full sklearn Pipeline (TfidfVectorizer + LogisticRegression) --
    # predict directly, no separate vectorizer step needed.
    intent_preds = model.predict(golden["tweet_text"])

    # Reduced, single-tier version of escalation_policy.decide()'s logic:
    # only the "always-escalate high-risk intent" rule, applied to this
    # baseline's OWN predicted intent. Deliberately reduced -- this baseline
    # has no confidence calibration and no retrieval similarity, so faking
    # those inputs to run the full 3-tier policy would misrepresent what a
    # simple intent-only classifier can actually support.
    escalation_preds = ["escalate" if intent in HIGH_RISK_INTENTS else "auto" for intent in intent_preds]

    metrics = compute_intent_metrics(golden["true_intent"], intent_preds)
    metrics.update(compute_escalation_metrics(golden["true_escalation_decision"], escalation_preds))
    return metrics


def run_full_system(golden: pd.DataFrame) -> dict:
    # Already computed by build_golden_set.py -- read, don't regenerate.
    metrics = compute_intent_metrics(golden["true_intent"], golden["predicted_intent"])
    metrics.update(compute_escalation_metrics(golden["true_escalation_decision"], golden["escalation_decision"]))
    return metrics


def write_report(results: dict) -> None:
    cols = [
        ("intent_accuracy", "Intent Accuracy"),
        ("intent_macro_f1", "Intent Macro-F1"),
        ("escalation_precision", "Escalation Precision"),
        ("escalation_recall", "Escalation Recall"),
        ("escalation_f1", "Escalation F1"),
        ("escalation_accuracy", "Escalation Accuracy"),
    ]
    header = "| Method | " + " | ".join(label for _, label in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = ["# Baseline comparison (golden set, n=200)\n", header, sep]
    for name, metrics in results.items():
        row = "| " + name + " | " + " | ".join(f"{metrics[key]:.3f}" for key, _ in cols) + " |"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(f"\nWrote {OUT_REPORT}\n")
    print(text)


def main() -> None:
    golden = pd.read_csv(IN_GOLDEN)
    print(f"Loaded {len(golden)} labeled golden-set rows from {IN_GOLDEN}\n")

    results = {
        "Trivial baseline": run_trivial(golden),
        "Simple baseline": run_simple(golden),
        "Full system": run_full_system(golden),
    }

    write_report(results)


if __name__ == "__main__":
    main()
