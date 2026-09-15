"""Surface failure-analysis candidates for hand review. Prints only -- no prose, no file output.

Usage: .venv/Scripts/python.exe src/surface_failure_candidates.py
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

golden = pd.read_csv("reports/golden_set_candidates_labeled.csv")
judge = pd.read_csv("reports/llm_judge_scores.csv")
df = golden.merge(judge, on="customer_tweet_id", how="left")


def show(title: str, rows: pd.DataFrame, extra_cols: list[str], limit: int | None = None):
    print(f"\n=== {title} ({len(rows)} total{f', showing {limit}' if limit else ''}) ===")
    for _, r in rows.head(limit).iterrows() if limit else rows.iterrows():
        print(f"\n[{r['customer_tweet_id']}] {r['tweet_text'][:120]}")
        print(f"  reply: {r['drafted_reply'][:150]}")
        print("  " + " | ".join(f"{c}={r[c]}" for c in extra_cols))


mis = df[df["true_intent"] != df["predicted_intent"]].sort_values("predicted_confidence", ascending=False)
show("Intent misclassifications (highest confidence first)", mis, ["true_intent", "predicted_intent", "predicted_confidence"], 5)

missed_esc = df[(df["true_escalation_decision"] == "escalate") & (df["escalation_decision"] == "auto")]
show("Missed escalations (ALL)", missed_esc, ["true_escalation_decision", "escalation_decision", "escalation_reason"])

over_esc = df[(df["true_escalation_decision"] == "auto") & (df["escalation_decision"] == "escalate")]
show("Over-escalations (sample)", over_esc, ["escalation_reason"], 2)

hard = df[df["hard_case_category"].notna() & (df["hard_case_category"] != "")]
hard_fail = hard[(hard["true_intent"] != hard["predicted_intent"]) | (hard["true_escalation_decision"] != hard["escalation_decision"])]
show("Stress-test failures on planted hard cases", hard_fail, ["hard_case_category", "true_intent", "predicted_intent", "true_escalation_decision", "escalation_decision"], 5)

low_ground = df[(df["groundedness_score"] <= 2) | (df["no_hallucinated_specifics_score"] <= 2)]
show("Low groundedness / hallucination candidates", low_ground, ["groundedness_score", "no_hallucinated_specifics_score"], 5)

flagged = df[df["validate_draft_flags"].notna() & (df["validate_draft_flags"] != "[]")]
show("validate_draft flags (near-verbatim/leakage)", flagged, ["validate_draft_flags"], 5)
