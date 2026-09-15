"""
Hour 9.5-10.5, Part 3: agreement between human hand-scores and the LLM
judge on the 45-row calibration sample.

Run ONLY after reports/human_judge_sample.csv has been filled in by hand --
not before. Refuses to run if any score cell is still blank.

Usage: .venv/Scripts/python.exe src/compute_judge_agreement.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_HUMAN_SAMPLE = Path("reports/human_judge_sample_hand_scored.csv")
IN_JUDGE_SCORES = Path("reports/llm_judge_scores.csv")
OUT_CALIBRATION = Path("reports/judge_calibration.md")

DIMENSIONS = ["relevance", "groundedness", "tone_match", "no_hallucinated_specifics", "actionability"]


def main() -> None:
    human = pd.read_csv(IN_HUMAN_SAMPLE)
    judge = pd.read_csv(IN_JUDGE_SCORES)

    score_cols = [f"{d}_score" for d in DIMENSIONS]
    missing = human[score_cols].isna().any(axis=1)
    if missing.any():
        raise SystemExit(
            f"{missing.sum()} row(s) in {IN_HUMAN_SAMPLE} still have a blank score -- "
            "fill in all five dimensions for every row before running this."
        )

    merged = human.merge(judge, on="customer_tweet_id", suffixes=("_human", "_llm"))
    print(f"Comparing {len(merged)} rows with both human and LLM-judge scores\n")

    rows = []
    for dim in DIMENSIONS:
        h = merged[f"{dim}_score_human"].astype(float)
        l = merged[f"{dim}_score_llm"].astype(float)
        rho, pval = spearmanr(h, l)
        adjacent_rate = ((h - l).abs() <= 1).mean()
        rows.append(
            {"dimension": dim, "spearman_rho": rho, "spearman_p": pval, "exact_or_adjacent_rate": adjacent_rate}
        )

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    lines = [f"# Judge calibration (human vs. LLM judge, n={len(merged)})\n"]
    lines.append("| Dimension | Spearman rho | p-value | Exact-or-adjacent rate |")
    lines.append("|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['dimension']} | {r['spearman_rho']:.3f} | {r['spearman_p']:.3f} | "
            f"{r['exact_or_adjacent_rate']:.1%} |"
        )

    OUT_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    OUT_CALIBRATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_CALIBRATION}")


if __name__ == "__main__":
    main()
