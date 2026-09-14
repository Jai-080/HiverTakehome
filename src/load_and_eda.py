"""
Hour 0-0.5: load the raw Twitter CSV, filter to AmazonHelp, reconstruct
(customer_tweet, brand_reply) pairs, and run a quick EDA.

Usage: .venv/Scripts/python.exe src/load_and_eda.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_CSV = Path("data/raw/twcs/twcs.csv")
OUT_PAIRS = Path("data/processed/amazonhelp_pairs.csv")
BRAND = "AmazonHelp"

# Deflection heuristic for AmazonHelp, calibrated on a sample of actual replies:
# classic "DM us" phrasing is RARE here (~0.5%) -- AmazonHelp instead redirects
# off-thread via a link ("report/contact/visit us here: [url]"), present in ~41%
# of replies. Flag either pattern. Note: bare "pm" is deliberately excluded from
# the DM pattern -- it collides with "8 PM" time-of-day mentions.
# This is a rough EDA-stage proxy; the Hour 3.5-5 substance filter should refine
# it further (e.g. some linked replies point to genuinely informative help articles
# rather than an identity-verification gate).
DM_DEFLECTION_RE = re.compile(
    r"\bdm\b.{0,40}\b(us|you|your)\b"
    r"|\b(us|you|your).{0,40}\bdm\b"
    r"|\bdirect message\b"
    r"|\bprivate message\b"
    r"|\bsend (us )?a (direct|private) message\b"
    r"|\bshoot us a dm\b"
    r"|\bfollow (&|and) dm\b"
    r"|https?://t\.co/\S+",
    flags=re.IGNORECASE,
)


def load_raw() -> pd.DataFrame:
    cols = ["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"]
    dtype = {
        "tweet_id": "int64",
        "author_id": "string",
        "inbound": "bool",
        "text": "string",
    }
    df = pd.read_csv(RAW_CSV, usecols=cols, dtype=dtype)
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce").astype("Int64")
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    replies = df[(df["author_id"] == BRAND) & (df["in_response_to_tweet_id"].notna())].copy()

    customer = df[df["inbound"]][["tweet_id", "text", "created_at"]].rename(
        columns={"tweet_id": "customer_tweet_id", "text": "customer_tweet", "created_at": "customer_created_at"}
    )

    pairs = replies.merge(
        customer, left_on="in_response_to_tweet_id", right_on="customer_tweet_id", how="inner"
    )
    pairs = pairs.rename(columns={"tweet_id": "brand_reply_id", "text": "brand_reply", "created_at": "brand_created_at"})
    pairs = pairs[
        ["customer_tweet_id", "customer_tweet", "customer_created_at", "brand_reply_id", "brand_reply", "brand_created_at"]
    ]
    return pairs.reset_index(drop=True)


def run_eda(pairs: pd.DataFrame) -> None:
    n = len(pairs)
    print(f"AmazonHelp (customer_tweet, brand_reply) pairs: {n:,}")

    reply_len = pairs["brand_reply"].str.len()
    print("\nBrand reply length (chars) distribution:")
    print(reply_len.describe().round(1))

    is_deflection = pairs["brand_reply"].str.contains(DM_DEFLECTION_RE, na=False)
    rate = is_deflection.mean()
    print(f"\nDM-deflection rate (regex heuristic): {rate:.1%} ({is_deflection.sum():,} / {n:,})")

    print("\nSample DM-deflection reply:")
    print(pairs.loc[is_deflection, "brand_reply"].iloc[0])
    print("\nSample non-deflection reply:")
    print(pairs.loc[~is_deflection, "brand_reply"].iloc[0])


def main() -> None:
    print(f"Loading {RAW_CSV} ...")
    df = load_raw()
    print(f"Total rows: {len(df):,}")

    pairs = build_pairs(df)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"Saved pairs to {OUT_PAIRS}")

    run_eda(pairs)


if __name__ == "__main__":
    main()
