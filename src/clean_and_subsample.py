"""
Hour 0.5-1.5: clean the AmazonHelp pairs, filter to English, dedupe,
and split into a working set + held-out pool.

Usage: .venv/Scripts/python.exe src/clean_and_subsample.py
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# langdetect samples n-grams internally with Python's global random state,
# so results are non-deterministic run-to-run unless seeded.
DetectorFactory.seed = 0

IN_PAIRS = Path("data/processed/amazonhelp_pairs.csv")
OUT_WORKING = Path("data/processed/amazonhelp_working.csv")
OUT_HELDOUT = Path("data/processed/amazonhelp_heldout.csv")

WORKING_N = 9_000
HELDOUT_N = 4_000
RANDOM_STATE = 42
REPLY_TEMPLATE_CAP = 5

MENTION_RE = re.compile(r"@\w+")
URL_REPLACE_RE = re.compile(r"https?://\S+")
URL_PLACEHOLDER = "<URL>"

# Email before phone: an email local-part with digits (e.g. "user12345678@gmail.com")
# would otherwise get partially mangled by the phone pattern first.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
EMAIL_PLACEHOLDER = "<EMAIL>"

# Matches runs of 9+ digits (with optional separators), not bounded by other digits.
# Heuristic, not exact: will also catch long order/tracking/AWB numbers as a false
# positive. Accepted trade-off -- PII takes priority, and those account-specific
# numbers shouldn't end up baked into reply templates for the RAG step either.
PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d\-.\s]{7,}\d(?!\d)")
PHONE_PLACEHOLDER = "<PHONE>"

# Identical to the URL branch inside load_and_eda.DM_DEFLECTION_RE (src/load_and_eda.py,
# the `https?://t\.co/\S+` alternative) -- kept as a separate constant here rather than
# imported, since that regex there also matches DM-language, not just links. Trade-off:
# this buys an independent has_link flag but the two patterns can now drift if one is
# edited without the other -- no shared source of truth enforces they stay identical.
HAS_LINK_RE = re.compile(r"https?://t\.co/\S+", flags=re.IGNORECASE)


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    text = PHONE_RE.sub(PHONE_PLACEHOLDER, text)
    return text


def clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = redact_pii(text)
    text = URL_REPLACE_RE.sub(URL_PLACEHOLDER, text)
    text = MENTION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_pii_redacted(raw_series: pd.Series) -> int:
    unescaped = raw_series.map(html.unescape)
    redacted = unescaped.map(redact_pii)
    return int((redacted != unescaped).sum())


def filter_first_touch(pairs: pd.DataFrame) -> pd.DataFrame:
    n_before = len(pairs)
    kept = pairs[pairs["customer_in_response_to_tweet_id"].isna()].copy()
    print(
        f"First-touch filter: {n_before - len(kept):,} rows removed "
        f"(customer tweet was itself a reply), {len(kept):,} kept"
    )
    return kept


def add_clean_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["has_link"] = pairs["brand_reply"].str.contains(HAS_LINK_RE, na=False)

    n_customer_pii = count_pii_redacted(pairs["customer_tweet"])
    n_brand_pii = count_pii_redacted(pairs["brand_reply"])
    print(f"PII redaction: {n_customer_pii:,} customer_tweet rows affected, {n_brand_pii:,} brand_reply rows affected")

    pairs["customer_tweet_clean"] = pairs["customer_tweet"].map(clean_text)
    pairs["brand_reply_clean"] = pairs["brand_reply"].map(clean_text)
    return pairs


def detect_lang(text: str) -> str | None:
    try:
        return detect(text)
    except Exception:
        return None


def filter_english(pairs: pd.DataFrame) -> pd.DataFrame:
    n_before = len(pairs)
    langs = pairs["customer_tweet_clean"].map(detect_lang)

    n_errored = langs.isna().sum()
    is_en = langs == "en"
    n_non_en = (~is_en & langs.notna()).sum()

    kept = pairs.loc[is_en].copy()
    print(
        f"Language filter: {n_errored:,} rows errored (dropped), "
        f"{n_non_en:,} rows non-English (dropped), "
        f"{n_before - len(kept):,} total removed, {len(kept):,} kept"
    )
    return kept


def dedupe(pairs: pd.DataFrame) -> pd.DataFrame:
    n_before = len(pairs)
    pairs = pairs.drop_duplicates(subset="customer_tweet_clean", keep="first").reset_index(drop=True)
    print(f"Exact-duplicate customer_tweet_clean drop: {n_before - len(pairs):,} rows removed, {len(pairs):,} kept")

    n_unique_templates = pairs["brand_reply_clean"].nunique()
    n_before_cap = len(pairs)
    rank_within_template = pairs.groupby("brand_reply_clean").cumcount()
    pairs = pairs.loc[rank_within_template < REPLY_TEMPLATE_CAP].reset_index(drop=True)
    print(
        f"Reply-template cap (max {REPLY_TEMPLATE_CAP}/template): "
        f"{n_unique_templates:,} unique brand_reply_clean templates before cap, "
        f"{n_before_cap - len(pairs):,} rows removed, {len(pairs):,} kept"
    )
    return pairs


def split_working_heldout(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = pairs.sample(n=WORKING_N, random_state=RANDOM_STATE)
    remaining = pairs.drop(working.index)
    heldout = remaining.sample(n=HELDOUT_N, random_state=RANDOM_STATE)

    assert set(working.index).isdisjoint(set(heldout.index)), "working/heldout index overlap!"
    return working, heldout


def sanity_report(full: pd.DataFrame, working: pd.DataFrame) -> None:
    print("\nSanity check: working set vs. full cleaned population")
    for name, df in [("full", full), ("working", working)]:
        reply_len = df["brand_reply_clean"].str.len()
        link_rate = df["has_link"].mean()
        print(
            f"  {name:8s} n={len(df):6,}  reply_len mean={reply_len.mean():6.1f}  "
            f"median={reply_len.median():6.1f}  has_link_rate={link_rate:.1%}"
        )


def main() -> None:
    print(f"Loading {IN_PAIRS} ...")
    pairs = pd.read_csv(IN_PAIRS)
    print(f"Input rows: {len(pairs):,}")

    pairs = filter_first_touch(pairs)
    pairs = add_clean_columns(pairs)
    pairs = filter_english(pairs)
    pairs = dedupe(pairs)

    working, heldout = split_working_heldout(pairs)

    OUT_WORKING.parent.mkdir(parents=True, exist_ok=True)
    working.to_csv(OUT_WORKING, index=False)
    heldout.to_csv(OUT_HELDOUT, index=False)
    print(f"\nSaved working set ({len(working):,} rows) to {OUT_WORKING}")
    print(f"Saved held-out pool ({len(heldout):,} rows) to {OUT_HELDOUT}")

    sanity_report(pairs, working)

    size_mb = OUT_WORKING.stat().st_size / (1024 * 1024)
    print(f"\n{OUT_WORKING} size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
