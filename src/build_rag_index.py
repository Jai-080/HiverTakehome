"""
Hour 3.5-5: build the RAG retrieval index -- embed substance-filtered
historical (customer_tweet, brand_reply) pairs from the working set, for
same-intent retrieval when drafting new replies.

Substance filter (reuses clean_and_subsample.py's own columns, not
recomputed differently): a reply counts as "substantive" -- usable as a
grounding example -- only if it (a) has no link (has_link == False; a link
is AmazonHelp's dominant deflection pattern, ~41-52% of replies, see
DECISIONS.md / load_and_eda.py's DM_DEFLECTION_RE) and (b) is at least
MIN_REPLY_LENGTH characters, excluding near-empty acknowledgments
("Thanks!", "Sorry about that.").

NOT YET RUN -- Phi-3-mini is occupying the GPU for classify_intent.py's
full pseudo-labeling pass. Fire this once that finishes and frees the GPU;
running both at once on a 6GB card risks an OOM crash for both jobs.

Usage: .venv/Scripts/python.exe src/build_rag_index.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_WORKING = Path("data/processed/amazonhelp_working.csv")
OUT_EMBEDDINGS = Path("data/processed/rag_corpus_embeddings.npy")
OUT_CORPUS = Path("data/processed/rag_corpus.csv")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_BATCH_SIZE = 32
MIN_REPLY_LENGTH = 40  # chars -- excludes near-empty acknowledgments like "Thanks!"


def substance_filter(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    reply_len = df["brand_reply_clean"].str.len()
    kept = df[(~df["has_link"]) & (reply_len >= MIN_REPLY_LENGTH)].reset_index(drop=True)
    print(
        f"Substance filter (has_link == False AND reply length >= {MIN_REPLY_LENGTH} chars): "
        f"{n_before - len(kept):,} rows removed, {len(kept):,} kept (of {n_before:,})"
    )
    return kept


def build_index() -> None:
    print(f"Loading {IN_WORKING} ...")
    df = pd.read_csv(IN_WORKING)
    print(f"Input rows: {len(df):,}")

    corpus = substance_filter(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()} -- using device={device}")
    print(f"Loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # Corpus side of asymmetric retrieval -- indexed as passages, left
    # unprefixed. The bge query-instruction prefix ("Represent this sentence
    # for searching relevant passages: ") belongs on the *new* customer
    # tweet at actual retrieval time (a later script), not on this indexed
    # corpus -- same bge asymmetric-retrieval convention noted in
    # discover_intents.py, which didn't need it since that was symmetric
    # clustering; this is the retrieval use case it was flagged for.
    texts = corpus["customer_tweet_clean"].tolist()
    embeddings = model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    OUT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBEDDINGS, embeddings)
    corpus.to_csv(OUT_CORPUS, index=False)
    print(f"\nSaved {len(corpus):,} embeddings ({embeddings.shape}) to {OUT_EMBEDDINGS}")
    print(f"Saved corpus metadata to {OUT_CORPUS}")


if __name__ == "__main__":
    build_index()
