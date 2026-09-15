"""
Hour 1.5-2.5: discover an intent taxonomy from a sample of AmazonHelp
customer tweets via local sentence-transformers embeddings + KMeans
clustering.

Switched from Gemini API embeddings to a local model after the free-tier
embed_content daily quota (1000/day) was exhausted -- see DECISIONS.md.

Usage:
  .venv/Scripts/python.exe src/discover_intents.py           # embed (or load cache), print silhouette scores for k in [5..10], stop
  .venv/Scripts/python.exe src/discover_intents.py --k 7     # cluster with k=7, print + write reports/intent_clusters_raw.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN_WORKING = Path("data/processed/amazonhelp_working.csv")
EMBED_CACHE = Path("data/processed/intent_sample_embeddings_local.npy")
OUT_REPORT = Path("reports/intent_clusters_raw.md")

SAMPLE_N = 1500
SAMPLE_RANDOM_STATE = 42
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_BATCH_SIZE = 32
K_CANDIDATES = [5, 6, 7, 8, 9, 10]
N_EXAMPLES_PER_CLUSTER = 8
N_TERMS_PER_CLUSTER = 15

# "<URL>" placeholder from clean_and_subsample.py tokenizes to "url" under the
# default TF-IDF token pattern -- drop it so it doesn't dominate link-heavy clusters.
STOP_WORDS = list(ENGLISH_STOP_WORDS) + ["url"]


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(IN_WORKING)
    return df.sample(n=SAMPLE_N, random_state=SAMPLE_RANDOM_STATE).reset_index(drop=True)


def get_embeddings(texts: list[str]) -> np.ndarray:
    if EMBED_CACHE.exists():
        arr = np.load(EMBED_CACHE)
        if len(arr) == len(texts):
            print(f"Loaded cached embeddings from {EMBED_CACHE} ({arr.shape})")
            return arr
        print(f"Cache at {EMBED_CACHE} has {len(arr)} rows, expected {len(texts)} -- re-embedding")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()} -- using device={device}")

    print(f"Loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # No query/passage instruction prefix here -- that's a bge convention for
    # asymmetric retrieval (query vs. passage), not relevant to clustering.
    arr = model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_CACHE, arr)
    print(f"Saved embeddings to {EMBED_CACHE} ({arr.shape})")
    return arr


def scan_k(embeddings: np.ndarray) -> None:
    print("\nSilhouette scores by k:")
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k:2d}  silhouette={score:.4f}")


def cluster_terms(texts: pd.Series, labels: np.ndarray, k: int) -> list[list[str]]:
    cluster_docs = [" ".join(texts[labels == c]) for c in range(k)]
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, max_features=5000)
    tfidf = vectorizer.fit_transform(cluster_docs)
    terms = np.array(vectorizer.get_feature_names_out())

    top_terms = []
    for c in range(k):
        row = tfidf[c].toarray().ravel()
        top_idx = row.argsort()[::-1][:N_TERMS_PER_CLUSTER]
        top_terms.append(list(terms[top_idx]))
    return top_terms


def write_report(df: pd.DataFrame, labels: np.ndarray, k: int, top_terms: list[list[str]]) -> None:
    lines = [f"# Intent clusters (raw), k={k}\n"]
    for c in range(k):
        cluster_df = df[labels == c]
        examples = cluster_df["customer_tweet_clean"].sample(
            n=min(N_EXAMPLES_PER_CLUSTER, len(cluster_df)), random_state=42
        )
        lines.append(f"## Cluster {c} (n={len(cluster_df)})\n")
        lines.append(f"**Top TF-IDF terms:** {', '.join(top_terms[c])}\n")
        lines.append("**Example tweets:**")
        for ex in examples:
            lines.append(f"- {ex}")
        lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_REPORT}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="cluster count to run step 3 with")
    args = parser.parse_args()

    print(f"Loading sample from {IN_WORKING} (n={SAMPLE_N}, random_state={SAMPLE_RANDOM_STATE})")
    df = load_sample()
    texts = df["customer_tweet_clean"].tolist()

    embeddings = get_embeddings(texts)  # already L2-normalized by encode(normalize_embeddings=True)

    scan_k(embeddings)

    if args.k is None:
        print("\nNo --k given -- stopping here. Re-run with --k <n> once you've picked one.")
        return

    if args.k not in K_CANDIDATES:
        print(f"\nWarning: k={args.k} wasn't in the scanned candidates {K_CANDIDATES}, running it anyway.")

    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    top_terms = cluster_terms(df["customer_tweet_clean"], labels, args.k)
    write_report(df, labels, args.k, top_terms)


if __name__ == "__main__":
    main()
