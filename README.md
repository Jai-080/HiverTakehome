# Hiver SDE Intern Take-Home — AmazonHelp Support Triage

Brand: **AmazonHelp**. Intent classification + retrieval-grounded reply drafting + escalation policy, evaluated against a 200-row hand-labeled golden set.

See `reports/report.md` for the full writeup, `DECISIONS.md` for the decision log, `reports/intent_taxonomy.md` for the taxonomy, `reports/failure_analysis.md` for the five failure modes.

## Final architecture

The original plan (below) assumed Gemini for everything. That changed mid-project: Gemini's free-tier `embed_content` (1,000/day) and `generate_content` (20/day, 5/min) quotas were both exhausted/confirmed-too-small before the pipeline reached meaningful scale — see `DECISIONS.md` for the exact numbers. The system now runs entirely on local models, no cloud API, no rate limits, no `.env`/API key required to reproduce:

- **Embeddings (clustering, RAG retrieval, query encoding):** `BAAI/bge-large-en-v1.5` via `sentence-transformers`.
- **Classification, reply drafting, and LLM-judging:** `microsoft/Phi-3-mini-4k-instruct`, 4-bit quantized via `bitsandbytes`, GPU-accelerated where available.
- **Intent taxonomy:** 8 classifier intents + 1 keyword-based fraud/security override, discovered via two independent clustering runs (Gemini embeddings, then local bge-large) and finalized by manual read, not silhouette score — see `reports/intent_taxonomy.md`.
- **Escalation policy:** three-tier deterministic `decide()` — fraud override → always-escalate on 4 high-risk intents → `validate_draft` safety-check failures → confidence+retrieval-similarity gate, thresholds set from the 25th percentile of real observed score distributions.
- **Judge ≠ generator model family** was the original goal but became infeasible after the Gemini pivot (no second local model fit the hardware/budget). Mitigated with a distinctly-worded judge persona and different decoding (temperature=0.3 sampling vs. the generator's greedy decoding), and the risk is measured, not just asserted — see `reports/judge_calibration.md` and Section 4 of `reports/report.md`.
- **Golden set sampled from a held-out pool**, never touched by taxonomy/RAG-corpus/classifier-training — 200 rows: 160 stratified across the 8 intents + 40 deliberately oversampled hard cases (fraud keywords, near-verbatim retrieval stress, billing/delivery vocabulary-overlap stress, low retrieval similarity).

## Reproduction

Two paths, depending on what you want to check.

### Quick: verify the reported numbers (~2 minutes, no GPU, no model downloads)

Reproduces the exact headline metrics in `reports/baseline_comparison.md` and `reports/judge_calibration.md` from the already-committed golden-set labels and LLM outputs — no re-running of the (multi-hour, GPU-bound) generation pipeline. Installs only what these two scripts actually need, not the full (torch-heavy) `requirements.txt` — that's the Deep path below.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install pandas scikit-learn scipy joblib

python src/run_baselines.py             # -> reports/baseline_comparison.md
python src/compute_judge_agreement.py   # -> reports/judge_calibration.md
```

Both scripts read only committed files: `data/processed/pseudo_labels.csv`, `data/processed/baseline_intent_classifier.joblib`, `reports/golden_set_candidates_labeled.csv`, `reports/human_judge_sample_hand_scored.csv`, `reports/llm_judge_scores.csv`. No `.env`, no API key.

### Deep: re-run pipeline stages on the committed subsample

`data/processed/amazonhelp_working.csv` (9,000 rows) and `amazonhelp_heldout.csv` (4,000 rows) are committed directly — the outputs of the full raw-data cleaning pipeline — so you can re-run any downstream stage without the 493MB Kaggle raw CSV. First run downloads two local models (`~/.cache/huggingface`): `bge-large-en-v1.5` (~1.3GB) and `Phi-3-mini-4k-instruct` (~7.6GB full-precision weights, quantized to 4-bit at load time, ~2.5GB VRAM). **CUDA is not hard-required** — every script falls back to CPU (`torch.cuda.is_available()` check) — but CPU generation is dramatically slower than the GPU numbers below; a GPU is strongly recommended for anything beyond a handful of rows.

`pip install -r requirements.txt` installs the **CPU-only** `torch` wheel by default (confirmed: `2.14.0+cpu`, `cuda.is_available()==False` straight off PyPI) — that's what happened during this project too. For GPU, reinstall after: `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu126` (match the `cu1xx` tag to what `nvidia-smi`'s "CUDA Version" reports as supported, not necessarily `cu126`).

Measured timings (RTX 3050 6GB laptop GPU, this machine):

| Script | What it does | Time |
|---|---|---|
| `src/discover_intents.py --k 8` | Embed 1,500 tweets, cluster, export | ~5 min |
| `src/classify_intent.py --stage full` | Classify 700 tweets (pseudo-labels) | ~65 min (~5.5s/tweet) |
| `src/build_rag_index.py` | Embed the substance-filtered RAG corpus (4,228 rows) | ~30s |
| `src/build_golden_set.py` | Sample + run full pipeline on 200 golden-set rows | ~35 min |
| `src/eval_harness.py --stage judge` | LLM-judge 200 drafted replies | ~35-40 min |

Full order, from the committed subsample (skips `load_and_eda.py`/`clean_and_subsample.py`, which need the raw Kaggle CSV — see below if you want to regenerate the subsample itself):

```bash
python src/discover_intents.py --k 8          # reports/intent_clusters_raw.md
python src/classify_intent.py --stage full    # data/processed/pseudo_labels.csv
python src/classify_intent.py --stage train --balanced
python src/build_rag_index.py                 # data/processed/rag_corpus*.{csv,npy}
python src/draft_reply.py                     # 15-tweet drafting sanity test
python src/escalation_policy.py --distributions --test
python src/build_golden_set.py                # reports/golden_set_candidates.csv (UNLABELED -- hand review needed for true_intent/true_escalation_decision before run_baselines.py's numbers mean anything)
python src/run_baselines.py
python src/eval_harness.py --stage judge       # reports/llm_judge_scores.csv
python src/eval_harness.py --stage sample      # reports/human_judge_sample.csv (blank -- needs hand-scoring before compute_judge_agreement.py)
python src/compute_judge_agreement.py
```

Two steps in that chain need a human in the loop (`golden_set_candidates.csv` → `true_intent`/`true_escalation_decision`; `human_judge_sample.csv` → the five score columns) — the committed `*_labeled.csv`/`*_hand_scored.csv` files are that human input already done, which is what the Quick path above uses instead.

### Regenerating the subsample from raw data (optional, needs the Kaggle dataset)

Only needed to rebuild `amazonhelp_working.csv`/`amazonhelp_heldout.csv` themselves. Download the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset from Kaggle, extract `twcs/twcs.csv` into `data/raw/twcs/twcs.csv`, then:

```bash
python src/load_and_eda.py          # data/processed/amazonhelp_pairs.csv (168,814 pairs) + EDA
python src/clean_and_subsample.py   # data/processed/amazonhelp_working.csv + amazonhelp_heldout.csv
```

