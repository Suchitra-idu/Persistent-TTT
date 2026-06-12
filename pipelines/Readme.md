# arXiv ML Pipeline — Modal

Everything runs on Modal. Nothing touches your local machine except terminal output.

```
Kaggle API  →  Modal Volume   (metadata snapshot, fetched once globally)
arXiv S3    →  Modal Volume   (manifest fetched once; tars streamed per run)
Volume      →  HuggingFace Hub  (push when a run is complete)
```

---

## One-time setup

### 1. Install Modal and authenticate
```bash
pip install modal
modal setup
```

### 2. Create Modal Secrets (dashboard at modal.com/secrets)

**aws-arxiv**
```
AWS_ACCESS_KEY_ID      = ...
AWS_SECRET_ACCESS_KEY  = ...
AWS_DEFAULT_REGION     = us-east-1
```

**kaggle**
```
KAGGLE_USERNAME  = your-kaggle-username
KAGGLE_KEY       = your-kaggle-api-key
```
Get your Kaggle API key from https://www.kaggle.com/settings → API → Create New Token

**huggingface** (only needed for --push-to-hub)
```
HF_TOKEN = hf_...
```

### 3. Fetch Kaggle metadata onto the Volume (once per workspace, ~10 min, ~4 GB)
```bash
modal run pipeline.py --fetch-metadata
```

Downloads the arXiv metadata snapshot from Kaggle directly into the Modal Volume.
Only needed once. Every subsequent pipeline run reuses this file.

---

## Running the pipeline

```bash
# 10k papers, Jan 2024 to Dec 2025 (defaults)
modal run pipeline.py

# Explicit parameters
modal run pipeline.py --papers 10000 --start 2401 --end 2512

# Run again — completely fresh 10k, same pool, same first-N ordering
modal run pipeline.py --papers 10000 --start 2401 --end 2512

# Different date range
modal run pipeline.py --papers 10000 --start 2301 --end 2312

# Scale up
modal run pipeline.py --papers 20000 --start 2401 --end 2512

# Detach so your terminal can close (run continues on Modal)
modal run --detach pipeline.py --papers 10000
```

---

## Resuming a failed run

If the pipeline crashes or is interrupted mid-way, resume from where it stopped:

```bash
# Find the run_id of the interrupted run
modal run pipeline.py --list-runs

# Resume — skips all steps that already completed
modal run pipeline.py --resume --run-id-arg a3f1bc92
```

Resume behaviour per step:

| Step | Resume behaviour |
|---|---|
| 0 Manifest | Always runs, idempotent — skips download if file present |
| 1 Filter metadata | Fully skipped if `paper_ids.txt` exists |
| 2 Build tar queue | Fully skipped if `tar_queue.jsonl` exists |
| 3 Download + extract | Partially skipped — only re-downloads tars whose papers are not yet in `extracted/` |
| 4 Clean LaTeX | Partially skipped — only re-runs batches with uncleaned papers; already-cleaned `.txt` files are never re-processed |
| 5 Build HF dataset | Fully skipped if `hf_dataset/dataset_info.json` exists |

The `--papers`, `--start`, and `--end` values from the resume command are used for
any steps that still need to run. Steps that are skipped use whatever was written
in the previous attempt.

---

## Managing runs

```bash
# List all runs with status, paper count, token estimate, date range
modal run pipeline.py --list-runs

# Push a completed run's dataset to HuggingFace Hub (runs entirely on Modal)
modal run pipeline.py --push-to-hub --run-id-arg a3f1bc92 --hf-repo your-username/arxiv-ml-10k

# Inspect the Volume interactively
modal shell --volume arxiv-pipeline-vol
# Inside the shell:
#   ls /data/runs/
#   ls /data/runs/a3f1bc92/
#   wc -l /data/runs/a3f1bc92/paper_ids.txt
```

---

## Date format

YYMM — e.g. `2401` = January 2024, `2512` = December 2025

---

## How run isolation works

Each run gets a fresh 8-char UUID (e.g. `a3f1bc92`). All data for that run lives
under `/data/runs/{run_id}/` on the Volume. Two runs with identical parameters
produce identical datasets (same papers, same ordering). To get different papers,
change the date range.

```
/data/
  arXiv_src_manifest.xml                  shared, fetched once
  arxiv-metadata-oai-snapshot.json        shared, fetched once
  runs/
    a3f1bc92/
      meta.json           run parameters and status
      paper_ids.txt       matched arXiv IDs for this run
      tar_queue.jsonl     S3 tars to download, sorted by density
      extracted/          raw LaTeX source files per paper
      cleaned/            plain text per paper
      hf_dataset/         final HuggingFace Dataset (arrow format)
    b7e2df11/             completely separate run
      ...
```

---

## Volume version

The pipeline uses Modal Volumes v2, which supports hundreds of concurrent writers.
This is required for step 4 (LaTeX cleaning), where ~200 containers write
simultaneously. If you encounter a "Volume not found" error on first run, create
the Volume manually first:

```bash
modal volume create --version=2 arxiv-pipeline-vol
```

---

## Cost estimate (10k papers, Jan 2024 – Dec 2025)

| Item | Cost |
|---|---|
| S3 tar downloads | ~$1–2 |
| Modal CPU compute | ~$1–3 |
| Volume storage | ~$0.10/GB/month |
| Total per run | under $5 |