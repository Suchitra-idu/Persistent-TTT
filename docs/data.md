# Dataset and Data Pipeline

Reference for [`data_utils.py`](../data_utils.py), the tokenization
pipeline in `train_modal.py`, and the loss-mask reference.

## Dataset selection (`DATASET_SPEC`)

Dataset identity is captured in a `DatasetSpec` dataclass. The active
spec is chosen by the `TTT_DATASET` env var (default `"arxiv"`) and
looked up in `DATASETS` at import time:

```python
DATASET_NAME = os.environ.get("TTT_DATASET", "arxiv")
DATASET_SPEC = get_dataset_spec(DATASET_NAME)
```

Everything downstream — `open_dataset`, `split_holdout`,
`apply_source_filter`, tokenization in `train_modal.py`,
`fetch_holdout_texts` in `infer_modal.py` — goes through `DATASET_SPEC`
rather than any individual column name. Switching datasets is one
env var, not a code change.

### The DatasetSpec fields

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str                                # short id (e.g. "arxiv")
    source: str                              # HF repo id or local dir
    text_column: str = "text"                # doc-text column
    tokens_est_column: str | None = None     # optional cheap pre-filter
    source_meta_column: str | None = None    # e.g. "meta"
    source_meta_key: str | None = None       # e.g. "redpajama_set_name"
    include_sources: tuple | None = None     # allow-list on extracted source
    holdout_last_n: int = 200                # newest-N reserved for eval
```

- **`text_column`** — the column that carries the raw doc text.
- **`tokens_est_column`**, when set and present in the dataset, lets
  the loader do a cheap pre-filter on token counts before tokenization
  (arxiv ships this precomputed; SlimPajama does not).
- **`source_meta_column` + `source_meta_key`**, when both set, extract
  a per-row source label into a top-level `source` column via
  `_annotate_source`. Datasets like SlimPajama carry the RedPajama
  subset name inside a `meta` struct — this is how that label surfaces
  at the row level.
- **`include_sources`**, when set, keeps only rows whose extracted
  source label is in this tuple. This is the mechanism by which we
  exclude CommonCrawl on SlimPajama.
- **`holdout_last_n`** — how many trailing rows (newest first) to
  reserve for eval.

### Registered datasets

Two ship in `DATASETS`:

| name             | source                       | text col | source label   | holdout | notes |
|------------------|------------------------------|----------|----------------|---------|-------|
| `arxiv`          | `suchitraIdu/arxiv-ml-16k`   | `text`   | —              | 200     | Original ML paper corpus. `tokens_est` present for cheap pre-filter. |
| `slimpajama-6b`  | `DKYoon/SlimPajama-6B`       | `text`   | `meta.redpajama_set_name` | 1000 | Diverse pretraining mix (2.5M docs after excluding CommonCrawl). |

The SlimPajama spec explicitly excludes `RedPajamaCommonCrawl` via
`include_sources`, keeping the C4 / Github / Book / ArXiv / Wikipedia /
StackExchange split. That drops ~54% of rows but leaves a diverse mix
that lets you characterize which document domains benefit most from
TTT.

### Adding a new dataset

Add an entry to `DATASETS`:

```python
"my-corpus": DatasetSpec(
    name="my-corpus",
    source="myorg/my-corpus",
    text_column="body",
    source_meta_column="metadata",
    source_meta_key="domain",
    include_sources=("papers", "manuals", "docs"),
    holdout_last_n=500,
),
```

Then run with `TTT_DATASET=my-corpus modal run --detach train_modal.py::train`.
No other code changes required. Tests should exercise
`extract_source_from_row` for the new meta shape if it's not the
common struct-or-JSON-string form.

### Private repos

Set the `HF_TOKEN` secret in Modal
(`modal secret create huggingface HF_TOKEN=hf_...`) and attach it to
both `train_modal.py` and `infer_modal.py` app functions. This is
already wired in the current apps.

## Load path (`open_dataset` → filter → shuffle → cap → tokenize)

`load_token_dataset` in [`train_modal.py`](../train_modal.py) is the
canonical path:

1. **`open_dataset(spec)`** — HF Hub or local parquet/arrow. If the
   spec declares `source_meta_column`, a `source` column is added
   here via `_annotate_source`.
2. **`split_holdout(ds, spec)`** — reserves the newest `holdout_last_n`
   rows.
3. **`apply_source_filter(ds, spec)`** — keeps only allow-listed
   sources. No-op when `include_sources` is unset.
4. **Deterministic shuffle** — `_shuffle_by_index(ds, seed=cfg.seed)`
   permutes row order via `.select(shuffled_indices)`. Runs *before*
   `--limit-docs` and *before* the in-corpus unigram count for the
   loss mask. Rationale:
   - `--limit-docs N` samples uniformly across sources instead of the
     natural head (SlimPajama ships grouped by source, so an
     unshuffled head is all-C4).
   - Interrupting mid-epoch leaves every source touched roughly
     proportionally.
   - In-corpus loss-mask counts reflect the whole training mix.
5. **Pre-filter by `tokens_est`** (if the spec declares it and the
   column is present). Skips docs shorter than `min_doc_tokens` before
   the expensive tokenization step.
6. **`--limit-docs`** cap, applied after shuffle so it's a uniform
   sample.
7. **Tokenize** with the model's tokenizer (`truncation=True,
   max_length=max_seq_len`). The `source` column is preserved through
   tokenization so downstream code can log per-source item counts.
8. **Drop-short filter (exact).** Post-tokenization double-check
   against `min_doc_tokens`, since `tokens_est` was only an estimate.

Result: a `datasets.Dataset` with an `input_ids` column (and `source`
when the spec carries one). Memory-mapped for cheap iteration.

## Holdout split (`split_holdout`)

```python
def split_holdout(ds, spec=None):
    spec = spec or DATASET_SPEC
    cut = len(ds) - spec.holdout_last_n
    return ds.select(range(cut)), ds.select(range(cut, len(ds)))
```

- **Train:** everything except the newest `spec.holdout_last_n` rows.
- **Holdout:** the newest rows.

For arxiv (rows ordered by arxiv id, a proxy for time) this gives a
"future-holdout" property. For SlimPajama the natural order isn't
temporal, so "holdout" here just means "a deterministic
tail-of-dataset slice reserved from training" — the guarantee is
contamination-freedom, not future-vs-past.

The split is deterministic and shared between train and inference,
so slow weights are guaranteed never to have seen holdout papers.

## Source labels and per-source eval

When the spec declares a source field, every row carries a `source`
column all the way through the pipeline. Three downstream users:

1. **Eval holdout sampling.** `fetch_holdout_papers_ids` in
   `train_modal.py`:
   - Filters holdout to docs with `>= cfg.eval_min_tokens` tokens
     (uses `tokens_est_column` when present; falls back to
     `len(text) / 4` char proxy).
   - When `cfg.eval_n_papers_per_source > 0`: picks exactly
     `eval_n_papers_per_source` papers per source (total =
     `n_sources * eval_n_papers_per_source`). Every domain
     represented every eval — required for per-source metrics to be
     non-noisy.
   - Otherwise: stratified round-robin one-per-source until
     `eval_n_papers` is hit.
2. **`fetch_holdout_texts`** in `infer_modal.py` returns
   `[{"text": ..., "source": ...}, ...]` so downstream local
   entrypoints can label papers by source.
3. **Per-source metrics.** `run_holdout_eval` accepts a parallel
   `paper_sources` list and emits `eval/<source>/carry_ppl`,
   `eval/<source>/fresh_ppl`, `eval/<source>/gap`,
   `eval/<source>/n_papers` in addition to the aggregate
   `eval/carry_ppl` etc. Inference reports the same thing in a
   printed table via `_print_per_source_summary`.

This is the mechanism behind the "which document domains benefit most
from TTT" characterization — see
[training.md](training.md#per-source-eval).

## The vocab-size wrinkle

Qwen3 pads the embedding for hardware efficiency:
- `tokenizer.vocab_size` = 151669 (actual BPE entries)
- `model.config.vocab_size` = 151936 (padded embedding size)

**Loss-mask reference is built against the padded size** in
`build_reference_counts`:
```python
from transformers import AutoConfig
vocab_size = AutoConfig.from_pretrained(BASE_MODEL).vocab_size
```

If this mismatched (using tokenizer.vocab_size), `load_reference_counts`
would raise on the size check at training start. See
[`train_modal.py`](../train_modal.py) `build_reference_counts` and
[`train_utils.py`](../train_utils.py) `load_reference_counts`.

## Session-level structure

Within a training epoch, papers are grouped into sessions with
carry threaded through. See [training.md](training.md#session-structure).
For diverse-length pretraining mixes (SlimPajama), use the **hybrid
session mode** (`--mode hybrid`) described in
[training.md](training.md#hybrid---for-diverse-length-pretraining-mixes).

## Holdout sampling

At eval time, `TTTInference.fetch_holdout_texts(n_papers, seed)`
returns a list of dicts `[{"text": str, "source": str}, ...]`.
`source` is `""` when the active dataset has no source column.

Deterministic given the seed. Different seeds sample different papers
from the same fixed holdout pool.

`--n-papers > holdout_last_n` is capped silently (no error). If you
want to eval on more papers than your holdout has, raise
`holdout_last_n` in the spec (this changes the train/eval split — a
prior training's clean-holdout property is invalidated when the
constant moves).

## Loss-mask reference dataset

Built via `build_reference_counts` in
[`train_modal.py`](../train_modal.py). Default source:
`Salesforce/wikitext / wikitext-103-raw-v1 / train`.

- 1.8M rows of general English text
- ~124M tokens (Qwen3 BPE)
- ~62k unique token ids observed

Why wikitext-103: general English is a good "common word" baseline
that doesn't over-mask domain content. In-corpus counts (fallback)
would mask ML-glue words like "model," "training," "layer" because
they're common in ML papers, defeating the purpose.

Rebuild after any tokenizer change (i.e., changing
`TTT_MODEL_SIZE`). See [training.md](training.md#loss-mask) for
details.

## Related docs

- [training.md](training.md) — how data flows into the loop
- [config.md](config.md) — every dataset/session knob with defaults
- [inference.md](inference.md) — `fetch_holdout_texts`, per-source table
