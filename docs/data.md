# Dataset and Data Pipeline

Reference for [`data_utils.py`](../data_utils.py), the tokenization
pipeline in `train_modal.py`, and the loss-mask reference.

## Dataset

`DATASET_SOURCE` in [`ttt_config.py`](../ttt_config.py) points at an
HF Hub dataset repo. Default: `"suchitraIdu/arxiv-ml-16k"`.

- ~22,500 arxiv ML papers, sorted by arxiv ID (proxy for time).
- Columns:
  - `text` — cleaned paper text (LaTeX artifacts partially stripped)
  - `tokens_est` — pre-computed token count estimate (cheap filter)

The dataset is downloaded once to the `hf-hub-cache` Modal volume.
Subsequent runs reuse the cache (subject to HF datasets' cache
freshness heuristics — it uses "latest cached configuration" if the
Hub is unreachable).

**Private repos:** set the `HF_TOKEN` secret in Modal
(`modal secret create huggingface HF_TOKEN=hf_...`) and attach it to
both `train_modal.py` and `infer_modal.py` app functions.

## Holdout split (`split_holdout`)

```python
def split_holdout(ds):
    cut = len(ds) - HOLDOUT_LAST_N
    return ds.select(range(cut)), ds.select(range(cut, len(ds)))
```

- **Train:** everything except the newest `HOLDOUT_LAST_N` papers
  (default 200 in current config).
- **Holdout:** the newest `HOLDOUT_LAST_N` papers.

Using arxiv-id order (which correlates with publication time) gives a
"future-holdout" property — eval papers are typically newer than any
training paper, so they can't have leaked backward.

The split is deterministic and shared between train and inference,
so slow weights are guaranteed never to have seen holdout papers.

## Tokenization

Happens inside `train_modal.py::_prepare_dataset`:

1. **Pre-filter by `tokens_est`.** Drops docs shorter than
   `min_doc_tokens` before tokenization (avoids wasting compute on
   short garbage).

2. **Tokenize with the model's tokenizer.**
   `AutoTokenizer.from_pretrained(BASE_MODEL)` — respects the
   configured `TTT_MODEL_SIZE`. Add `add_special_tokens=False` so
   BOS/EOS don't pollute the sequence.

3. **Drop-short filter (exact).** After tokenization, drop docs
   still shorter than `min_doc_tokens`.

Result: a HuggingFace Dataset with `input_ids` column, memory-mapped
for cheap iteration.

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

## Holdout sampling

At eval time, `TTTInference.fetch_holdout_texts(n_papers, seed)`:

```python
_, holdout = split_holdout(open_dataset())
rng = random.Random(seed)
idx = rng.sample(range(len(holdout)), min(n_papers, len(holdout)))
return [holdout[i][TEXT_COLUMN] for i in idx]
```

Deterministic given the seed. Different seeds sample different papers
from the same fixed holdout pool.

`--n-papers > HOLDOUT_LAST_N` is capped silently (no error). If you
want to eval on more papers than your holdout has, increase
`HOLDOUT_LAST_N` and retrain (the training/eval split is a config
constant — changing it invalidates prior training's clean-holdout
property).

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
- [config.md](config.md) — `HOLDOUT_LAST_N`, `min_doc_tokens`, `max_seq_len`
- [inference.md](inference.md) — `fetch_holdout_texts`
