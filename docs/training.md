# Training

The outer loop, session structure, and support tools.
Reference: [`train_modal.py`](../train_modal.py).

## What training does

Continual pretraining of Qwen3 with:
- Base model **frozen** (pretrained weights unchanged)
- LoRA adapters on attention + MLP paths (all layers, but skipping
  `down_proj` on TTT layers)
- Full training of `down_proj` on TTT layers (moves gently — pretrained
  fast-weight initial state)
- Full training of fresh modules: `W_target` and `target_conv` on each
  TTT layer

Session-level TTT threads accumulated fast weights across items in a
session (papers, or paper slices) so the model learns to leverage
prior context via fast-weight memory.

## Training entrypoint

```
modal run --detach train_modal.py::train [FLAGS]
```

Flags:
- `--limit-docs N` — cap training documents (`0` or omit = full)
- `--num-epochs N` — override `TrainConfig.num_epochs`
- `--grad-accum N` — override `grad_accum_steps`
- `--session 0|1` — override `session_training`
- `--single-paper 0|1` — override `single_paper_sessions`

Env vars (read at module import):
- `TTT_MODEL_SIZE` (default 8B)
- `TTT_LAYER_STRIDE` (default 2)
- `TTT_LAYER_START` (default 1)
- `TTT_BASE_MODEL` (full override)

## The training loop, step by step

1. **Load dataset** via `data_utils.open_dataset()`, split off the
   holdout via `data_utils.split_holdout()`. The training slice is
   everything except the newest `HOLDOUT_LAST_N` papers.

2. **Pre-filter by token estimate.** Cheap heuristic to drop
   short/malformed docs before tokenization.

3. **Tokenize + drop-short filter.** Documents shorter than
   `min_doc_tokens` (default 2048) are dropped.

4. **Build the loss mask** (if `loss_mask_enabled=True`).
   - Load the wikitext-103 reference counts if the file exists at
     `loss_mask_reference_counts_path`; otherwise fall back to
     in-corpus frequency (with a warning).
   - Build the "common token" mask (keeps the rarest tokens up to
     `1 - keep_fraction` frequency mass).
   - Run protect passes: force-unmask numeric tokens (single digits
     etc.), math symbols (`=`, `@`, `^`, …), and the domain terms
     list (Transformer, LSTM, VAE, …). All variants are BPE-expanded
     and multi-piece pieces are protected only if all pieces are
     ≥3 stripped chars (avoids leak-unmask on acronym splits).

5. **Set up model** via `build_model()`:
   - Load base Qwen3
   - Patch TTT layers (`patch_model_with_ttt`)
   - Wrap in LoRA (`get_peft_model`)
   - Unfreeze the TTT trainables (LoRA freezes everything non-LoRA;
     we re-enable grads for `W_target`, `target_conv`, and TTT-layer
     `down_proj`)
   - Enable gradient checkpointing (`use_reentrant=False`)

6. **Build param groups** via `build_param_groups()`:
   - `lora` group (LoRA A+B on attn + gate/up + non-TTT down_proj)
   - `wdown` group (TTT-layer down_proj)
   - `new` group (W_target + target_conv)
   - Each gets its own LR and weight decay.

7. **Snapshot initial `wdown`** for drift tracking (per-layer
   `||W_current - W_init||_F / ||W_init||_F`).

8. **Compute total steps** (`items_per_epoch × epochs / grad_accum`)
   and build a cosine warmup+decay LR schedule.

9. **`set_session_mode(model, cfg.session_training)`** — this is the
   master toggle. When False, `_scan_forward` doesn't stage the next
   carry, and `carried_delta` stays `None` forever, so items are
   effectively independent.

10. **For each epoch:**
    - Build a fresh session schedule (either multi-paper or single-paper).
    - For each session:
      - `reset_session_state(model)` — clear carry.
      - For each `SessionItem`:
        - Extract input_ids from the tokenized document.
        - Build labels via `apply_loss_mask(ids, common_mask,
          first_tokens=cfg.loss_mask_first_tokens if item.start == 0 else 0)`.
          `first_tokens` only applies at paper-start items; mid-paper
          slices are real content.
        - `loss = model(input_ids=ids, labels=labels).loss`
        - `(loss / grad_accum).backward()`
        - `advance_session_state(model)` — promote next_carried into
          carried_delta.

    - Every `grad_accum` micro-steps:
      - `clip_grad_norm_(all_trainable_params, max_grad_norm)` and
        record `grad_clip_ratio`.
      - `optimizer.step()`, `scheduler.step()`, `optimizer.zero_grad()`.
      - Log lightweight metrics (loss, grads, state ratio, LR).

    - Every `param_log_every` optimizer steps:
      - Log heavier metrics (`health/*`, param drift, gate stats).

    - Every `eval_every` optimizer steps:
      - Run in-loop eval on `eval_n_papers` × `eval_n_slices`, log
        `eval/*` metrics (carry ppl, fresh ppl, gap, state ratio).

    - Every `save_every` optimizer steps:
      - `save_checkpoint()` — writes adapter + `ttt_params.pt`.

## Session structure

### Multi-paper (default)

`make_session_schedule()` produces a list of sessions, each with
`k ∼ U[session_papers_min, session_papers_max]` papers randomly drawn
from the training split (without replacement across the epoch).

`build_session_items()` then optionally applies within-paper slicing
per `slice_prob`. Each session ends up as an ordered list of
`SessionItem(doc_idx, start, end)` triples.

### Single-paper (`--single-paper 1`)

`make_single_paper_sessions()` builds one session per training paper.
Each session cuts the paper into `k ∼ U[single_paper_slices_min,
single_paper_slices_max]` consecutive pieces of ≥`slice_min_tokens`
tokens.

**When to pick which:**

- **Single-paper:** cleanest signal — every item shares content with
  the session's other items. No cross-paper noise polluting the carry.
  Best when the goal is "carry within a paper."
- **Multi-paper:** teaches the model to handle cross-paper carry. This
  is the distribution `holdout_eval` actually tests. But the training
  signal is noisier because each new paper's content is uncorrelated
  with prior papers' carry.

Empirically at 0.6B, single-paper training generalized to cross-paper
eval better than expected — the clip keeps applied magnitude bounded,
so the "OOD carry magnitude" isn't as damaging as feared.

## Loss mask

**Purpose:** during pretraining, most CE loss comes from very common
tokens (function words, punctuation) that the model already predicts
correctly. Loss masking sets `label = -100` on those positions so
gradient concentrates on content tokens.

**Two frequency sources:**

- **External reference (preferred):** counts built once from
  wikitext-103 via `build_reference_counts`. Domain-agnostic — words
  that are common in general English get masked. This avoids the
  pathological case where ML-glue words like "model," "training,"
  "layer" get masked because they're common in the corpus.

- **In-corpus fallback:** if the reference file is missing, falls back
  to counting frequencies in the training slice. Only use as a last
  resort — you'll silently mask domain vocabulary.

**Protect passes:**

- `protect_token_ids(mask, tokenizer, terms)` — force-unmask common
  domain terms (Transformer, LSTM, PPO, …). Terms are BPE-expanded
  into 8 variants (with/without leading space, capitalization). Multi-
  piece BPE splits protect only if all pieces are ≥3 stripped chars.
- `protect_numeric_tokens(mask, tokenizer)` — force-unmask pure-digit
  tokens (0–9, 10, 100, …). Numbers carry meaning in papers.
- `protect_symbol_tokens(mask, tokenizer, symbols)` — force-unmask
  math symbols configured in
  `loss_mask_protect_symbols`.

**First-N-tokens mask** (`loss_mask_first_tokens=16`): mask the first
N tokens of each paper-start item (`item.start == 0`). Skips the
boilerplate ("# Introduction", author line, etc.) that isn't real
content signal. Mid-paper slices are unaffected.

**Diagnostic:**
```
modal run train_modal.py::diagnose_loss_mask --limit-docs 100 --use-reference
```
Prints:
- Mask fraction per section
- Top-K masked tokens (should be common English words like "the",
  "and")
- Top-K protected tokens (should be ML terms + digits + symbols)
- Spot-check table showing per-piece breakdown for multi-piece BPE

**When to enable:** on a large training corpus with heavy boilerplate
overlap between papers. On small runs, the mask can slow effective
learning (fewer active positions → less gradient per step).

## Building the wikitext-103 reference

Run once per tokenizer change:

```
modal run train_modal.py::build_reference_counts
```

Optional args:
- `--dataset-id` (default `Salesforce/wikitext`)
- `--dataset-config` (default `wikitext-103-raw-v1`)
- `--split` (default `train`)
- `--out-name` (default `reference_wikitext103.pt`)
- `--limit-docs N` — for quick tests

Output: `<CKPT_MOUNT>/loss_mask/<out_name>`. Contains:
- `counts`: `[vocab_size]` tensor of unigram counts (aligned on
  `model.config.vocab_size`, which is the padded embedding size)
- Metadata: `tokenizer_name`, `dataset_id`, `dataset_config`, `split`,
  `n_docs`, `n_tokens`, `n_unique`, `vocab_size`

At training start, `load_reference_counts(path, expected_vocab_size)`
raises `RuntimeError` if vocab size doesn't match — catches
tokenizer changes early.

## Gradient checkpointing

Enabled unconditionally in `train_modal.py`:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()   # PEFT + checkpointing trap
```

`enable_input_require_grads()` is required because PEFT freezes
embeddings; without it, checkpointed segments don't get gradient from
the frozen input path.

**Interaction with `advance_session_state`:** the gradient checkpoint
recomputes the forward pass during backward. Staging into
`_next_carried` happens during forward and is deterministic
(idempotent to recompute), so re-running forward during backward
produces the same value. `advance_session_state` promotes
`_next_carried` after backward, so it's promoted exactly once per
item.

## In-loop eval

Every `eval_every` (default 100) optimizer steps, an in-loop eval
runs on `eval_n_papers` × `eval_n_slices`. Logged as `eval/carry_ppl`,
`eval/fresh_ppl`, `eval/gap`, `eval/state_ratio_mean`.

**What it measures:** whether the mechanism is contributing anything
at the current checkpoint. Not a substitute for a full `holdout_eval`
run — it uses 3 fixed papers, so per-paper variance is high — but a
useful "is it moving in the right direction" signal during a long
training run.

Set `eval_every=0` to disable if the extra forward passes are hurting
throughput.

## Nonfinite-loss guard

If any `.backward()` produces a NaN or Inf loss, the step is skipped,
`anomaly/nonfinite_count` is incremented, and a wandb alert fires.
Prevents a single bad batch from destroying the run.

## Related docs

- [architecture.md](architecture.md) — how the loop connects to the mechanism
- [mechanism.md](mechanism.md) — what `_scan_forward` actually computes
- [checkpoints.md](checkpoints.md) — save/load format
- [observability.md](observability.md) — full metric reference
- [failure-modes.md](failure-modes.md) — what to do when training goes wrong
