# Training

The outer loop, session structure, param groups, gradient flow, and
every metric the training loop emits.
Reference: [`train_modal.py`](../train_modal.py).

## Contents

1. [What training does](#what-training-does)
2. [Entrypoint and flags](#entrypoint-and-flags)
3. [The training loop, step by step](#the-training-loop-step-by-step)
4. [Param groups](#param-groups)
5. [Session structure](#session-structure)
6. [Loss mask](#loss-mask)
7. [Gradient checkpointing](#gradient-checkpointing)
8. [Grad accumulation semantics](#grad-accumulation-semantics)
9. [What each metric means](#what-each-metric-means)
10. [In-loop eval](#in-loop-eval)
11. [Nonfinite-loss guard](#nonfinite-loss-guard)
12. [Reference-count build](#reference-count-build)
13. [How to read a training run](#how-to-read-a-training-run)
14. [Related docs](#related-docs)

---

## What training does

Continual pretraining of Qwen3 with three trainable groups:

- **Base model:** frozen. Pretrained weights unchanged.
- **LoRA adapters:** attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`),
  MLP up-path (`gate_proj`, `up_proj`) on every layer, and `down_proj`
  on non-TTT layers. Rank 16, alpha 32, dropout 0.05.
- **TTT-layer `down_proj` (the `wdown` group):** full training,
  gently. This is the pretrained fast-weight initial state; it moves
  from the pretrained value but only slowly.
- **Fresh TTT modules (the `new` group):** `W_target` and `target_conv`
  per TTT layer, plus `output_gate` (weight + bias). Standard training
  from init (`W_target` zero-init, `target_conv` pass-through init).

Session-level TTT threads the fast weight `carried_delta` across items
within a session, so each new item starts from the previous items'
accumulated state. Gradient does not cross the item boundary (TBPTT).

The overall goal: teach the model to **use** the fast-weight mechanism
as short-term memory during a session, and have the slow-weight
adaptations (`wdown`, LoRA) support that usage.

---

## Entrypoint and flags

```
modal run --detach train_modal.py::train [FLAGS]
```

Flags (override the corresponding `TRAIN_CFG` field):

| flag | overrides | notes |
|---|---|---|
| `--limit-docs N` | (not a config field) | Cap training documents to the first N. `0` = full dataset. |
| `--num-epochs N` | `num_epochs` | Full passes over the training split. |
| `--grad-accum N` | `grad_accum_steps` | Micro-steps per optimizer step. |
| `--session 0\|1` | `session_training` | Master switch for cross-item carry at training time. |
| `--mode multi\|single\|hybrid` | `single_paper_sessions` + `hybrid_sessions` | Session-building strategy. `multi` = 2–6 random papers per session. `single` = one paper per session cut into k pieces. `hybrid` = short docs → single-item no-carry, long docs → k-slice carry (for diverse-length mixes like SlimPajama). Empty string leaves the config defaults. |
| `--min-doc-tokens N` | `min_doc_tokens` | Filter docs shorter than this many tokens before training. Set to `256` when running hybrid on SlimPajama. `0` leaves default. |
| `--hybrid-carry-min N` | `hybrid_carry_min_tokens` | Hybrid X threshold: below this length, no carry/no slice. `0` leaves default. |
| `--hybrid-slice-min N` | `hybrid_slice_min_tokens` | Hybrid n: min tokens per slice. `0` leaves default. |
| `--hybrid-slices-min N` | `hybrid_slices_min` | Hybrid y: min slice count for long docs. `0` leaves default. |
| `--hybrid-slices-max N` | `hybrid_slices_max` | Hybrid z: max slice count for long docs. `0` leaves default. |
| `--eval-n-papers N` | `eval_n_papers` | Number of eval papers when the dataset has no source column. `0` leaves default. |
| `--eval-n-papers-per-source N` | `eval_n_papers_per_source` | With a multi-source dataset: exactly N eval papers per source. Total = `N * n_sources`. `0` leaves default (`1`). |
| `--eval-min-tokens N` | `eval_min_tokens` | Filter eval holdout to docs with at least this many tokens before sampling. Guards against picking tiny StackExchange posts. `0` leaves default. |
| `--resume-from PATH` | (not a config field) | Resume from `step_<n>` (same-run) or `other_run/step_<n>` (cross-run). Optimizer momentum is NOT preserved. |

Env vars (read at `ttt_config` import time, **before** `modal run`
executes):

- `TTT_DATASET` (default `"arxiv"`) — one of the registered specs
  in `DATASETS`. Currently `"arxiv"` or `"slimpajama-6b"`. See
  [data.md](data.md#dataset-selection-dataset_spec).
- `TTT_MODEL_SIZE` (default `"0.6B"`) — one of `0.6B, 1.7B, 4B, 8B`.
- `TTT_LAYER_STRIDE` (default `2`) — every stride-th layer is TTT.
- `TTT_LAYER_START` (default `1`) — index of the first TTT layer.
- `TTT_BASE_MODEL` — full override for `BASE_MODEL`.

Recommended pattern for a test run (2000 papers, one epoch):

```bash
modal run --detach train_modal.py::train \
    --limit-docs 2000 --num-epochs 1 --session 1 --mode single
```

Recommended pattern for a scale-up run:

```bash
TTT_MODEL_SIZE=4B TTT_LAYER_STRIDE=2 \
    modal run --detach train_modal.py::train \
    --limit-docs 3000 --num-epochs 1 --session 1 --mode single
```

Recommended pattern for a diverse-mix pretraining run:

```bash
TTT_DATASET=slimpajama-6b \
    modal run --detach train_modal.py::train \
    --limit-docs 20000 --num-epochs 1 --session 1 --mode hybrid \
    --min-doc-tokens 256 --eval-n-papers-per-source 2
```

`--mode hybrid` routes short docs (< `hybrid_carry_min_tokens`, default
3000) through a single-item no-carry path and long docs through a
k-slice carry path — avoiding the "must be at least min_doc_tokens
long" cliff that the arxiv-style modes assume. Pair it with
`--min-doc-tokens 256` (SlimPajama has short docs you want to admit)
and `--eval-n-papers-per-source 2` so every source contributes two
holdout papers per eval.

---

## The training loop, step by step

The full sequence, from `train()` entry to the last checkpoint:

### 1. Config resolution

`_apply_cli_overrides(...)` merges CLI flags into `TRAIN_CFG` and
returns a fresh dataclass. The unmodified `TRAIN_CFG` module singleton
is not mutated — every override goes through `dataclasses.replace`.

The `session=-1` / `single_paper=-1` sentinel means "leave the
`TRAIN_CFG` default unchanged." This is because Modal parameters can't
cleanly hold `Optional[bool]`, so we use `-1` as a wire sentinel.

### 2. Resume resolution

`_resolve_resume(resume_from, run_name)` returns
`(adapter_path, ttt_ckpt_path)` or `(None, None)`:

- Empty string → `(None, None)` (fresh training).
- `"step_600"` → `/ckpt/<run_name>/step_600/{adapter, ttt_params.pt}`.
- `"other_run/step_600"` → `/ckpt/other_run/step_600/...`.

If either `adapter/` or `ttt_params.pt` is missing at the resolved
path, raises `FileNotFoundError`. This catches half-uploaded
checkpoints before training tries to consume them.

### 3. Model assembly

```python
model, tokenizer = build_model(
    adapter_path=..., ttt_ckpt_path=..., trainable=True,
)
```

See [architecture.md#model-assembly-order](architecture.md#model-assembly-order)
for what `build_model` does. Post-conditions:

- Base Qwen3 loaded in bf16 with flash-attention-2.
- TTT layers patched with `InPlaceTTTMLP`.
- LoRA adapters wrapped (either fresh or loaded from `adapter_path`).
- TTT trainables' `.requires_grad_` flipped on by `unfreeze_ttt_params`.
- If `ttt_ckpt_path` was provided, W_target / target_conv / TTT-layer
  down_proj / output_gate all loaded.

### 4. Training-mode configuration

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
model.config.use_cache = False
model.train()
for m in iter_ttt_modules(model):
    m.session_mode = cfg.session_training
```

Three subtleties:

- **`use_reentrant=False`** — the non-reentrant variant of grad
  checkpointing (the default in modern PyTorch). Reentrant would create
  autograd re-entry issues with PEFT's frozen embedding layer.
- **`enable_input_require_grads()`** — required because PEFT freezes
  embeddings; without it, checkpointed segments don't get gradient
  through the frozen input path. Classic PEFT + checkpointing trap.
- **`m.session_mode = cfg.session_training`** — sets the master toggle
  on every TTT module. When False, `_scan_forward` never stages
  `_next_carried`, so `carried_delta` stays `None` forever, so items
  are effectively independent.

### 5. Param groups + optimizer

```python
optim_groups, named_groups = build_param_groups(
    model, TTT_CFG,
    lr_lora, lr_wdown, lr_new,
    wd_full, wd_lora,
)
optimizer = bnb.optim.PagedAdamW8bit(optim_groups, betas=(0.9, 0.95))
```

See [Param groups](#param-groups) below for what goes where.

**PagedAdamW8bit** vs regular AdamW: 8-bit paged Adam keeps optimizer
state in 8-bit compressed form on GPU, paging chunks to CPU as needed.
For a 4B+ model with hundreds of MB of optimizer state, this saves ~4×
GPU memory vs fp32 Adam. Betas `(0.9, 0.95)` match Qwen3 team recs (Adam
beta2 slightly lower than the 0.999 default; useful for continual
pretraining).

**QLoRA is deliberately not used.** NF4 quantization of the base model
is incompatible with fully training `down_proj` on TTT layers
(quantized weights aren't backprop-friendly). We keep the base in bf16.

### 6. Snapshot `wdown_init`

```python
wdown_init = [p.detach().clone() for p in named_groups["wdown"]]
```

Kept for drift tracking: at each `param_log_every` step, we log
`||W_current - W_init||_F / ||W_init||_F` per TTT layer as
`health/wdown_drift_L<i>`. This is the "how far has `down_proj`
drifted" telemetry — see [observability.md](observability.md).

Cost: 600 MB on GPU at 0.6B (14 TTT layers × ~40 MB each in bf16).
Tolerable.

### 7. Telemetry init

`Telemetry` wraps wandb. If `WANDB_API_KEY` isn't set, degrades to
console print. If wandb fails at init or log time, degrades silently
(logs to console) without stopping the run.

Config logged to wandb includes `TRAIN_CFG` + `TTT_CFG` (as dicts),
`base_model`, `num_layers`, and `trainable_params_M`.

### 8. Dataset load

```python
ds = load_token_dataset(tokenizer, cfg, limit_docs or None)
```

1. `open_dataset(spec)` from HF Hub (or local dir). If the spec
   declares `source_meta_column`, a top-level `source` column is
   added here.
2. `split_holdout(ds, spec)` reserves the newest `spec.holdout_last_n`
   rows for eval.
3. `apply_source_filter(ds, spec)` drops rows whose source isn't in
   `spec.include_sources` (SlimPajama uses this to drop CommonCrawl).
4. **Deterministic shuffle** — `_shuffle_by_index(ds, seed=cfg.seed)`
   permutes row order via `.select(shuffled_indices)`. Runs *before*
   `--limit-docs` and *before* the in-corpus unigram count for the
   loss mask, so:
   - `--limit-docs N` samples uniformly across sources instead of
     picking the natural head (SlimPajama ships grouped by source, so
     an unshuffled head is all-C4).
   - Interrupting mid-epoch still leaves every source touched
     roughly proportionally.
   - In-corpus loss-mask counts reflect the whole mix.
5. Filter by `tokens_est >= min_doc_tokens` (cheap heuristic if the
   spec declares a `tokens_est_column`).
6. Slice by `limit_docs` if set.
7. Tokenize with `max_length=max_seq_len` truncation. The `source`
   column is preserved.
8. Drop docs with actual token count `< min_doc_tokens` (exact
   post-tokenize filter).

Output: HF Datasets object with `input_ids` (+ `source` when the spec
has one).

### 9. Loss mask setup

If `cfg.loss_mask_enabled`:

- Load the external reference counts from
  `loss_mask_reference_counts_path` if it exists; else fall back to
  counting in-corpus frequency (with a warning). Vocab size must match
  `model.config.vocab_size` (the padded vocab, 151936 for Qwen3) —
  mismatch raises `RuntimeError`.
- Build the "common token" mask (positions with common ids get
  `label = -100`).
- Apply three protect passes (domain terms, numeric, symbols).
- Move mask to CUDA.

See [Loss mask](#loss-mask) below for details on what each pass does.

### 10. Session count and LR schedule

```python
items_per_epoch = _items_per_epoch(cfg, doc_lengths)
steps_per_epoch = math.ceil(items_per_epoch / cfg.grad_accum_steps)
total_steps = steps_per_epoch * epochs
```

`_items_per_epoch` dispatches on the active mode:

- **Hybrid mode**: exact sum of `derive_slice_count(L, ...)` over all
  doc lengths — no estimation error.
- **Single-paper mode**: `len(ds) · 0.5 · (slice_min + slice_max)` —
  an expectation.
- **Multi-paper mode**: `len(ds) · expected_items_per_doc(...)`, i.e.
  `(1 - slice_prob) + slice_prob · 0.5 · (slice_min + slice_max)`.
  Overestimates in the tail (short docs decrement k toward
  feasibility) but close enough for sizing the schedule.

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(cfg.warmup_min_steps, int(cfg.warmup_ratio * total_steps)),
    num_training_steps=total_steps,
)
```

Cosine decay from peak LR down to 0 at `total_steps`. Warmup is
`max(10 steps, 2% of total)`.

### 11. Per-epoch, per-session, per-item loop

The inner three-level loop (see the pseudocode in
[architecture.md#training-dataflow-step-by-step](architecture.md#training-dataflow-step-by-step)):

```python
for epoch in range(epochs):
    for session_items in _make_epoch_sessions(cfg, len(ds), doc_lengths, rng):
        reset_session_state(model)             # start of session
        for pos, item in enumerate(session_items):
            full_ids = ds[item.doc_idx]["input_ids"]
            ids = torch.tensor([full_ids[item.start:item.end]], device="cuda")

            # Only mask first_tokens on paper-start items
            first_n = cfg.loss_mask_first_tokens if item.start == 0 else 0
            labels = apply_loss_mask(ids, common_mask, first_tokens=first_n)

            loss = model(input_ids=ids, labels=labels).loss
            if TTT_CFG.output_gate and TTT_CFG.gate_reg_weight > 0:
                loss = loss + TTT_CFG.gate_reg_weight * gate_reg_term(model)

            # Skip backward on non-finite loss
            if not torch.isfinite(loss):
                nonfinite += 1
                advance_session_state(model)   # still advance
                micro += 1
                continue

            (loss / cfg.grad_accum_steps).backward()
            advance_session_state(model)

            step_loss += loss.item() / cfg.grad_accum_steps
            micro += 1

            # Only every grad_accum micros: run the optimizer
            if micro % cfg.grad_accum_steps != 0:
                continue

            # Grad norm bookkeeping (per group, pre-clip)
            norms = ...

            total_norm = clip_grad_norm_(all_trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # log train/*, session/*, grad/*, gpu/*, perf/*
            # every param_log_every: also log health/*
            # every eval_every: run in-loop eval
            # every save_every: write a checkpoint
        sessions_done += 1
```

Key invariants:

- **`reset_session_state` at every session boundary.** New session, new
  carry.
- **`advance_session_state` after every backward** (even after nonfinite
  loss — otherwise `_next_carried` would still be stashed from a
  discarded forward).
- **`micro` counts items,** `step` counts optimizer steps. Their ratio
  is `grad_accum_steps`.
- **`optimizer.zero_grad(set_to_none=True)`** is preferred over
  zero-tensor-fill; the None form releases memory back to the allocator.

### 12. Checkpointing

`save_checkpoint(model, run_dir, step)` writes:

- `<run_dir>/step_<step>/adapter/` — the LoRA adapter via
  `model.save_pretrained(...)`.
- `<run_dir>/step_<step>/ttt_params.pt` — the TTT tensors via
  `save_ttt_state_dict`.

Then `ckpt_vol.commit()` flushes the Modal volume so the write is
visible to future container starts. See
[checkpoints.md](checkpoints.md) for the file format.

---

## Param groups

Three groups, three learning rates:

### `lora` (LoRA A + B tensors)

**Match:** parameter name contains `"lora_"` (LoRA A and B).

**Coverage:** attention projections and `gate_proj`/`up_proj` on every
layer, plus `down_proj` on non-TTT layers. **Never** matches
`down_proj` on TTT layers (enforced by the LoRA target regex).

**Default LR:** `1e-5`. Weight decay: `0.0`. LoRA training is typically
smoother than full-weight training, doesn't need weight decay.

**Approx param count at 0.6B:** ~4M.

### `wdown` (`down_proj.weight` on TTT layers)

**Match:** parameter name ends with `layers.<i>.mlp.down_proj.weight`
where `i` is in `TTT_CFG.layer_indices`.

**Coverage:** the pretrained fast-weight initial state, per TTT layer.
For 0.6B (14 TTT layers × 1024 × 3072): ~44M params (a real chunk of
the model).

**Default LR:** `3e-5` — 3× LoRA. Moves gently because the initial
value is meaningful (a good pretrained down-projection); we don't want
to drift far.

**Weight decay:** `0.1`. Standard AdamW; small pull toward zero to
prevent drift accumulation.

### `new` (fresh TTT modules)

**Match:** parameter name contains any of
`TTT_PARAM_MARKERS = ("target_conv", "w_target", "output_gate",
"v_source_norm")`. Note `v_source_norm.weight` is *in* markers so its
value ships in the checkpoint, but `.requires_grad_(False)` in
`InPlaceTTTMLP.__init__` prevents it from getting gradients.

**Coverage:** `W_target`, `target_conv`, `output_gate.weight`,
`output_gate.bias` per TTT layer.

**Default LR:** `2e-5` — between LoRA and wdown. Fresh modules with no
pretrained prior. Empirically the "right" LR: bumping higher makes
`W_target` overshoot the useful basin fast; lower and it doesn't escape
the dead basin at all.

### The `unclassified` guard

`build_param_groups` raises `RuntimeError` if any trainable parameter's
name matches none of the three groups:

```python
group = _classify_ttt_param(name, ttt_down)
if group is None:
    raise RuntimeError(f"Unclassified trainable parameter: {name}")
```

If you add a new fresh TTT module and forget to add its name-substring
to `TTT_PARAM_MARKERS`, this fires at model-setup time — early,
noisy, easy to fix. Silent misclassification would result in the
parameter getting a wrong LR or no LR at all.

### LR compensation at scale

At 4B / 8B, per-parameter gradient magnitude is 6–11× smaller than at
0.6B (bigger denominator in `grad / sqrt(fan_in)`). To keep the
per-parameter step size roughly equal at scale, bump all three LRs
~10×. See [scaling.md](scaling.md#1-per-parameter-gradient-dilution).

Do NOT bump `eta` to compensate for gradient dilution. `eta` is the
inner-loop update magnitude, not a learning rate; bumping it saturates
the Frobenius clip faster and quietly kills the mechanism.

---

## Session structure

`_make_epoch_sessions(cfg, ...)` returns a list of "sessions," each a
list of `SessionItem(doc_idx, start, end)` triples.

### Multi-paper (`--mode multi`, config default)

`make_slice_sessions(session_papers=(cfg.session_papers_min, cfg.session_papers_max), ...)`
produces sessions of `k ∼ Uniform(session_papers_min, session_papers_max)`
papers, randomly drawn from the training split without replacement
across the epoch.

Within each session, each paper is optionally sliced into
`k ∼ Uniform(slice_min, slice_max)` pieces with probability
`slice_prob`. Slices are contiguous and each is `≥ slice_min_tokens`;
`k` decrements toward feasibility if the paper is too short.

### Single-paper (`--mode single`)

`make_slice_sessions(session_papers=(1, 1), slice_prob=1.0,
slice_range=(single_paper_slices_min, single_paper_slices_max), ...)`
produces one session per paper, each session cutting the paper into
`k ∼ Uniform(single_paper_slices_min, single_paper_slices_max)`
consecutive pieces.

### Hybrid (`--mode hybrid`) — for diverse-length pretraining mixes

`make_hybrid_sessions(...)` routes each doc based on its length:

- **`L < hybrid_carry_min_tokens`** → single-item session, whole doc,
  no slicing. Session-training still calls `reset_session_state` at
  session start, so the model sees the "S_0 = 0" case for these — a
  correct training signal for short docs that shouldn't accumulate.
- **`L >= hybrid_carry_min_tokens`** → single-paper session sliced
  into `k = derive_slice_count(L, ...)` pieces, each `>= hybrid_slice_min_tokens`
  tokens. `k` is clamped to `[hybrid_slices_min, hybrid_slices_max]`.
  Carry propagates across the k slices via TBPTT.

Config knobs (all `hybrid_*` fields on `TrainConfig`):

| knob | default | meaning |
|---|---|---|
| `hybrid_carry_min_tokens` | 3000 | Below this length: no slicing, no carry. |
| `hybrid_slices_min` | 2 | Long docs: minimum slice count. |
| `hybrid_slices_max` | 6 | Long docs: maximum slice count. |
| `hybrid_slice_min_tokens` | 800 | Long docs: minimum tokens per slice. |

Constraint (enforced at builder call time):
`hybrid_carry_min_tokens ≥ hybrid_slices_min · hybrid_slice_min_tokens`.
Otherwise a doc at the boundary would be forced through the
multi-slice path but couldn't meet the minimum.

Intended for datasets like SlimPajama-6B where document length varies
wildly (from short tweets to book chapters). The point of training
with carry is to teach the model to use a non-zero fast-weight
initialization when it exists — it's fine to skip carry on the
short-doc tail rather than force-slice into sub-min-token pieces.
Docs shorter than `min_doc_tokens` are still filtered out at
tokenization time; you'll typically lower `min_doc_tokens` (e.g. to
256) when running hybrid on SlimPajama.

### When to pick which

- **Single-paper:** cleanest signal for arxiv-style corpora where
  every doc is long. Every item shares content with the session's
  other items, so the carry is always "relevant." No cross-paper noise.
- **Multi-paper:** teaches the model to handle cross-paper carry. This
  is the distribution `holdout_eval` (default form) actually tests. But
  the training signal is noisier because each new paper's content is
  uncorrelated with prior papers' carry.
- **Hybrid:** diverse-length pretraining mixes. Preserves the
  "learn to use a non-zero S_0" signal on long docs while cleanly
  handling short docs without slicing artifacts.

Empirically at 0.6B, single-paper training generalized to cross-paper
eval better than expected — the Frobenius clip keeps applied magnitude
bounded, so OOD carry magnitude isn't as damaging as feared.

### Determinism

Session schedules are deterministic per seed. `rng =
np.random.default_rng(cfg.seed)` seeded once, drawn from throughout the
epoch. Reproducibility guarantee: same `seed` + same code + same
dataset → same schedule. Verified in
[`test_session.py::test_builders_are_deterministic_per_seed`](../tests/test_session.py).

---

## Loss mask

**Purpose:** during pretraining, most CE loss comes from very common
tokens (function words, punctuation) that the model already predicts
correctly. Loss masking sets `labels = -100` on those positions so
gradient concentrates on content tokens.

Disabled by default (`loss_mask_enabled = False`). Enable with a config
edit; not a CLI flag.

### Two frequency sources

**External reference (preferred).** Unigram counts built once from
wikitext-103 via `build_reference_counts`. Stored at
`/ckpt/loss_mask/reference_wikitext103.pt`.

- **Pro:** domain-agnostic. Words common in general English get
  masked; domain-specific frequent words (like "model" in ML papers) do
  not.
- **Con:** requires running `build_reference_counts` once (a few
  minutes for wikitext-103).

**In-corpus fallback.** If the reference file is missing at
`loss_mask_reference_counts_path`, the train step falls back to
counting frequencies in the current training slice.

- **Pro:** no setup step.
- **Con:** ML-glue words (model, training, layer, function) become
  common in the corpus and get masked. Silently kills domain-signal
  learning. Only use as a last resort.

### Protect passes

The frequency-based common mask is over-aggressive — it will drop
domain-critical vocabulary if that vocabulary happens to be frequent
in the reference. Three protect passes force-unmask specific tokens:

**Domain terms (`loss_mask_protect_terms`).** Default is
`LOSS_MASK_DEFAULT_PROTECT_TERMS` in `train_utils.py`, ~150 ML terms
(Transformer, LSTM, VAE, PPO, ...). Each term is BPE-expanded into 8
variants (with/without leading space, capitalized/lowercase/uppercase).

- Single-piece variants (the whole term is one token) protect
  unconditionally.
- Multi-piece variants (BPE splits into ≥2 pieces) protect only if
  **every piece is ≥3 stripped chars.** This filters out weird
  acronym-onset splits like `" V" + "AE"` — protecting `" V"` would
  leak-unmask every capital-V word.

**Numeric (`loss_mask_protect_numeric = True`).** Force-unmask
pure-digit tokens. Reasoning: numbers in scientific papers are content
(sample sizes, hyperparameters), not filler.

**Symbols (`loss_mask_protect_symbols`).** Force-unmask math symbols
that are literally a single character: `= @ ^ _ \ + - * / | < >`. All
have leading-space variants too. Reasoning: same as numeric — these
are content in ML papers.

### First-N-tokens mask (`loss_mask_first_tokens = 16`)

Mask the first N tokens of each paper-**start** item (`item.start ==
0`). Skips the boilerplate ("# Introduction", author line, arxiv tag,
etc.) that isn't real content signal. Mid-paper slices
(`item.start > 0`) are unaffected. `0` disables.

### Diagnostic

```bash
modal run train_modal.py::diagnose_loss_mask \
    --limit-docs 100 --use-reference True
```

Prints:
- Per-slice mask fractions (25/50/75/100% of corpus).
- Top-K masked tokens with counts and decoded strings (should be
  common English: "the", "and", "of", punctuation).
- Spot-check table for function-words / ML-glue / domain content,
  with a two-row explanation for BPE splits.
- Per-piece breakdown showing what happens to "model" (single-piece
  MASKED) vs "diffusion" (single-piece protected) vs "VAE" (2-piece,
  first piece " V" MASKED but tail protected).

Use this to verify the protect list is doing what you expect before
enabling loss masking in a real run.

### When to enable

Loss masking helps most on large training corpora with heavy
boilerplate overlap between papers. On small runs (limit-docs=500), the
mask can slow effective learning because fewer active positions means
less gradient per step. Rule of thumb: enable only when you've observed
a plateau on unmasked training and want to concentrate remaining
capacity on content.

---

## Gradient checkpointing

Enabled unconditionally in `train_modal.py`:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
```

Why it's on:

- 16k-token forward passes at 0.6B produce ~10 GB of activations across
  the residual stream. Gradient checkpointing drops activations after
  forward and recomputes during backward, cutting peak activation
  memory to ~1 GB.
- At 4B / 8B, this is not optional — the run OOMs without it.

Why `use_reentrant=False`:

- The reentrant variant creates autograd re-entry issues when combined
  with PEFT's frozen embedding (the frozen embed_tokens output enters
  the checkpointed segment as a leaf that doesn't require grad).
- Non-reentrant is also the modern PyTorch default and is generally
  better-tested with modern autograd features.

Why `enable_input_require_grads()`:

- PEFT freezes the embedding layer. Without this call, checkpointed
  segments don't get gradient through the frozen input path — because
  the input to each checkpointed block appears as a `.requires_grad =
  False` leaf.
- `enable_input_require_grads` inserts a wrapper that makes checkpointed
  segments' inputs require grad even when they're derived from frozen
  parameters.

### Interaction with `advance_session_state`

The gradient-checkpointing recompute runs the checkpointed forward
segment **twice**: once during the forward pass (to compute the loss),
once during backward (to reconstruct activations for gradient
computation).

`_scan_forward` stages `_next_carried` during forward. If staging were
`accum += new`, the recompute would double-stage. Staging is instead
**idempotent** (`_next_carried = f(carried_delta, this_item_total)` —
overwrite, not accumulate; and `carried_delta` isn't mutated inside
`_scan_forward`), so both runs produce the same `_next_carried`. See
[mechanism.md#idempotence-under-gradient-checkpointing](mechanism.md#idempotence-under-gradient-checkpointing).

`advance_session_state` is called after backward, promoting
`_next_carried → carried_delta` exactly once per item.

---

## Grad accumulation semantics

`grad_accum_steps` (default 16) micro-steps are accumulated before one
optimizer step:

- Each micro-step: `(loss / grad_accum_steps).backward()`. The `/=
  grad_accum` normalizes so accumulated grad approximates the average
  loss over the window.
- After every `grad_accum_steps` micro-steps: `clip_grad_norm_`,
  `optimizer.step()`, `scheduler.step()`, `zero_grad`.

Important interaction with **sessions**: session boundaries do
**not** align with grad-accumulation boundaries. A session of 5 items
starts at micro `k`; the next session starts at micro `k+5`; the
optimizer step lands wherever `micro % 16 == 0`. So a single optimizer
step may span multiple sessions, and each session's items are split
across possibly multiple optimizer steps.

This is fine because:

- Each `advance_session_state` happens per item, not per optimizer
  step.
- `reset_session_state` at session boundaries clears the *carry* but
  not the *accumulated gradient*.

The math: the accumulated gradient reflects "average loss across the
last N items regardless of session boundaries," which is exactly the
right objective for the outer optimizer.

### `total_norm` and the grad clip ratio

```python
total_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
metrics["train/grad_clip_ratio"] = min(1.0, cfg.max_grad_norm / (float(total_norm) + 1e-12))
```

- `total_norm` is the pre-clip L2 norm across all trainable params.
- `clip_grad_norm_` scales in-place so the post-clip norm is at most
  `max_grad_norm`.
- `grad_clip_ratio = 1` → no clip active. `< 1` → clip fired, scaled
  down by that factor.

`max_grad_norm = 10.0` (was `1.0`, bumped after observing that `1.0`
was clipping new-module gradient spikes that were actually productive
signal in the escape-the-dead-basin regime).

---

## What each metric means

Everything logged to wandb, grouped by category. See
[observability.md](observability.md) for the full metric reference and
alert thresholds.

### Micro-step metrics (`micro/*`)

Logged every item. X-axis is `micro/step`.

- `micro/paper_loss` — this item's CE loss (post-mask).
- `micro/paper_tokens` — token count of this item's input_ids.
- `micro/session_pos` / `micro/session_n` — position within session, total items in session. Watch that `session_n > 1` when session-mode is on.
- `micro/state_ratio_mean` — mean `||eta · carried_delta||_F / ||W_down||_F` across TTT layers, after this item. Zero at session start; grows with each item.
- `micro/unmasked_token_frac` — fraction of positions whose label ≠ -100. Sanity check on the loss mask.

### Train-step metrics (`train/*`, `grad/*`, `session/*`, `perf/*`, `gpu/*`)

Logged every optimizer step. X-axis is `train/step`.

- `train/loss` — the loss averaged across the accum window. (Sum of `loss.item() / grad_accum` across the last `grad_accum` items.)
- `train/lr_lora`, `train/lr_wdown`, `train/lr_new` — the three LR values from the scheduler.
- `train/grad_clip_ratio` — as above.
- `grad/lora`, `grad/wdown`, `grad/new` — L2 norm of gradients in each param group, pre-clip.
- `session/state_ratio_L<i>` — per-layer state ratio at optimizer-step time.
- `session/state_ratio_mean` / `session/state_ratio_max` — aggregates across TTT layers.
- `session/sessions_done` — how many sessions have completed.
- `perf/tokens_per_s` — tokens processed in the last window / window duration.
- `perf/sec_per_step` — window duration.
- `perf/total_tokens` — running total token count.
- `gpu/mem_alloc_gb`, `gpu/mem_reserved_gb`, `gpu/mem_peak_gb` — CUDA memory. Peak is running max.

### Health metrics (`health/*`)

Logged every `param_log_every` (default 50) steps. Heavier — walks the
model to compute norms.

- `health/wdown_drift_L<i>` — `||W_current - W_init||_F / ||W_init||_F` per TTT layer. Should slowly rise.
- `health/w_target_L<i>` — Frobenius norm of `W_target` per layer. Starts at 0 (dead basin), grows as training proceeds.
- `health/conv_L<i>` — Frobenius norm of `target_conv.weight`. Starts near 1 (single non-zero position × d channels), grows as the conv learns.
- `health/gate_mean_L<i>` / `health/gate_std_L<i>` — per-layer sigmoid gate mean and std. Healthy: mean in `(0.1, 0.9)`, std `> 0.05`.
- `health/lora_norm` — global L2 norm of all LoRA parameters.

### Anomaly metrics (`anomaly/*`)

Logged on demand.

- `anomaly/nonfinite_count` — cumulative count of NaN/Inf losses across
  the run. Also triggers `Telemetry.alert(...)` at counts 1, 10, 100.

### Eval metrics (`eval/*`)

Logged every `eval_every` (default 100) optimizer steps.

- `eval/carry_ppl` — token-weighted geometric mean of per-slice ppls
  with `evolve=True`.
- `eval/fresh_ppl` — same with `evolve=False`.
- `eval/gap` — `fresh_ppl - carry_ppl`. **Positive means carry helps.**
- `eval/state_ratio_final` — mean state ratio at the end of each paper.
- `eval/paper_<i>/carry_ppl`, `eval/paper_<i>/fresh_ppl`,
  `eval/paper_<i>/gap`, `eval/paper_<i>/state_ratio_final` — per-paper
  breakdown.
- `eval/carry_ppl_slice_<j>`, `eval/fresh_ppl_slice_<j>`,
  `eval/state_ratio_slice_<j>` — per-slice-position aggregates across
  papers (so you can see if the gap grows with position within the
  paper).

---

## In-loop eval

Every `eval_every` (default 100) optimizer steps, `run_holdout_eval` is
invoked. Uses whatever dataset `TTT_DATASET` selected — the eval
holdout is always drawn from the same spec you're training on, not
some fixed arxiv snapshot.

### How eval papers are chosen

`fetch_holdout_papers_ids(tokenizer, cfg)` picks the eval sample:

1. **Load holdout.** `split_holdout(open_dataset(spec), spec)` returns
   the newest `spec.holdout_last_n` rows. Then `apply_source_filter`
   drops sources not in `spec.include_sources` (SlimPajama drops
   CommonCrawl).
2. **Filter by min tokens.** `_filter_holdout_by_min_tokens(...,
   cfg.eval_min_tokens)` keeps docs likely to have at least this many
   tokens. Uses the spec's `tokens_est_column` when present; otherwise
   `len(text) >= 4 * eval_min_tokens` (rough char-to-token proxy).
   Default `eval_min_tokens = 2048` guards against picking tiny
   StackExchange posts that can't be sliced into `eval_n_slices`
   meaningful pieces.
3. **Sample.** Three sampling policies, dispatched by config:
   - **`n-per-source`** (default when `eval_n_papers_per_source > 0`
     and the dataset has a source column): pick exactly
     `eval_n_papers_per_source` papers per source. Total =
     `n_sources * eval_n_papers_per_source`. Every domain represented
     every eval — required for per-source metrics to be non-noisy.
   - **`stratified round-robin`** (when there's a source column but
     `eval_n_papers_per_source = 0`): round-robin one-per-source until
     `eval_n_papers` is hit, then fill from what's left.
   - **`uniform`** (no source column): plain random draw of
     `eval_n_papers` papers.
4. **Tokenize** the picked docs, return `[(input_ids, source_label), ...]`.

Set `--eval-min-tokens 4096` if your `eval_n_slices` is 8 and you
want each slice to have >= 512 tokens on average.

### What each eval run does

```python
run_holdout_eval(model, eval_papers, cfg.eval_n_slices,
                 train_session_mode=cfg.session_training,
                 paper_sources=eval_sources)
```

1. Snapshot every TTT module's `carried_delta` and `_next_carried`.
2. Set `model.eval()`, `session_mode = True`, `ttt_evolve = ...`.
3. For each of the sampled papers:
   - Split into `eval_n_slices` (default 8) equal token slices.
   - Run twice: `evolve=True` (carry accumulates within the paper) and
     `evolve=False` (`state.delta` never updates — fresh baseline).
4. Compute per-paper token-weighted ppls (geometric mean weighted by
   slice token count).
5. Aggregate to `eval/carry_ppl`, `eval/fresh_ppl`, `eval/gap`, plus
   the per-source dictionary when sources are provided.
6. Restore `model.train()`, `session_mode = train_session_mode`, and
   the snapshotted `carried_delta` / `_next_carried`. Restore
   `ttt_evolve = True`.

**Zero side-effect on the training loop:** the snapshot + restore
ensures the carry state that resumes training is identical to what
it was pre-eval.

**Set `eval_every = 0` to disable.** The extra forward passes add
20-30% to per-step time; disable if you're throughput-bound.

### Per-source eval

When the sample has source labels, `run_holdout_eval` also emits:

- `eval/<source>/carry_ppl` — token-weighted geometric mean over
  papers of this source.
- `eval/<source>/fresh_ppl` — same, `evolve=False`.
- `eval/<source>/gap` — `fresh_ppl - carry_ppl` per source.
- `eval/<source>/n_papers` — how many holdout papers of this
  source were sampled.

This is the "which document domains benefit most from TTT" signal —
the bar chart across sources is the central figure for a diverse-mix
training run. Reads directly into a wandb panel grouped by source
prefix.

The inference-side `holdout_eval` local entrypoint prints the same
information as a `per-source (token-weighted)` table under
`_print_per_source_summary`. See
[inference.md#per-source-eval-table](inference.md#per-source-eval-table).

---

## Nonfinite-loss guard

If `loss.isfinite()` is False (NaN or ±Inf):

```python
nonfinite += 1
telemetry.log({"anomaly/nonfinite_count": nonfinite, "micro/step": micro})
if nonfinite in (1, 10, 100):
    telemetry.alert("Nonfinite loss", ...)
advance_session_state(model)      # still advance so _next_carried isn't stashed
micro += 1
continue                          # skip backward
```

The item is skipped: no backward, no gradient accumulation for this
micro-step. But we still:

- Advance the session state (so the next item's carry reflects reality).
- Increment `micro` (so grad-accumulation boundaries stay aligned).
- Log to wandb + alert on 1/10/100 counts.

A single bad item can't destroy the run. A cascade of bad items
(hundreds of nonfinites) suggests something structurally wrong (LR too
high, unstable inputs) and the alert should be noticed.

---

## Reference-count build

Run once, per tokenizer:

```bash
modal run train_modal.py::build_reference_counts \
    [--dataset-id ... --dataset-config ... --split ... --out-name ... --limit-docs N]
```

Defaults produce `wikitext-103-raw-v1` train counts at
`/ckpt/loss_mask/reference_wikitext103.pt`.

Contents:

```python
{
    "counts": Tensor[vocab_size, int64],   # unigram counts
    "tokenizer_name": BASE_MODEL,
    "dataset_id": "Salesforce/wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "split": "train",
    "n_docs": len(tokens_ds),
    "n_tokens": int,
    "n_unique": int,
    "vocab_size": int,
}
```

**Vocab size alignment.** The tensor is sized to
`AutoConfig.from_pretrained(BASE_MODEL).vocab_size` (padded embedding
size, 151936 for Qwen3), **not** `tokenizer.vocab_size` (151669). This
was a real bug pre-fix: the training loop validates counts against
`model.config.vocab_size`, so an unpadded reference triggered a
`RuntimeError`. The build tool aligns on the model side.

**BOS/EOS excluded.** Tokenization uses `add_special_tokens=False` so
BOS/EOS don't inflate the unigram tally.

**Run again when:** you switch base model (different vocab), retokenize
with different special-token handling, or add a new reference source.

---

## How to read a training run

The healthiest possible run at 0.6B looks like:

- `train/loss` falls smoothly from ~4.5 to ~2.5 over the first
  100-200 steps, then declines slowly.
- `grad/new` shows a **spike** in the first 20-50 steps as the
  mechanism escapes the dead basin, then decays to a steady value
  around 0.4.
- `grad/lora`, `grad/wdown` are stable at 0.5-1.0 throughout.
- `health/w_target_L<i>` starts at 0.0 and rises smoothly to some
  layer-dependent plateau (typically 0.1-2.0).
- `health/conv_L<i>` starts near `sqrt(d)` (since only the last
  position is 1.0 per channel) and drifts slowly.
- `health/gate_mean_L<i>` starts near 0.12 (sigmoid of -2), rises to
  0.3-0.7 as training proceeds, and stabilizes.
- `health/gate_std_L<i>` grows from ~0.05 to 0.15-0.25.
- `session/state_ratio_mean` in session mode: near 0 at session
  start, rises to 0.5-2 by mid-session, plateaus (with `carried_decay
  < 1`) or grows unboundedly (`carried_decay = 1`).
- `session/state_ratio_max` (max across layers) may be 2-5× the mean
  and can hit the clip.
- `eval/gap` (in-loop): flat or slightly positive early; grows to
  +3-5 ppl by step 200-400.
- `train/grad_clip_ratio` stays near 1.0 (no clip active). Occasional
  dips are fine.
- `anomaly/nonfinite_count` stays at 0.

Warning signs:

- `grad/new` never spikes → dead basin, mechanism didn't wake up. See
  [failure-modes.md#dead-basin](failure-modes.md#dead-basin).
- `session/state_ratio_mean` hits `clip_tau` and stays there →
  saturation regime. See
  [failure-modes.md#state-saturation](failure-modes.md#state-saturation).
- `eval/gap` stays flat despite `state_ratio > 0` → mechanism is
  active but not producing useful adaptation. Check `gate_mean`; if
  stuck near 0, the model has learned to ignore TTT.
- `anomaly/nonfinite_count` rising → LR too high, or a specific
  malformed input. Inspect the recent items via micro-step wandb view.
- `train/grad_clip_ratio << 1` sustainedly → grad clip is firing every
  step. Either bump `max_grad_norm` or lower LRs.

---

## Related docs

- [architecture.md](architecture.md) — how the loop connects to the mechanism
- [mechanism.md](mechanism.md) — what `_scan_forward` actually computes
- [config.md](config.md) — every knob referenced above
- [checkpoints.md](checkpoints.md) — save/load format and resume behavior
- [observability.md](observability.md) — full metric reference + alerts
- [failure-modes.md](failure-modes.md) — what to do when training goes wrong
- [scaling.md](scaling.md) — retuning across model sizes
