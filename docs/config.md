# Configuration Reference

Every knob in [`ttt_config.py`](../ttt_config.py), plus the env-var
level constants and CLI flags, with **what it does, when to tune it,
and how it fails**. Grouped by concern.

## Contents

1. [Environment variables](#environment-variables-read-at-import-time)
2. [Module-level constants](#module-level-constants)
3. [`TTTConfig` (the mechanism)](#ttconfig-the-mechanism)
4. [`TrainConfig` (the outer loop)](#trainconfig-the-outer-loop)
5. [CLI flag overrides](#cli-flag--override-map)
6. [Sensitivity notes](#sensitivity-notes)
7. [Per-knob failure modes](#per-knob-failure-modes)
8. [Recipe: minimal safe change](#recipe-minimal-safe-change)


## Environment variables (read at import time)

Set before `modal run` — they're captured into module-level constants
when `ttt_config` is imported.

| var | default | effect |
|---|---|---|
| `TTT_MODEL_SIZE` | `"0.6B"` | Substitutes into `BASE_MODEL = f"Qwen/Qwen3-{SIZE}"`. Valid: `"0.6B"`, `"1.7B"`, `"4B"`, `"8B"`. |
| `TTT_BASE_MODEL` | (unset) | Full override for `BASE_MODEL`. Use for non-Qwen3 paths. |
| `TTT_LAYER_STRIDE` | `2` | TTT layers = every stride-th layer. Bigger stride = fewer TTT layers (less memory, less capacity). |
| `TTT_LAYER_START` | `1` | Index of the first TTT layer. Range: `[0, stride)`. |

**Layer indices** are derived from these via `derive_ttt_layer_indices()`
called by `build_model` at load time. For Qwen3-0.6B (28 layers) with
default `stride=2, start=1`: `(1, 3, 5, 7, ..., 27)` = 14 TTT layers.

For 8B with `TTT_LAYER_STRIDE=4`, `TTT_LAYER_START=1`, 36 layers:
`(1, 5, 9, 13, ..., 33)` = 9 TTT layers.

## Module-level constants

Living in [`ttt_config.py`](../ttt_config.py) top-of-file:

| constant | default | purpose |
|---|---|---|
| `CKPT_VOLUME_NAME` | `"ttt-checkpoints"` | Modal volume for checkpoints, snapshots, loss-mask reference |
| `HF_CACHE_VOLUME_NAME` | `"hf-hub-cache"` | Modal volume for HF Hub downloads |
| `CKPT_MOUNT` | `"/ckpt"` | Mount path in containers |
| `HF_CACHE_MOUNT` | `"/hf-cache"` | Mount path in containers |
| `DATASET_SOURCE` | `"suchitraIdu/arxiv-ml-16k"` | HF Hub dataset repo id |
| `TEXT_COLUMN` | `"text"` | Column name in the dataset |
| `TOKENS_EST_COLUMN` | `"tokens_est"` | Pre-computed token estimate column (used for cheap filtering) |
| `HOLDOUT_LAST_N` | `200` | Newest N papers reserved for eval (never trained on) |
| `LOSS_MASK_DEFAULT_PROTECT_TERMS` | (tuple of 151 ML terms) | Domain terms force-unmasked even if frequent |

## `TTTConfig` (the mechanism)

| field | default | notes |
|---|---|---|
| `layer_indices` | `None` (lazily populated) | Explicit tuple can be passed for tests. In production, populated from `derive_ttt_layer_indices` in `build_model`. |
| `chunk_size` | `50` | Tokens per fast-weight update. Changing requires retraining. Smaller = finer temporal resolution + more chunks + more memory (`k=N/C` chunk-dim tensors); larger = coarser resolution but memory-friendly. Raise to ~200-400 at 8B to fit the scan tensors. |
| `eta` | `7e-2` | Inner-loop learning rate: `W_eff = W_down + eta·S`. Scale ~inversely with model width. Do NOT bump like an LR. |
| `normalize_delta_by_chunk` | `True` | Divide each chunk's delta by chunk size (uses actual token count, so last-chunk padding doesn't inflate). Makes `eta` roughly C-independent. |
| `conv_kernel_size` | `8` | Causal Conv1D width for V = Conv1D(source) @ W_target. With `v_source="hidden_state"`, 4-8 is sweet spot; with `"embedding"`, 16-32. |
| `v_source` | `"hidden_state"` | Where V's source sequence comes from. `"embedding"` = raw token embeddings (paper reference); `"hidden_state"` = per-layer input (more expressive, needs a per-layer stream buffer). |
| `v_bidirectional` | `False` | Symmetric pad (past+current+future) for target_conv. **WARNING:** breaks chunk-causality under standard NTP — carry can leak ground-truth right-context. Streaming inference ignores this flag. |
| `output_gate` | `True` | Per-position sigmoid gate on the TTT contribution. Opens extra gradient path for W_target. |
| `output_gate_bias_init` | `-2.0` | Initial gate bias (sigmoid ≈ 0.12 → mostly closed). |
| `gate_reg_weight` | `0.0` | Coefficient on `mean(sigmoid(W_g h)^2)`. Add if you want to bias the gate closed. |
| `clip_enabled` | `True` | Frobenius clip on `||eta * cum||_F` per chunk. |
| `clip_tau` | `5.0` | Clip threshold. When active, applied delta magnitude is capped at 5, direction preserved. |
| `clip_at_inference_only` | `False` | When True, clip is off during training. Currently off so training and inference are consistent — recommended to leave off. |
| `carried_decay` | `1.0` | EMA staging: `carried ← decay·carried + this_item_total`. `1.0` (default) = pure accumulation, state grows unbounded — set `0.9-0.95` for bounded long sessions. `0.9` = ~10-item half-life. `0.8` = ~5-item. Steady-state plateau ≈ per_item_delta / (1 - decay). |

## `TrainConfig` (the outer loop)

### Data and batching

| field | default | notes |
|---|---|---|
| `max_seq_len` | `16384` | Truncate papers past this. Papers ≥90k tokens exist; the tail is lost. |
| `min_doc_tokens` | `2048` | Documents shorter than this are dropped. |
| `micro_batch_size` | `1` | Fixed at 1. Fast weights are per-stream; the loop rejects B changes mid-session. |
| `grad_accum_steps` | `16` | Optimizer steps land every this many forward+backward passes. |
| `num_epochs` | `1` | Passes over the training split. Override with `--num-epochs`. |

### Learning rates (three groups)

| field | default | applies to |
|---|---|---|
| `lr_lora` | `1e-5` | LoRA A+B on attention + gate/up (all layers), down_proj (non-TTT layers) |
| `lr_wdown` | `3e-5` | down_proj weights on TTT layers (the fast-weight initial state) |
| `lr_new_modules` | `2e-5` | W_target + target_conv on all TTT layers |

**At bigger model sizes (4B, 8B), all three typically need ~10× bump**
to compensate for smaller per-param gradients at scale. See
[scaling.md](scaling.md).

### Regularization + schedule

| field | default | notes |
|---|---|---|
| `weight_decay_full` | `0.1` | Weight decay for the wdown and new groups. |
| `weight_decay_lora` | `0.0` | LoRA typically doesn't want weight decay. |
| `warmup_ratio` | `0.02` | Warmup steps = `int(warmup_ratio * total_steps)`. |
| `warmup_min_steps` | `10` | Minimum warmup regardless of ratio. |
| `max_grad_norm` | `10.0` | Global grad clip. Bumped from `1.0` after observing that the tight clip was killing new-module gradient spikes. |

### LoRA

| field | default | notes |
|---|---|---|
| `lora_r` | `16` | Rank. Alpha is set to 2× rank per current practice. |
| `lora_alpha` | `32` | LoRA scaling factor. |
| `lora_dropout` | `0.05` | Applied on LoRA-A output during training. |

### Session structure

| field | default | notes |
|---|---|---|
| `session_training` | `False` (config default) | Master switch for cross-item carry at training time. CLI `--session 1` overrides. |
| `session_papers_min` | `2` | Multi-paper session lower bound. |
| `session_papers_max` | `6` | Multi-paper session upper bound. |
| `slice_prob` | `0` | Probability a paper is sliced into token-range sub-papers within a session. `0` disables slicing. |
| `slice_min` | `2` | Slice count lower bound when slicing triggers. |
| `slice_max` | `6` | Slice count upper bound. |
| `slice_min_tokens` | `1024` | Minimum tokens per slice (guards against sub-chunk-size slices). |
| `single_paper_sessions` | `False` (config default) | Sessions = one paper cut into k pieces. CLI `--mode single` overrides. |
| `single_paper_slices_min` | `2` | When single-paper mode is on. |
| `single_paper_slices_max` | `6` | When single-paper mode is on. |
| `hybrid_sessions` | `False` | Per-doc split by length: short docs → single-item no-carry, long docs → k-slice carry. CLI `--mode hybrid` overrides. Takes precedence over `single_paper_sessions`. |
| `hybrid_carry_min_tokens` | `3000` | X threshold: docs shorter than this get a single-item session; longer docs get sliced. |
| `hybrid_slices_min` | `2` | y: min slices for the carry path. |
| `hybrid_slices_max` | `6` | z: max slices for the carry path. |
| `hybrid_slice_min_tokens` | `800` | n: min tokens per slice. Must satisfy `y*n <= hybrid_carry_min_tokens`. |
| `source_preset` | `""` | If set, rebalance training rows by source using the named preset (`SOURCE_PRESETS`). Empty = keep natural dataset mix. See [training.md#source-balancing](training.md#source-balancing---source-preset). Presets that ship: `slim-paper`, `slim-research`. |

### Loss mask (all off by default)

| field | default | notes |
|---|---|---|
| `loss_mask_enabled` | `False` | Master switch. |
| `loss_mask_keep_fraction` | `0.5` | Positions with a token in the "kept" fraction get CE; the rest get `-100`. Interpreted on frequency: keep the rarest tokens up to the frac. |
| `loss_mask_protect_terms` | `LOSS_MASK_DEFAULT_PROTECT_TERMS` | Domain terms force-unmasked (expanded into BPE variants). Pass `()` to disable. |
| `loss_mask_reference_counts_path` | `"/ckpt/loss_mask/reference_wikitext103.pt"` | External-reference frequency file. If missing, falls back to in-corpus frequency (with a warning). |
| `loss_mask_protect_numeric` | `True` | Force-unmask pure-digit tokens (scientific content). |
| `loss_mask_protect_symbols` | `("=", "@", "^", "_", "\\", "+", "-", "*", "/", "|", "<", ">")` | Force-unmask math symbols. |
| `loss_mask_first_tokens` | `16` | Mask the first N tokens of paper-start items (boilerplate skipping). Mid-paper slices unaffected. `0` disables. |

### Misc + telemetry

| field | default | notes |
|---|---|---|
| `seed` | `42` | Torch and numpy seeded. |
| `log_every` | `10` | Cadence for lightweight metrics (loss, grad norms, state ratio). |
| `save_every` | `200` | Steps between checkpoints. |
| `run_name` | `"ttt-v1.1"` | wandb run name prefix and ckpt subdirectory. |
| `wandb_enabled` | `True` | Master telemetry switch. |
| `wandb_project` | `"inplace-ttt"` | wandb project. |
| `param_log_every` | `50` | Cadence for heavier metrics (`health/*`, param drift, gate stats). |

### In-loop eval

| field | default | notes |
|---|---|---|
| `eval_every` | `100` | Steps between in-loop eval. `0` disables. |
| `eval_n_papers` | `3` | Papers per in-loop eval when the dataset has no source column (arxiv). Ignored when `eval_n_papers_per_source > 0` and the dataset has a source column. |
| `eval_n_papers_per_source` | `1` | With a multi-source dataset (SlimPajama): exactly N papers per source per eval. Effective total = `N * n_sources`. Set to 0 to fall back to the flat `eval_n_papers` behavior. |
| `eval_n_slices` | `8` | Slices per paper. |
| `eval_min_tokens` | `2048` | Filter eval holdout to docs with at least this many tokens before sampling. Guards against picking tiny StackExchange posts that can't be sliced. |
| `eval_holdout_seed` | `0` | Deterministic paper selection. |

## CLI flag → override map

`train_modal.py::train` accepts these flags, which override the
corresponding config field:

| CLI flag | overrides |
|---|---|
| `--limit-docs N` | Not a config field. Slices training data to N docs after shuffle. `0` = all. |
| `--num-epochs N` | `num_epochs` |
| `--grad-accum N` | `grad_accum_steps` |
| `--session 0\|1` | `session_training` |
| `--mode multi\|single\|hybrid` | `single_paper_sessions` + `hybrid_sessions` (mode dispatch). Empty string leaves defaults. Unknown mode raises. |
| `--min-doc-tokens N` | `min_doc_tokens` |
| `--hybrid-carry-min N` | `hybrid_carry_min_tokens` |
| `--hybrid-slice-min N` | `hybrid_slice_min_tokens` |
| `--hybrid-slices-min N` | `hybrid_slices_min` |
| `--hybrid-slices-max N` | `hybrid_slices_max` |
| `--eval-n-papers N` | `eval_n_papers` |
| `--eval-n-papers-per-source N` | `eval_n_papers_per_source` |
| `--eval-min-tokens N` | `eval_min_tokens` |
| `--source-preset NAME` | `source_preset` (validated at parse time; unknown name raises fast). |
| `--resume-from PATH` | Not a config field. `step_<n>` or `<other_run>/step_<n>`. |

Any int flag == 0 means "leave the config default unchanged." `mode`
uses the empty string as its "unchanged" sentinel; `session` uses -1.

Any field NOT listed here can only be changed by editing
`ttt_config.py`.

## Env vars

Set these in the shell that invokes `modal run` — the Modal image
definition captures them from your local env and forwards them into
the container. So:

```bash
TTT_DATASET=slimpajama-6b modal run --detach train_modal.py::train ...
```

works: `TTT_DATASET` is captured at image-definition time (which runs
locally when you invoke `modal run`) and set as a container env var
via `image.env(...)`. The container then reads it at `ttt_config`
import time. Without this forwarding step, the container would fall
back to the default because it doesn't inherit your laptop's env.

The forwarded set is `_FORWARD_ENV_KEYS` in `train_modal.py` /
`infer_modal.py` — currently `TTT_DATASET, TTT_MODEL_SIZE,
TTT_LAYER_STRIDE, TTT_LAYER_START, TTT_BASE_MODEL`. To forward more,
add them to that tuple.

| var | default | meaning |
|---|---|---|
| `TTT_DATASET` | `"arxiv"` | One of the registered specs in `DATASETS`. Currently `"arxiv"` or `"slimpajama-6b"`. See [data.md](data.md#dataset-selection-dataset_spec). |
| `TTT_MODEL_SIZE` | `"0.6B"` | Qwen3 size: `0.6B`, `1.7B`, `4B`, `8B`. |
| `TTT_BASE_MODEL` | derived from `TTT_MODEL_SIZE` | Full override for the HF repo id (non-Qwen3 paths). |
| `TTT_LAYER_STRIDE` | `2` | Every stride-th layer gets a TTT MLP. Bump to `4` at 8B for memory. |
| `TTT_LAYER_START` | `1` | Index of the first TTT layer. |
| `WANDB_API_KEY` | (Modal secret) | wandb telemetry. |
| `HF_TOKEN` | (Modal secret) | HF token for gated repos. |
| `HF_HOME` | `/hf-cache` | Set in the Modal image so downloads route to the shared volume. |

## Sensitivity notes

**Don't bump `eta` like an LR.** It's the inner-loop update magnitude.
Bumping it scales the applied fast-weight delta multiplicatively per
chunk, which quickly saturates the clip and kills the mechanism. If
gradients are small, bump the LRs.

**Don't raise `clip_tau` on a trained model.** The model calibrates
its direction-quality expectations to the clip during outer training.
Raising `clip_tau` at inference exposes it to unclipped magnitudes it
wasn't trained for.

**`carried_decay` matters most on long sessions.** With `1.0`, state
grows linearly with items. `0.9` puts a hard plateau around
`per_item_delta / 0.1`. Pick based on your expected session length —
tighter decay for chat (many turns), looser for short paper-batch eval.

**LoRA rank sensitivity:** `r=16` is enough to add several ppl of
gain on Qwen3-0.6B ML papers. Bigger ranks marginally help; the
dominant lever is training data.

## Per-knob failure modes

For every important knob, what "goes wrong" and how you'd see it in
metrics.

### `chunk_size`

- **Too small (e.g. 8):** k = N/C explodes, scan-tensor memory (`k × d
  × d_ff × 2`) blows the GPU. Also, the fast-weight update becomes
  noisier per-chunk because there are only C tokens' worth of samples.
- **Too big (e.g. 2048):** the update is coarse. Fewer chunks means
  fewer opportunities for the fast weight to reflect what was just
  read. State only updates ~8 times per 16k forward.
- **Symptom of bad choice:** OOM (too small) or flat `state_ratio`
  growth (too big).

### `eta`

- **Too small:** the fast weight barely moves the down-projection.
  `gap ≈ 0` even though `state_ratio > 0`.
- **Too big:** clip saturates immediately (`state_ratio` pegged at
  `clip_tau / ||W_down||_F`). Direction-only regime kicks in fast, and
  the model wasn't trained for that magnitude regime.
- **Symptom:** `state_ratio` at bounded plateau, `gap` flat or
  slightly negative.

### `conv_kernel_size`

- **`K = 1`:** conv becomes a per-position transformation, no temporal
  context. V uses only the current position of X0.
- **`K` bigger than useful window:** each V position sees a broader
  temporal average, smoothing per-position information.
- **With `v_source="hidden_state"`, sweet spot is 4-8.** With
  `"embedding"`, 16-32 (embeddings are less temporally
  differentiated).

### `output_gate_bias_init`

- **Too negative (e.g. `-5`):** initial gate ~0.007, TTT contribution
  ~1% of raw at init. Dead basin barely receives gradient — hard
  escape.
- **Too positive (e.g. `+1`):** initial gate ~0.73, TTT contribution
  ~73% of raw at init. `sanity_check` still passes (`W_target = 0`
  gives `ttt_out = 0`), but as soon as `W_target` moves off zero, the
  gate lets big TTT contributions through untamed. Loss can spike.
- **Default `-2` (~0.12):** balances "let some signal through" with
  "don't blow up at first optimizer step."

### `clip_tau`

- **Too small (e.g. 1):** most chunks are clipped. Direction-only
  regime from the start. Gap may be flat because information-per-chunk
  is truncated.
- **Too big (e.g. 50):** clip almost never fires; `state_ratio` grows
  unbounded within long sessions. Applied signal becomes huge; loss
  destabilizes.
- **Changing at inference:** don't. The model calibrates its
  expectations to the training-time clip. Verified empirically.

### `carried_decay`

- **`1.0` (default):** pure accumulation. `state_ratio` grows without
  bound over long sessions. Clip eventually bounds magnitude but keeps
  state in direction-only regime.
- **`0.90-0.95`:** bounded plateau ≈ `per_item_delta / (1 - decay)`.
  Recommended for long sessions.
- **`< 0.8`:** aggressive forgetting. State effectively lasts a few
  items. Turns into "last-item-only" memory at `0.0`.

### `session_training`

- **`False`:** items are independent. Cross-item carry is not trained.
  Model may still work in session-eval, but training never saw the
  cross-item distribution.
- **`True` in training but no session structure at eval:**
  `session_perplexity` still works, but the carry never accumulates
  because each session is one item.
- **Recommended:** train with `True` when you plan to eval with
  session mode.

### `single_paper_sessions`

- **`True`:** every session is one paper, sliced. Clean signal, but
  training never sees "unrelated paper follows unrelated paper."
- **`False`:** multi-paper sessions with `k ∼ U[papers_min,
  papers_max]`. Noisier but distributionally-similar to `holdout_eval`
  default.

### `max_grad_norm`

- **Too small (e.g. `1.0`):** clip fires every step during the
  dead-basin-escape spike; `grad/new` gets scaled to ~zero and the
  mechanism can't wake up.
- **Too big (e.g. `100`):** ineffective as a safety net; a single bad
  batch's large gradient pollutes momentum.
- **`10.0`** (default) balances both.

### `loss_mask_enabled`

- **`False`:** all positions contribute. Slower per-step domain
  learning (function words dominate gradient) but no risk of
  mis-masking.
- **`True` without reference file:** falls back to in-corpus frequency,
  which will mask ML-glue words (model, training, layer, function).
  Kills the training signal on the exact vocabulary the model is
  supposed to learn.
- **`True` with reference file and full protect list:** the intended
  configuration. Concentrates gradient on content tokens.

### `loss_mask_keep_fraction`

- **`1.0`:** disables mask entirely (nothing masked).
- **`0.5` (default):** mask covers the tokens accounting for 50% of
  reference occurrences. Balanced.
- **`< 0.3`:** aggressive. May mask domain terms even after protect
  passes if the protect list is incomplete. Verify with
  `diagnose_loss_mask`.

### `loss_mask_first_tokens`

- **`0`:** don't mask paper-start boilerplate. Loss on "# Introduction"
  etc. counts.
- **Too high (e.g. 500):** mask real content. Papers vary in how much
  boilerplate they have; `16` is a safe conservative default.

---

## Recipe: minimal safe change

For any config change, work through this checklist:

1. **Identify the invariant.** Every change to `TTTConfig` breaks
   backward-compat for old checkpoints unless the change is additive.
   Renaming or removing a `TTT_CFG` field means old `ttt_params.pt`
   files won't load without a shape-check failure.
2. **Update the doc.** Every field with a code default has a table row
   here. Keep them in sync — if you change the default, update this
   file at the same commit.
3. **Update the tests.** If the field affects behavior tested in
   [tests/](../tests/), the test either needs to override the field
   (via `dataclasses.replace(cfg, ...)`) or should be updated to match
   the new default.
4. **Run `sanity_check`.** Any config change that could affect the
   identity property (W_target init, TTT layer set, conv init) must
   pass `modal run train_modal.py::sanity_check` before you train.
5. **Run the test suite.** `python -m pytest tests/ -q`. If it fails,
   diagnose before proceeding.

## Related docs

- [architecture.md](architecture.md) — where these fields flow
- [training.md](training.md) — how config drives the training loop
- [scaling.md](scaling.md) — retuning across model sizes
