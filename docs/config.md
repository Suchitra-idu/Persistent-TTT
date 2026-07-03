# Configuration Reference

Every knob in [`ttt_config.py`](../ttt_config.py), plus the env-var
level constants. Grouped by concern.

## Environment variables (read at import time)

Set before `modal run` — they're captured into module-level constants
when `ttt_config` is imported.

| var | default | effect |
|---|---|---|
| `TTT_MODEL_SIZE` | `"8B"` | Substitutes into `BASE_MODEL = f"Qwen/Qwen3-{SIZE}"`. Valid: `"0.6B"`, `"1.7B"`, `"4B"`, `"8B"`. |
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
| `chunk_size` | `400` | Tokens per fast-weight update. Changing requires retraining. Smaller = finer resolution + more memory. At 8B, keep ≥400 to fit. |
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
| `single_paper_sessions` | `False` (config default) | Sessions = one paper cut into k pieces. CLI `--single-paper 1` overrides. |
| `single_paper_slices_min` | `2` | When single-paper mode is on. |
| `single_paper_slices_max` | `6` | When single-paper mode is on. |

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
| `eval_n_papers` | `3` | Papers per in-loop eval. |
| `eval_n_slices` | `8` | Slices per paper. |
| `eval_holdout_seed` | `0` | Deterministic paper selection. |

## CLI flag → override map

`train_modal.py::train` accepts these flags, which override the
corresponding config field:

| CLI flag | overrides |
|---|---|
| `--limit-docs N` | Not a config field. Slices training data to N docs. `0` = all. |
| `--num-epochs N` | `num_epochs` |
| `--grad-accum N` | `grad_accum_steps` |
| `--session 0\|1` | `session_training` |
| `--single-paper 0\|1` | `single_paper_sessions` |

Any field NOT listed here can only be changed by editing
`ttt_config.py`.

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

## Related docs

- [architecture.md](architecture.md) — where these fields flow
- [training.md](training.md) — how config drives the training loop
- [scaling.md](scaling.md) — retuning across model sizes
