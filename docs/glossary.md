# Glossary

Terms used throughout the docs. When in doubt, look here first.

## Core mechanism terms

**Fast weight (`S`, `carried_delta`, `state.delta`).** A rank-`d` matrix
of shape `[d, d_ff]` that gets added to `W_down` at apply time:
`W_effective = W_down + eta · S`. Not a `Parameter` — it's a runtime
state derived from `V^T @ Z` per-chunk. Has multiple names depending on
context: `S` in math, `carried_delta` in session-mode training,
`state.delta` in streaming mode.

**Slow weight.** The base transformer weights, LoRA adapters, and
TTT-layer `W_down`, `W_target`, `target_conv`, `output_gate`. Trained
by the outer optimizer via SGD. Don't change during inference (unlike
fast weights).

**In-place TTT.** The fast weight lives *inside* the MLP block —
specifically, it modifies the `down_proj`. Contrast with the paper
variant where TTT is a separate module inserted between transformer
layers.

**Scan path (`_scan_forward`).** Parallel-chunk implementation of the
mechanism. All `k = N/C` chunk updates computed via two einsums.
Efficient for many-token forward passes. Used for training and
whole-sequence eval.

**Stream path (`_stream_forward`).** Incremental implementation.
Chunks commit as they fill; the fast weight updates chunk-by-chunk.
Used for autoregressive generation where future tokens aren't
available.

**Chunk / chunk_size (`C`).** The number of tokens per fast-weight
update. Default 50. Sequence of length `N` produces `k = ceil(N / C)`
chunks. Chunk `i` sees only fast weights from chunks `< i`
("exclusive cumsum").

**Session.** A list of `SessionItem` triples processed with a shared
carry. `reset_session_state` at the start; `advance_session_state`
after each item.

**Session item / `SessionItem(doc_idx, start, end)`.** One
"batch-of-1" unit within a session. Represents a contiguous token
range from a document. Multiple items can come from the same document
(slicing) or different documents (multi-paper session).

**Carry.** Shorthand for `carried_delta` — the accumulated fast weight
inherited from prior items in a session. Detached at the item boundary
(TBPTT).

**TBPTT (Truncated Backprop Through Time).** Gradient is restricted to
flow only within one item's forward-backward. The `.detach()` when
staging `_next_carried` is what enforces this. The model can *use* the
carry it receives but doesn't learn to plan cross-item updates.

**Session mode (`session_mode`).** Master toggle on each TTT module.
When True, `_scan_forward` reads `S_0 = carried_delta` at start and
stages `_next_carried` at end. When False, per-forward `S_0 = 0` and
nothing carries.

**Hybrid sessions (`hybrid_sessions`).** Session-building mode where
short docs (< `hybrid_carry_min_tokens`) become single-item sessions
with no slicing/carry, and long docs get sliced into k pieces with
carry. Intended for diverse-length pretraining mixes (SlimPajama).
See [training.md#hybrid-sessions](training.md#hybrid---for-diverse-length-pretraining-mixes).

**Streaming mode (`stateful`).** Toggle on each TTT module + the tap.
When True, `_stream_forward` runs instead of `_scan_forward`. The
mechanism accumulates `state.delta` across forward calls.

**`ttt_evolve`.** Freeze/unfreeze the fast-weight update. `False` in
either mode means: apply current fast weight, don't update it. Used by
`evolve=False` in inference to isolate "just LoRA" behavior.

## Wiring / architecture terms

**LoRA target regex.** The pattern (in `build_lora_target_regex`) that
determines which `Linear` layers get wrapped by a LoRA adapter.
Explicitly excludes `down_proj` on TTT layers.

**Param groups (`lora`, `wdown`, `new`).** The three-way split used by
`build_param_groups`. Each has its own LR. See
[training.md#param-groups](training.md#param-groups).

**`TTT_PARAM_MARKERS`.** Substrings that identify TTT-owned
parameters: `("target_conv", "w_target", "output_gate",
"v_source_norm")`. Used by both `_classify_ttt_param` and
`save_ttt_state_dict`. Adding a new module without adding its name to
this list = silent misclassification.

**Layer indices (`layer_indices`, `LAYER_STRIDE`, `LAYER_START`).**
The subset of transformer layers that get TTT MLPs. Default: every
2nd layer starting at index 1. Stride 4 for 8B for memory.

**`v_source`.** Where `X0` (input to `target_conv`) comes from.
`"embedding"` uses `embed_tokens` output (shared across layers).
`"hidden_state"` (default) uses per-layer input hidden state.

**Tap (`EmbeddingTap`).** Small helper object that provides `X0` on
demand during forward. For `v_source="embedding"`, it's a real
`forward_hook` on `embed_tokens`. For `"hidden_state"` it holds the
shared `stateful` flag; per-layer X0 is provided by the layer itself.

## Optimization terms

**Grad accumulation (`grad_accum_steps`).** Number of micro-steps
(items) per optimizer step. Each micro-step divides its loss by
`grad_accum_steps` before backward; every `grad_accum_steps` micros
triggers one `optimizer.step()`.

**Dead basin.** The initial state where `W_target = 0` means `V = 0`
means the fast weight stays zero throughout the forward. Gradient
signal to `W_target` is small until it moves off zero, at which point
the mechanism "wakes up." Also called *cold start*.

**State ratio (`state_ratio_mean`).**
`||eta · S||_F / ||W_down||_F`. Dimensionless. Tells you how large the
fast weight is *relative to* the down-projection it's modifying.
Bounded above by `clip_tau / ||W_down||_F` when the clip is active.

**Frobenius clip.** Per-chunk scaling of `S` so that
`||eta · S||_F ≤ clip_tau`. Preserves direction; caps magnitude.

**Direction-only regime.** When the clip is active most of the time,
`||eta · S||_F` is pegged at `clip_tau`; the model's fast-weight
signal is entirely in its direction, not magnitude.

## Eval terms

**Carry / carry-off / fresh.** Three modes in `session_perplexity` and
`run_holdout_eval`, all run on the same items:
- **carry** — `evolve=True`, fast weight persists across items (full TTT).
- **carry-off** — `evolve=True`, `reset_session_state` called between
  items. Chunk-scan still fires *within* an item's forward, but nothing
  carries across item boundaries. Isolates within-item adaptation.
- **fresh** — `evolve=False`, fast weight = 0 throughout. TTT completely
  silent.

**Δwithin / Δbetween / Δtotal.** Three ways to read the three-mode
output. `Δwithin = fresh - carry-off` is the within-item chunk-scan
benefit. `Δbetween = carry-off - carry` is the cross-item persistence
benefit. `Δtotal = fresh - carry = Δwithin + Δbetween`. Positive = TTT
helping. The decomposition tells you *where* the benefit comes from.

**Three-way eval.** BASE (no adapter, no TTT) / LORA-ONLY (adapter +
`load_ttt=False`) / FULL (adapter + TTT). Three separate model loads,
three separate tables. Isolates the contribution of each component.

**Holdout.** The newest `HOLDOUT_LAST_N` (default 200) papers in the
dataset, reserved for eval — never seen during training. Enforced by
`split_holdout`.

**Token-weighted ppl.** Per-paper perplexity aggregation as
`exp(sum(log(ppl_slice) · n_tok_slice) / sum(n_tok_slice))`. The
correct aggregation for perplexity because ppl is a geometric mean of
token-level probabilities.

## Loss-mask terms

**Loss mask.** A boolean vector of shape `[vocab_size]` marking which
token IDs get `label = -100` (ignored in CE). Built from unigram
frequency counts.

**Reference counts.** Precomputed unigram counts from an external
corpus (wikitext-103) used as the "what's common in general English"
baseline. Alternative to in-corpus counting which would drop domain
vocabulary.

**Protect pass.** A pass that force-unmasks specific token IDs after
the frequency mask is built. Three passes: domain terms (Transformer,
LSTM, ...), pure-digit tokens, math symbols.

**`first_tokens`.** Mask the first N tokens of paper-start items only.
Skips boilerplate. Mid-paper slices are unaffected.

## Chat terms

**Turn.** One `user → assistant` exchange. `chat_turn` handles one
turn. Sends the user's new message through the model with no re-fed
prior turns.

**Turn boundary.** The transition between two turns. At the boundary:
KV cache dropped, conv left-context reset. `state.delta` + pending
PERSIST.

**Snapshot.** A `dict[layer_idx, Tensor]` dump of every TTT layer's
`state.delta`, saved to `/ckpt/<run>/sessions/<name>.pt`. Loadable via
`chat_reset(from_snapshot_name=...)`.

**Thinking.** The `<think>...</think>` section produced by Qwen3
thinking mode. Disabled by default in `chat_client.py` because our
training data has no thinking traces.

## File terms

**`ttt_params.pt`.** Serialized TTT tensors (`W_target`,
`target_conv`, `output_gate`, TTT-layer `down_proj`), one per
checkpoint. Keys stripped of `base_model.model.` prefix so load works
with or without PEFT wrap.

**`adapter/`.** PEFT-format LoRA adapter directory
(`adapter_config.json` + `adapter_model.safetensors`). Written by
`model.save_pretrained(...)` and read by `PeftModel.from_pretrained(...)`.

**`reference_wikitext103.pt`.** Precomputed unigram counts + metadata
for the loss mask. Built by `build_reference_counts`.

## Dataset terms

**`DatasetSpec`.** Frozen dataclass in `ttt_config.py` describing one
dataset: `source` (HF repo id or local dir), `text_column`, optional
`tokens_est_column`, optional `source_meta_column` + `source_meta_key`
+ `include_sources` for per-row source labeling and filtering, and
`holdout_last_n` for the eval split. Everything downstream reads
`DATASET_SPEC` rather than individual column names.

**`DATASET_SPEC`.** The active spec, selected by the `TTT_DATASET`
env var at import time. `DATASETS[TTT_DATASET]`. Currently `"arxiv"`
or `"slimpajama-6b"`.

**Source column.** A top-level `source` column added by
`_annotate_source` when the spec declares `source_meta_column` +
`source_meta_key`. For SlimPajama it holds the RedPajama subset name
(`"RedPajamaC4"`, `"RedPajamaGithub"`, ...). Used for per-source eval
aggregation and stratified holdout sampling.

**`include_sources`.** Tuple on `DatasetSpec`; keeps only rows whose
extracted source label is in this set. How we exclude CommonCrawl
from SlimPajama training.

**Source preset (`source_preset`, `SOURCE_PRESETS`).** Named
per-source ratio dict used to rebalance the training pool at load
time. `slim-paper` restores SlimPajama-627B's advertised
proportions; `slim-research` downweights C4 for cleaner per-domain
gap signal. Applied by `_balance_by_source_preset` in
`train_modal.py` before `--limit-docs` takes effect. See
[training.md#source-balancing](training.md#source-balancing---source-preset).

**Per-source eval.** In `run_holdout_eval` and
`_print_per_source_summary`, per-domain token-weighted PPL keyed by
source label. The `eval/<source>/gap` numbers are the "which domain
benefits most from TTT" signal.

## Env-var terms

**`TTT_DATASET`.** Chooses which registered dataset to use. Default
`"arxiv"`. Env var read at `ttt_config` import.

**`TTT_MODEL_SIZE`.** Chooses which Qwen3 size to load. Default
`"0.6B"`. Env var read at `ttt_config` import.

**`TTT_LAYER_STRIDE`.** How dense the TTT layers are. Default 2 (every
second layer). 4 for 8B.

**`TTT_LAYER_START`.** Where TTT layers start. Default 1.

**`TTT_BASE_MODEL`.** Full override for the base model repo id
(non-Qwen3 paths).

**`WANDB_API_KEY`.** Enables wandb telemetry. Comes from the Modal
`wandb` secret.

**`HF_TOKEN`.** For gated model / dataset access. Comes from the Modal
`huggingface` secret.

**`HF_HOME`.** Set to `/hf-cache` in-container so downloads route to
the shared Modal volume.
