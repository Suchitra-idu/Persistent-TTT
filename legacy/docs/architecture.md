# Architecture

Module boundaries, dataflow at every level, and the design rationale
for each choice. This doc explains **how the codebase is put together**;
[mechanism.md](mechanism.md) explains **what the mechanism computes**.

## Contents

1. [Design goals](#design-goals)
2. [Module map](#module-map)
3. [Per-module responsibilities](#per-module-responsibilities)
4. [Import graph](#import-graph)
5. [How Modal ships this code](#how-modal-ships-this-code)
6. [Model assembly order](#model-assembly-order)
7. [Training dataflow, step by step](#training-dataflow-step-by-step)
8. [Inference dataflows](#inference-dataflows)
9. [Chat dataflow (streaming)](#chat-dataflow-streaming)
10. [Wiring contracts](#wiring-contracts)
11. [Filesystem layout at runtime](#filesystem-layout-at-runtime)
12. [Related docs](#related-docs)

---

## Design goals

Four principles drive the whole layout. Every architectural choice
below is justifiable by one of these.

### 1. Mechanism is Modal-free

[`inplace_ttt.py`](../inplace_ttt.py) and
[`ttt_wiring.py`](../ttt_wiring.py) have **zero Modal imports**.
This makes the entire TTT layer unit-testable on CPU with just
PyTorch + numpy, and reusable outside Modal. The test suite exercises
the full mechanism on tiny fake modules; see
[testing.md](testing.md).

If someone wants to run this on their own inference cluster, they can
copy the four pure modules (`inplace_ttt.py`, `ttt_wiring.py`,
`ttt_config.py`, `model_setup.py`) and build their own runtime around
them.

### 2. Config is single-source

Every constant lives in [`ttt_config.py`](../ttt_config.py). Constants
that duplicate across files rot: someone updates one and forgets the
other; nobody notices until an experiment produces mystifying results.
A single dataclass makes the config canonical. If the value has to
appear somewhere else, it must be *derived* from `TTT_CFG` or
`TRAIN_CFG`, not restated.

Exception: environment-variable-driven constants
(`TTT_MODEL_SIZE`, `TTT_LAYER_STRIDE`, `TTT_LAYER_START`) are read at
`ttt_config.py` import time, so they need to be set before any import
of `ttt_config`. Modal reads them from the local environment at
serialization time.

### 3. Train and inference build the model identically

[`model_setup.py`](../model_setup.py) contains **one** function,
`build_model`, called from both [`train_modal.py`](../train_modal.py)
and [`infer_modal.py`](../infer_modal.py). There's exactly one way to
assemble the base model + TTT patch + LoRA wrap + checkpoint load.

Divergence between train-time and inference-time model assembly is one
of the highest-cost bug classes: it produces good training metrics but
broken serving. Forcing both codepaths through the same function
eliminates the possibility.

### 4. Session vs stream, one method each

The mechanism has two execution modes: parallel scan
(`_scan_forward`) for training and whole-sequence eval, and
incremental stream (`_stream_forward`) for autoregressive generation.
Each is its own method — no shared branchy code.

The two paths are mathematically equivalent when applied to the same
token stream (see
[mechanism.md](mechanism.md#scan--stream-equivalence)), and their
equivalence is unit-tested. If they diverge, the tests catch it.

---

## Module map

```
Our-TTT/
├── ttt_config.py       config dataclasses + env-var-driven constants
├── inplace_ttt.py      the mechanism (pure PyTorch, no LoRA/PEFT deps)
├── ttt_wiring.py       LoRA regex, param groups, checkpoint I/O
├── model_setup.py      one function: assemble model → TTT patch → LoRA
├── data_utils.py       dataset loading + holdout split
├── train_utils.py      session schedule, slicing, loss-mask helpers
├── chat_utils.py       chat sampling, template, stop tokens
├── observability.py    wandb wrapper + metric collectors
├── train_modal.py      Modal app for training + sanity_check + support tools
├── infer_modal.py      Modal app for inference + eval + generation
└── chat_client.py      local REPL that talks to deployed TTTInference
```

**Flat structure by design.** Modal ships modules into containers via
`image.add_local_python_source("ttt_config", "inplace_ttt", ...)`,
which imports modules by name from the working directory. Moving files
into `src/` or a subpackage requires updating both the
`add_local_python_source` args and every `import` — silent, easy-to-miss
breakage.

---

## Per-module responsibilities

Below, "owned by" means: this module is the single source of truth for
that concern. If a thing appears in another module, it should be *used
via* the owner, not redefined.

### [`ttt_config.py`](../ttt_config.py)

Owns:

- `TTT_CFG` (a `TTTConfig`) — mechanism knobs.
- `TRAIN_CFG` (a `TrainConfig`) — outer-loop knobs.
- Environment-variable-driven constants: `MODEL_SIZE`, `BASE_MODEL`,
  `LAYER_STRIDE`, `LAYER_START`.
- Volume names and mount paths (`CKPT_VOLUME_NAME`, `CKPT_MOUNT`, etc.).
- Dataset identity (`DATASET_SOURCE`, `TEXT_COLUMN`, `TOKENS_EST_COLUMN`).
- `HOLDOUT_LAST_N` — how many newest papers to reserve.

No Torch imports. Cheap to import from anywhere.

### [`inplace_ttt.py`](../inplace_ttt.py)

Owns:

- `InPlaceTTTMLP` — the layer class.
- `EmbeddingTap` — the tap.
- `TTTState` — dataclass for per-module streaming state.
- All reset helpers (`reset_fast_weights`, `reset_v_left_context`,
  `reset_session_state`).
- `advance_session_state`.
- Diagnostic helpers (`state_norms`, `mean_state_ratio`,
  `gate_reg_term`, `gate_stats`, `stream_pending_progress`).
- Snapshot / restore of streaming fast weights
  (`export_fast_weights`, `import_fast_weights`).

Imports only PyTorch and `ttt_config.TTTConfig`. Explicitly does **not**
import: PEFT, Modal, HF transformers, or anything else.

### [`ttt_wiring.py`](../ttt_wiring.py)

Owns:

- `build_lora_config` and its target regex generator
  `build_lora_target_regex` — the LoRA target list.
- `build_param_groups` — the three-way param split (`lora`, `wdown`,
  `new`).
- `unfreeze_ttt_params` — PEFT freezes everything non-LoRA; this
  re-enables grads on the TTT trainables.
- Checkpoint I/O (`save_ttt_state_dict`, `load_ttt_state_dict`).
- `TTT_PARAM_MARKERS` — the substrings that identify TTT parameters by
  name.

Imports Torch and PEFT (lazily). This is where the "LoRA never lands on
a TTT-layer down_proj" invariant is enforced. The invariant is
unit-tested in [`test_wiring.py`](../tests/test_wiring.py).

### [`model_setup.py`](../model_setup.py)

Owns the model assembly pipeline. Exactly one exported function:

```python
def build_model(
    adapter_path: str | None = None,
    ttt_ckpt_path: str | None = None,
    trainable: bool = True,
    attn_impl: str = "flash_attention_2",
) -> tuple[nn.Module, PreTrainedTokenizer]:
```

Assembly order:

1. Load base Qwen3 (`AutoModelForCausalLM.from_pretrained`).
2. Derive `TTT_CFG.layer_indices` from `num_hidden_layers` if unset.
3. `patch_model_with_ttt(model, TTT_CFG)` — replace `.mlp` on TTT
   layers.
4. Either `PeftModel.from_pretrained(model, adapter_path)` (resume /
   inference) or `get_peft_model(model, lora_cfg)` (fresh train).
5. If training: `unfreeze_ttt_params(model, TTT_CFG)` to re-enable TTT
   trainables that PEFT froze.
6. If `ttt_ckpt_path` exists: `load_ttt_state_dict(model,
   ttt_ckpt_path)`.
7. Return `(model, tokenizer)`.

The order **matters**. See [Model assembly order](#model-assembly-order)
below.

### [`data_utils.py`](../data_utils.py)

Owns:

- `open_dataset()` — load the parquet dataset from HF Hub or a local
  path. Handles both `load_dataset` (parquet shards) and
  `load_from_disk` (arrow directory).
- `split_holdout(ds)` — reserve the last `HOLDOUT_LAST_N` rows for
  contamination-free eval.

**Why this lives in a shared module:** train and inference both need to
compute the holdout boundary. If each computed it independently and
one drifted, holdout papers would leak into training.

### [`train_utils.py`](../train_utils.py)

Owns:

- `SessionItem` dataclass and session schedule builders
  (`make_session_schedule`, `make_slice_sessions`, `slice_doc`,
  `equal_token_slices`, `expected_items_per_doc`).
- Loss-mask primitives (`count_unigrams`, `common_mask_from_counts`,
  `apply_loss_mask`, `protect_by_predicate`, `protect_numeric_tokens`,
  `protect_symbol_tokens`, `protect_token_ids`,
  `apply_protect_passes`, `load_reference_counts`).
- `LOSS_MASK_DEFAULT_PROTECT_TERMS` — the domain-terms list.

Pure Python (Torch imports deferred inside functions). CPU-testable.

### [`chat_utils.py`](../chat_utils.py)

Owns:

- `sample_top_p` — nucleus + top-k + temperature sampling on `[1, V]`
  logits.
- `chat_stop_token_ids` — the set of token IDs that end a chat turn
  (special tokens + `eos_token_id` + `pad_token_id`).
- `strip_chat_specials` — remove ChatML markers from a decoded string.
- `split_thinking` — parse `<think>…</think>` sections.

Pure Python. Deliberately kept out of the training image (the training
container doesn't need chat).

### [`observability.py`](../observability.py)

Owns:

- `Telemetry` — thin wandb wrapper. Failure-proof: broken wandb
  degrades to console print; no exception ever propagates.
- `gpu_stats` — three CUDA memory metrics.
- `param_health` — heavier-metric collector (drift, gate stats, LoRA
  norm) for periodic logging.

Failure-proof means: telemetry failures scope to `wandb.Error`, so
programmer bugs (`AttributeError`, `TypeError`) *do* propagate.

### [`train_modal.py`](../train_modal.py)

Modal app for training. Contains:

- The Modal `Image` (Python 3.11, Torch 2.8, transformers, PEFT,
  flash-attn wheel).
- `train` — the training function (Modal `@app.function`).
- `sanity_check` — GPU-side identity-property verification.
- `build_reference_counts` — precompute the wikitext-103 unigram
  reference for loss masking.
- `diagnose_loss_mask` — print what the mask catches.
- The support helpers: `_apply_cli_overrides`, `_resolve_resume`,
  `_setup_loss_mask`, `_make_epoch_sessions`, `_eval_paper`,
  `_token_weighted_ppl`, `run_holdout_eval`.

No fast-weight logic — that's all inside `inplace_ttt.py`.

### [`infer_modal.py`](../infer_modal.py)

Modal app for inference. Contains:

- The Modal `Image` (Python 3.11, Torch 2.8, transformers, PEFT — no
  wandb, no bnb).
- `TTTInference` — a Modal `@app.cls` with `@modal.method` endpoints:
  `perplexity`, `session_perplexity`, `generate`,
  `fetch_holdout_texts`, `save_session`, `chat_reset`, `chat_turn`.
- Local entrypoints: `compare_ppl`, `holdout_eval`, `session_eval`,
  `generate_cli`, `holdout_generate`, `single_paper_eval`.
- Print helpers: `_print_paper_preview`, `_print_session_results`.
- Three-way eval helper: `_three_way_eval`.

### [`chat_client.py`](../chat_client.py)

Not a Modal file. Runs as a regular local Python process because
`modal run` swallows stdin and can't do interactive REPL. Uses
`modal.Cls.from_name("inplace-ttt-infer", "TTTInference")` to talk to
the deployed `infer_modal` app.

Own concerns: argparse, the REPL loop, and the `/m` / `/save` / `/reset`
/ `/quit` commands.

---

## Import graph

```
                                    ┌──────────────┐
                                    │  ttt_config  │  (no torch)
                                    └──────┬───────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────┐
             ▼                             ▼                         ▼
     ┌──────────────┐             ┌──────────────┐            ┌────────────┐
     │ inplace_ttt  │             │  ttt_wiring  │            │ data_utils │
     └──────┬───────┘             └──────┬───────┘            └─────┬──────┘
            │                            │                          │
            └──────────────┬─────────────┘                          │
                           ▼                                        │
                   ┌──────────────┐                                 │
                   │ model_setup  │                                 │
                   └──────┬───────┘                                 │
                          │                                         │
              ┌───────────┴───────────────────────────────┐         │
              ▼                                           ▼         │
     ┌───────────────────┐                     ┌────────────────────┴──┐
     │   train_modal     │◀── train_utils ────▶│      infer_modal      │
     └───────────────────┘   observability     └────────────────────┬──┘
              ▲                                                      │
              │                                                      │
              │                                    ┌─────────────────┼──────┐
              │                                    ▼                 │      ▼
              │                            chat_utils           chat_client │
              │                                                             │
     (train tests import train_utils, ttt_config)                           │
     (mechanism tests import inplace_ttt, ttt_wiring)                       │
                                                                            │
                                                        (chat_client calls  │
                                                         modal.Cls.from_name│
                                                         to talk to deployed)
```

Key constraints enforced by this graph:

- `ttt_config` is a leaf (no torch, no downstream imports).
- `inplace_ttt` and `ttt_wiring` only depend on `ttt_config` + Torch —
  no Modal, no PEFT (for inplace_ttt).
- `model_setup` is the single fan-in that combines TTT patch + LoRA.
- **No cross-import between `train_modal` and `infer_modal`.** They
  share code only through the pure Python modules. This keeps each
  Modal container image minimal — the training image doesn't ship
  `chat_utils`, and the inference image doesn't ship `train_utils`.

---

## How Modal ships this code

Modal ships local Python source into containers via the image builder.
The pattern:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(...)
    .env({"HF_HOME": HF_CACHE_MOUNT})
    .add_local_python_source(
        "ttt_config", "inplace_ttt", "ttt_wiring",
        "model_setup", "data_utils", "observability", "train_utils",
    )
)
```

Each name in `add_local_python_source` becomes a top-level module in
the container, imported by name. This is why the flat-layout matters:
if you moved these into `src/ttt/`, you'd need to say
`add_local_python_source("src.ttt.inplace_ttt", ...)` — which Modal
supports, but breaks the "just import" ergonomics.

The two apps ship slightly different module sets:

- **Training image:** `ttt_config, inplace_ttt, ttt_wiring,
  model_setup, data_utils, observability, train_utils`. Notably
  **no** `chat_utils`, because training doesn't use it.
- **Inference image:** `ttt_config, inplace_ttt, ttt_wiring,
  model_setup, data_utils, chat_utils, train_utils`. **No**
  `observability` (inference doesn't log to wandb).

`train_utils` is shipped to the inference image because
`session_perplexity` reuses `make_slice_sessions` and `SessionItem` for
slicing eval sessions.

Env vars in the containers:

- `HF_HOME=/hf-cache` — routes HuggingFace's cache to the shared
  Modal volume, so all containers see the same downloaded model
  weights.
- `WANDB_API_KEY` — from the `wandb` Modal secret; training only.
- `HF_TOKEN` — from the `huggingface` Modal secret; required only if
  the base model or dataset is gated.

---

## Model assembly order

`build_model` in [`model_setup.py`](../model_setup.py) does:

```
1. AutoModelForCausalLM.from_pretrained(BASE_MODEL, bf16, sdpa/flash)
2. Derive TTT_CFG.layer_indices from num_hidden_layers (if not set)
3. patch_model_with_ttt(model, TTT_CFG)         ─────┐
                                                     │ ORDER MATTERS
4. Either:                                           │
   a. PeftModel.from_pretrained(model, adapter)      │
   b. get_peft_model(model, lora_cfg)                │
                                                <────┘
5. If trainable: unfreeze_ttt_params(model, TTT_CFG)
6. If ttt_ckpt_path: load_ttt_state_dict(model, ttt_ckpt_path)
```

### Why patch BEFORE LoRA

If we LoRA-wrap *before* patching, PEFT's LoRA adapters end up wrapped
around the `Qwen3MLP` module. Then when we patch, we replace `mlp` on
TTT layers with `InPlaceTTTMLP`, which discards those wrappers — so
LoRA on TTT-layer `gate_proj` / `up_proj` is silently dropped.

Patch first, LoRA second, and the LoRA regex sees the freshly-patched
model's parameters: `InPlaceTTTMLP.gate_proj`, `InPlaceTTTMLP.up_proj`,
and the base `down_proj` — all with the expected names.

### Why unfreeze AFTER LoRA

`get_peft_model` / `PeftModel.from_pretrained` unconditionally freezes
every non-LoRA parameter (that's how PEFT works — only the LoRA layers
train). We need to re-enable grads on `W_target`, `target_conv`, and
`down_proj` (on TTT layers) — that's what `unfreeze_ttt_params` does.

If we unfroze before LoRA, PEFT would re-freeze them, so the order is
forced.

### Why load TTT checkpoint LAST

`load_ttt_state_dict` copies tensor values directly into the model's
parameters. If done before LoRA wrap, PEFT would ignore anything not
matching its expected structure and might refuse to load. Loading last
means the LoRA adapters are already in place; we only fill the TTT
tensors, and the LoRA adapters come from `PeftModel.from_pretrained`.

### Why `attn_impl` differs

- Training uses `flash_attention_2` — needed for the long sequences
  (`max_seq_len=16384`) and gradient checkpointing.
- Inference uses `sdpa` — simpler dependency footprint, doesn't need
  the flash-attn wheel. Slightly slower but avoids build issues.

Both work with the TTT patch; the attention implementation is
orthogonal to what `InPlaceTTTMLP` does.

---

## Training dataflow, step by step

Full pipeline from dataset to optimizer step:

```
HF Hub / local dataset  ──▶  data_utils.open_dataset()

                              ▼

data_utils.split_holdout()  ──▶  (train_ds, holdout_ds)
                                   │              │
                                   │              └── reserved for eval only
                                   ▼

tokenize + drop-short filter  ──▶  ds with input_ids column

                                  ▼

loss-mask setup:
    load_reference_counts(...)
    common_mask_from_counts(...)
    apply_protect_passes(...)
                                  ▼

model, tokenizer = build_model(trainable=True)
                                  ▼

param_groups = build_param_groups(model, ...)
                                  ▼

optimizer = bnb.optim.PagedAdamW8bit(param_groups)
scheduler = cosine_warmup(optimizer, total_steps)
                                  ▼

set every TTT module's session_mode ← cfg.session_training

┌─── for each epoch: ─────────────────────────────────────────────┐
│                                                                 │
│    _make_epoch_sessions(...)  ──▶  list of sessions             │
│                                                                 │
│    ┌── for each session: ─────────────────────────────────────┐ │
│    │                                                          │ │
│    │   reset_session_state(model)                             │ │
│    │                                                          │ │
│    │   ┌── for each SessionItem: ────────────────────────────┐│ │
│    │   │                                                     ││ │
│    │   │   ids = tensor(doc_input_ids[start:end])            ││ │
│    │   │   labels = apply_loss_mask(ids, common_mask, ...)   ││ │
│    │   │   loss = model(input_ids=ids, labels=labels).loss   ││ │
│    │   │   loss += gate_reg_weight * gate_reg_term(model)    ││ │
│    │   │   (loss / grad_accum).backward()                    ││ │
│    │   │   advance_session_state(model)                      ││ │
│    │   │                                                     ││ │
│    │   │   log micro-metrics                                 ││ │
│    │   │   if micro % grad_accum == 0:                       ││ │
│    │   │       clip_grad_norm_(...)                          ││ │
│    │   │       optimizer.step(); scheduler.step()            ││ │
│    │   │       optimizer.zero_grad(set_to_none=True)         ││ │
│    │   │       log train + session metrics                   ││ │
│    │   │       if step % param_log_every == 0: param_health()││ │
│    │   │       if step % save_every == 0: save_checkpoint()  ││ │
│    │   │       if step % eval_every == 0: run_holdout_eval() ││ │
│    │   └────────────────────────────────────────────────────┘│ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Key points:

- **`reset_session_state` at every session boundary.** Fast weights
  start from zero for the new session.
- **`advance_session_state` after every backward.** Idempotent, so the
  gradient-checkpointing recompute doesn't double-stage.
- **Loss mask applies only to paper-start items** (`item.start == 0`)
  for the `first_tokens` mask. Mid-paper slices (`item.start > 0`) skip
  the first-N mask because they're real content.
- **Gradient accumulation is by `micro % grad_accum`.** The
  `optimizer.step()` runs only when `micro % grad_accum == 0`, i.e.
  after every `grad_accum` items. This is independent of session
  boundaries — a session's items may span multiple accumulation windows,
  and vice versa.
- **In-loop eval** snapshots the TTT state, runs eval, and restores.
  Zero side-effect on the training carry.

---

## Inference dataflows

### `holdout_eval` / `session_eval` / `single_paper_eval`

All three call into `session_perplexity` on a `TTTInference` instance.
The differences are in *what texts* they feed:

- `holdout_eval`: pulls `n_papers` random papers from the holdout via
  `fetch_holdout_texts`.
- `session_eval`: reads every `.txt` in a local dir, in sorted order.
- `single_paper_eval`: pulls one holdout paper, slices into
  `n_slices` equal pieces.

All three use `_three_way_eval` when `ckpt` is provided, printing
three tables side-by-side:

- **BASE** (`TTTInference(ckpt="")`) — plain Qwen3, no adapters.
- **LORA-ONLY** (`TTTInference(ckpt=ckpt, load_ttt=False)`) — LoRA
  loaded, TTT tensors *not* loaded (so `W_target = 0` at init).
- **FULL** (`TTTInference(ckpt=ckpt, load_ttt=True)`) — LoRA + TTT.

Each row shows both `carry_ppl` (evolve=True) and `fresh_ppl`
(evolve=False), and their gap.

```
TTTInference.load()  →  build_model(adapter, ttt_ckpt, trainable=False)

                       ▼

fetch_holdout_texts(n_papers, seed)  → list[str]
                       ▼

session_perplexity(texts, evolve=True|False, slice_papers, ...):

    _set_mode(evolve, stateful=False)
    every TTT module .session_mode = True
    reset_session_state(model)

    (build items via SessionItem based on slice_papers / equal_n_slices)

    ┌── for each item: ───────────────────────────────────────┐
    │  ids = tensor(paper_token_ids[item.doc_idx][s:e])       │
    │  with torch.no_grad():                                  │
    │      loss = model(input_ids=ids, labels=ids).loss       │
    │  advance_session_state(model)                           │
    │  record ppl, state_ratio, paper_idx, session_pos        │
    └─────────────────────────────────────────────────────────┘

    (in finally): reset_session_state(model), turn session_mode off

    return list of per-item dicts

The two runs (evolve=True, evolve=False) are then paired position-by-
position for the per-item table, and aggregated to a token-weighted
per-paper table.
```

The `evolve=False` run has `ttt_evolve=False` on every TTT module,
so `_scan_forward` doesn't stage `_next_carried` and `state.delta` is
never updated. But the current carry (which is None because we reset)
still shows in `apply` if any prior item had left `carried_delta`
non-None — hence the `reset_session_state` at the start of each run.

### `generate` (non-chat)

Used by `compare_ppl` and `generate_cli`. Runs the base HuggingFace
`.generate()` method — the streaming path fires internally as tokens
are decoded one at a time.

Optional `fast_weight_snapshot` seeds `state.delta` from a prior
snapshot, allowing cross-session persistence outside of chat.

---

## Chat dataflow (streaming)

Chat is the only path where the streaming `_stream_forward` is
actually used. See [chat.md](chat.md) for the full memory model; the
key architectural fact is:

```
chat_client.py  ──▶  engine.chat_turn.remote(user_message, ...)
                         │
                         ▼
                     _set_mode(stateful=True) or inline flag set
                         │
                         ▼
                     apply Qwen3 chat template to user_message
                     new_ids = tokenizer(new_text, ...).input_ids
                         │
                         ▼
                     prefill: model(input_ids=new_ids, use_cache=True)
                     past_kv = out.past_key_values     # within-turn only
                         │
                         ▼
                     ┌── for _ in range(max_new_tokens): ────┐
                     │   next_id = sample_top_p(next_logits) │
                     │   if next_id in stop_ids: break       │
                     │   out = model(next_id, past_kv, use_cache=True)
                     │   past_kv = out.past_key_values       │
                     │   next_logits = out.logits[:, -1, :]  │
                     └───────────────────────────────────────┘
                         │
                         ▼
                     reset_v_left_context(model)   # soft turn boundary
                         │
                         ▼
                     decode + strip specials + split thinking
                         │
                         ▼
                     return dict(text, thinking_text, state_ratio_mean,
                                 pending_tokens, chunk_size, ...)
```

The **only** cross-turn state carrier is the TTT streaming state
(`state.delta` + `pending_z / pending_v`). The KV cache and conv
left-context are dropped at the turn boundary. This means:

- The model can't cheat by attending directly to prior turns via KV
  cache.
- If TTT is the memory mechanism it claims to be, it should keep the
  turn coherent using only the fast weight it accumulated during the
  previous turn's generation.

See [chat.md](chat.md) for the full memory-model discussion.

---

## Wiring contracts

These are invariants the code enforces (via regex, tests, and runtime
checks). Violating any of them silently breaks the mechanism; the tests
catch the ones amenable to unit testing.

### C1. LoRA never lands on TTT-layer `down_proj`

Enforced by the regex in
[`build_lora_target_regex`](../ttt_wiring.py):

```
LoRA targets: (q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj) anywhere,
              PLUS down_proj on NON-TTT layers only.
```

Silent failure mode: LoRA adapter lands on `down_proj`; the "fast weight
initial state" is now `W_down + LoRA_A @ LoRA_B`, which changes every
optimizer step in a direction the fast-weight mechanism doesn't
anticipate. Sanity_check probably still passes at init (LoRA_B init
is zero), but as soon as training starts, the fast-weight
interpretation degrades.

Tested in
[`test_wiring.py::test_lora_regex_never_targets_ttt_down_proj`](../tests/test_wiring.py).

### C2. `gate_proj` and `up_proj` called as modules

Not `F.linear(H, self.gate_proj.weight)`. The former includes LoRA in
`Z`; the latter bypasses it silently.

### C3. `down_proj` used functionally

The reverse: `w0 = self.down_proj.weight` and `z @ w0.T`, not
`self.down_proj(z)`. This bypasses any LoRA that might have leaked onto
`down_proj` (contract C1 says it shouldn't, but belt-and-braces).

### C4. `W_target` zero-initialized

Enforced in `InPlaceTTTMLP.__init__`. If this changes, `sanity_check`
fails. See
[mechanism.md#initialization-and-the-dead-basin](mechanism.md#initialization-and-the-dead-basin).

### C5. `v_source_norm.weight` frozen at 1.0

Enforced in `InPlaceTTTMLP.__init__`. If a change ever wants
this trainable, the optimizer can inflate it to counteract the clip.

### C6. Session state is per-stream

Enforced at runtime in `_scan_forward`: raises `RuntimeError` if
`carried_delta.shape[0]` (batch dim) doesn't match the incoming batch.

### C7. TTT parameter markers

Every fresh parameter's name must contain one of
`TTT_PARAM_MARKERS = ("target_conv", "w_target", "output_gate",
"v_source_norm")`. This is what tells `_classify_ttt_param` a parameter
belongs to the "new" group, and what tells `save_ttt_state_dict` which
parameters to serialize.

If you add a new fresh parameter and forget the marker, `save_ttt_state_dict`
won't save it (silent data loss on checkpoint) and `build_param_groups`
will raise `RuntimeError: Unclassified trainable parameter`. The
runtime check catches the second case; the first requires vigilance.

### C8. Fast weight in fp32

`carried_delta`, `_next_carried`, `state.delta`, and per-commit
accumulation all in fp32. bf16 drifts over many chunks.

### C9. Reset ladder

Documented in `inplace_ttt.py` module docstring. Adding a new mode of
state requires deciding which reset helper clears it. Silent state
leaking across sessions is one of the worst-class bugs in this
codebase.

---

## Filesystem layout at runtime

Modal maps two persistent volumes:

- **CKPT_MOUNT** (`/ckpt`) — the "ttt-checkpoints" volume.
- **HF_CACHE_MOUNT** (`/hf-cache`) — the "hf-hub-cache" volume, shared
  with `HF_HOME` env var.

Directory layout under `/ckpt`:

```
/ckpt/
├── <run_name>/                          # e.g. "ttt-v1.1"
│   ├── step_100/
│   │   ├── adapter/                     # LoRA weights (PEFT format)
│   │   │   ├── adapter_config.json
│   │   │   └── adapter_model.safetensors
│   │   └── ttt_params.pt                # TTT tensors (W_target, target_conv, W_down TTT-layer, output_gate)
│   ├── step_200/
│   ├── step_300/
│   ├── ...
│   └── sessions/                        # chat snapshot dir (created by /save)
│       ├── my-conversation.pt
│       └── ...
└── loss_mask/
    └── reference_wikitext103.pt         # unigram counts + metadata
```

Layout under `/hf-cache` follows the standard HuggingFace layout
(`HF_HOME`-based).

Resume-from paths:

- Same-run: `--resume-from step_600` → resolves to
  `/ckpt/<current_run>/step_600/`.
- Cross-run: `--resume-from other_run/step_600` → resolves to
  `/ckpt/other_run/step_600/`.

If either `adapter/` or `ttt_params.pt` is missing, resume fails with
`FileNotFoundError`. Optimizer momentum is **not** preserved across
resume (checkpoints don't save it).

---

## Related docs

- [mechanism.md](mechanism.md) — the math and code of what the layer computes
- [config.md](config.md) — every knob
- [training.md](training.md) — the outer loop in detail
- [inference.md](inference.md) — the three eval paths + generate + chat
- [checkpoints.md](checkpoints.md) — save/load format and resume behavior
- [failure-modes.md](failure-modes.md) — how each wiring rule breaks when violated
