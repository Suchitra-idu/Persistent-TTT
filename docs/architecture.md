# Architecture

Module boundaries, dataflow, and design rationale.

## Design goals

1. **Mechanism is Modal-free.** [`inplace_ttt.py`](../inplace_ttt.py)
   and [`ttt_wiring.py`](../ttt_wiring.py) have zero Modal imports, so
   the whole TTT layer is unit-testable on CPU with only PyTorch, and
   reusable outside Modal.

2. **Config is single-source.** Every constant lives in
   [`ttt_config.py`](../ttt_config.py). Duplicated defaults across
   files rot; a single dataclass forces the config to be
   canonical.

3. **Train and inference build the model identically.**
   [`model_setup.py`](../model_setup.py) contains one `build_model`
   function called from both `train_modal.py` and `infer_modal.py`, so
   there's exactly one way to assemble the base model + TTT patch +
   LoRA wrap + checkpoint load.

4. **Session vs stream, one method each.** The mechanism has two
   execution modes (parallel scan for training/eval, incremental
   stream for autoregressive generation). Each is its own method
   (`_scan_forward`, `_stream_forward`); no shared branchy code.
   See [mechanism.md](mechanism.md).

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

Everything sits flat at the project root. Modal ships modules into
containers via
`image.add_local_python_source("ttt_config", "inplace_ttt", ...)`,
which imports modules by name from the working directory. Moving files
into `src/` or a subpackage breaks Modal deployment.

## Import graph

```
train_modal.py  ──> model_setup.py ──> inplace_ttt.py ──> ttt_config.py
                │                    ─> ttt_wiring.py ──> ttt_config.py
                ├─> data_utils.py    ────────────────────> ttt_config.py
                ├─> train_utils.py   (loss mask, session schedule)
                └─> observability.py

infer_modal.py  ──> model_setup.py, data_utils.py, inplace_ttt.py, chat_utils.py

chat_client.py  ─── modal.Cls.from_name("inplace-ttt-infer", "TTTInference")

tests/          ──> inplace_ttt.py, ttt_wiring.py, train_utils.py, ttt_config.py
                    (no Modal, no GPU, no downloads)
```

**No cross-imports between the two Modal apps.** They share code only
via the pure Python modules. This keeps the `train_modal.py` container
image minimal (no chat_utils) and the `infer_modal.py` image minimal
(no train_utils in the hot path).

## Dataflow: training

```
HF Hub dataset
    │
    ▼
data_utils.open_dataset()  →  HF Datasets object
    │
    ▼
data_utils.split_holdout()  →  (train_ds, holdout_ds)
    │                              │
    │                              └── reserved for eval only
    ▼
tokenize + drop-short filter
    │
    ▼
train_utils.make_session_schedule() or make_single_paper_sessions()
    │
    ▼  (per session)
reset_session_state(model)
    │
    ▼  (per SessionItem in session)
apply_loss_mask(ids, common_mask)  →  labels with -100 on masked tokens
    │
    ▼
model(input_ids=ids, labels=labels).loss
    │
    ▼
loss.backward()
    │
    ▼
advance_session_state(model)  →  promotes _next_carried → carried_delta
```

Every N steps: optimizer.step(), zero grads, log to wandb, optionally
save checkpoint or run in-loop eval.

## Dataflow: inference (session_perplexity)

```
TTTInference.load()  →  build_model with adapter + ttt_ckpt
    │
    ▼
fetch_holdout_texts()  →  N held-out papers
    │
    ▼
session_perplexity(texts, evolve, slice_papers, slice_seed):
    │
    ├─  _set_mode(evolve=evolve, stateful=False)
    ├─  set_session_mode(model, True)
    ├─  reset_session_state(model)
    │
    ▼  (per item in items)
    │
    │   model(input_ids=ids, labels=ids).loss  →  ppl for this item
    │   advance_session_state(model)
    │   session_state_norms(model)  →  state_ratio_mean
    │
    ▼
per-item results list + per-paper token-weighted summary
```

## Dataflow: chat_turn (autoregressive inference)

```
chat_client.py  ──> engine.chat_turn.remote(user_message, ...)
                        │
                        ▼
                    set_ttt_stateful(model, True)
                    set_ttt_evolve(model, evolve)
                        │
                        ▼
                    apply Qwen3 chat template to user_message
                        │
                        ▼
                    prefill forward on new_ids (past_kv=None)
                        │
                        ▼  (per generated token, up to max_new_tokens)
                    sample from logits (top-p, top-k, temperature)
                        │
                        ▼
                    forward with past_kv, use_cache=True
                        │
                    (streaming _stream_forward path in TTT layers:
                     each chunk_size tokens accumulated
                     _commit_chunk() updates state.delta)
                        │
                        ▼
                    reset_v_left_context()  ← at turn end, not chunk buffer
                        │
                        ▼
                    return {text, thinking_text, state_ratio_mean,
                            pending_tokens, chunk_size, raw, ...}
```

Cross-turn memory: only `state.delta` + pending chunk buffer persists.
`past_key_values` and the tap's `_rolling` context are dropped at the
turn boundary. See [chat.md](chat.md) for the invariant details.

## Key design constraints

**Wiring rules (violating any of these breaks the mechanism):**

1. `gate_proj` / `up_proj` are called as modules so LoRA is included
   in Z. Never use `F.linear(h, self.gate_proj.weight)`.
2. `down_proj.weight` on TTT layers is used **functionally** as W0
   (the fast-weight initial state). LoRA never targets it — enforced
   by the regex in [`ttt_wiring.py`](../ttt_wiring.py) `build_lora_target_regex`.
3. `W_target` is zero-initialized. At step 0 every TTT layer is
   bit-equivalent to the original MLP. Verified by `sanity_check`.
4. The tap on `embed_tokens` (or on layer inputs when
   `v_source="hidden_state"`) provides X0 without changing model
   signatures.

**Fp32 boundary:**
- `carried_delta` is stored in fp32. Bf16 drifts over hundreds of
  session boundaries.
- Chunk-commit deltas in the streaming path also accumulate in fp32.
- Cast to activation dtype only at apply time.

**No batch > 1 mid-session:**
Fast weights are per-stream. The code rejects batch size changes
mid-session with an explicit `RuntimeError`. See
[`inplace_ttt.py`](../inplace_ttt.py) `_scan_forward`.

## Related docs

- [mechanism.md](mechanism.md) — what the layer computes
- [training.md](training.md) — outer loop
- [inference.md](inference.md) — eval paths
- [config.md](config.md) — every knob
