# Inference and Evaluation

Reference for [`infer_modal.py`](../infer_modal.py) — the deployed
`TTTInference` class, local entrypoints, and eval semantics.

## The two inference paths

1. **Scan path** — `_scan_forward`. Whole-sequence forward, used for
   perplexity scoring and session eval. Cheaper for long sequences,
   supports session-mode carry with detached TBPTT boundary.

2. **Stream path** — `_stream_forward`. Chunk-by-chunk forward, used
   for autoregressive generation (`generate`, `chat_turn`). Fast
   weights persist across forward calls in `state.delta` + a pending
   partial chunk. Supports fast-weight snapshot export/import for
   cross-session persistence.

The two modes are wired via `set_ttt_stateful(model, bool)`.
`stateful=True` selects the stream path; `stateful=False` + calling
`set_session_mode(model, True)` selects the scan path with cross-item
carry.

## `TTTInference` class

Modal-hosted class with GPU attached. Warm-loaded on first call and
reused across method invocations to avoid per-call model reload.

**Class parameters** (set at instantiation):
- `ckpt: str` — checkpoint name, e.g. `"step_400"` or `""` for base.
- `load_ttt: bool` (default `True`) — when `False`, loads the trained
  LoRA adapter but skips `ttt_params.pt`. `W_target` stays at zero,
  TTT-layer `W_down` stays at pretrained. The LORA-ONLY ablation.

**Loading:**
```python
adapter, ttt_ckpt = _ckpt_paths(self.ckpt)
if not self.load_ttt:
    ttt_ckpt = None
model, tokenizer = build_model(
    adapter_path=adapter, ttt_ckpt_path=ttt_ckpt,
    trainable=False, attn_impl="sdpa"
)
model.eval()
model.config.use_cache = True
```

**Checkpoint paths** (`_ckpt_paths`):
- `""` (empty) → `(None, None)` — base model, no adapter, no TTT weights
- `"step_N"` → adapter + ttt_params under `<CKPT_MOUNT>/<run_name>/step_N/`
- `"other_run/step_N"` → explicit run name

## Methods

### `perplexity(text, evolve)`

Whole-sequence CE loss over one text. Returns `exp(loss)`. Uses the
scan path with `session_mode=False`, so this is a **stateless** ppl —
no cross-call carry.

Use for one-shot ppl on a single text without session semantics.

### `session_perplexity(texts, evolve, slice_papers, slice_seed, equal_n_slices)`

Per-item ppl with fast weights carrying across the session. Uses the
scan path with `session_mode=True`.

**Slicing modes (precedence order):**
1. `equal_n_slices > 0`: each paper cut into N equal-token slices
   (deterministic, no RNG). Used by `single_paper_eval`.
2. `slice_papers=True` (default): each paper randomly sliced per
   `TRAIN_CFG.slice_prob/min/max`. Same `slice_seed` for `evolve=True`
   and `evolve=False` → byte-identical inputs; only the TTT toggle
   differs.
3. `slice_papers=False`: one whole-paper item per input text.

**Returns:** list of per-item dicts:
```python
{
    "paper_idx": int,
    "slice_in_paper": int,      # 0-indexed within its paper
    "session_pos": int,          # 0-indexed within the session
    "start": int, "end": int,
    "n_tokens": int,
    "ppl": float,                # exp(item mean CE)
    "state_ratio_mean": float,   # mean across TTT layers, AFTER advance
}
```

Per-paper ppl is reconstructable via
`exp(sum(ln(ppl) * n_tok) / sum(n_tok))` over the paper's slices.

**Lifecycle:**
```
_set_mode(evolve, stateful=False)   # reset streaming state
set_session_mode(model, True)
reset_session_state(model)
for item in items:
    forward → advance_session_state → session_state_norms
set_session_mode(model, False)      # off before returning
reset_session_state(model)
```

### `generate(prompt, evolve, max_new_tokens, fast_weight_snapshot, temperature, top_p, seed, do_sample)`

Streaming autoregressive generation via
`self.model.generate(do_sample, temperature, top_p)`. Fast weights
evolve chunk by chunk over prompt + generated tokens when
`evolve=True`.

Optional `fast_weight_snapshot` (dict) restores a prior state.
Returns `{text, fast_weights}` where `fast_weights` is the CPU
snapshot of the final state (for later import).

`seed` (optional) — if set, `torch.manual_seed(seed)` is called before
generation so two calls with the same prompt + seed produce comparable
outputs. Used by `holdout_generate` to A/B carry-on vs carry-off.

### `fetch_holdout_texts(n_papers, seed)`

Sample n papers from the contamination-free holdout (the newest
`HOLDOUT_LAST_N` rows of the dataset). Deterministic given the seed.

### `chat_reset(evolve, from_snapshot_name)`

Reset fast weights and (optionally) load a snapshot from
`<CKPT_MOUNT>/<run_name>/sessions/<name>.pt`. See [chat.md](chat.md).

### `chat_turn(user_message, system_prompt, enable_thinking, evolve, max_new_tokens, temperature, top_p, top_k)`

One conversation turn. See [chat.md](chat.md) for the full invariant.

### `save_session(name)`

Persist the current streaming fast-weight state via
`export_fast_weights(model)` to `<CKPT_MOUNT>/<run_name>/sessions/<name>.pt`.

## Local entrypoints

### `holdout_eval` — three-way comparison

```
modal run infer_modal.py::holdout_eval --n-papers 5 --ckpt step_400
```

Prints three tables:
1. **BASE** — pure Qwen3 (no LoRA, no TTT). Pretraining floor.
2. **LORA-ONLY** — trained LoRA, TTT silent (`W_target=0`). LoRA contribution.
3. **FULL** — LoRA + trained TTT. The full model.

For (1) and (2), `state` should be exactly `0.00e+00` at every row and
`ppl carry == ppl fresh`. If not, wiring is broken (same signal as
`sanity_check`).

For (3), `state` grows across positions; `gap = fresh - carry` is the
TTT contribution.

**Omit `--ckpt`** to see only BASE (nothing to compare against).

Each of the three uses a separate `TTTInference` instance, so 3 model
loads happen. Failures in one section don't kill the others (try/except
around each).

### `single_paper_eval` — clean within-paper carry signal

```
modal run infer_modal.py::single_paper_eval --n-slices 8 --ckpt step_400
```

One paper cut into N equal-token slices, run as one session. Same
three-way structure as `holdout_eval`. Best for isolating the
within-paper carry from paper-to-paper ppl variance.

### `session_eval` — custom local papers

```
modal run infer_modal.py::session_eval --papers-dir ./papers --ckpt step_400
```

Feed every `.txt` in a directory (sorted). Same three-way structure.

### `compare_ppl` — single-text on/off

```
modal run infer_modal.py::compare_ppl --text-path paper.txt --ckpt step_400
```

Two whole-sequence ppl scores (evolve on vs off), no session, no
comparison to LORA-ONLY or BASE. Fast sanity check on one text.

### `generate_cli` — one-shot generation

```
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_400
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_400 --no-evolve
```

### `holdout_generate` — side-by-side carry comparison

```
modal run infer_modal.py::holdout_generate --n-papers 1 --ckpt step_400 --greedy
```

For each held-out paper, print prompt + carry-on continuation +
carry-off continuation. Same seed for both so the only difference is
the carry. Pass `--greedy` for deterministic decoding (only carry
differs, no sampling noise).

Flags: `--prefix-chars` (default 1200), `--max-new-tokens` (default 120),
`--temperature`, `--top-p`, `--seed`, `--greedy`.

## Reading the eval output

### The `state` column

`state_ratio_mean = mean_over_layers(||eta · carried_delta||_F / ||W_down||_F)`,
captured AFTER `advance_session_state` at each position.

- `0.00e+00` throughout → carry never staged. Bug (or `evolve=False`).
- Grows monotonically → carry is accumulating. Mechanism is engaged.
- Growth slope: linear = pure accumulation (`carried_decay=1.0`).
  Plateauing = EMA taking effect.
- Big magnitudes (state > 5) = clip is active. Applied signal is
  direction-only.

### The `gap` column

`gap = ppl_fresh - ppl_carry`. Positive means carry is helping.

At 0.6B best-performing, gap ranges +2 to +9 per slice, with
per-paper mean around +4.5 ppl improvement.

If gap flips negative on later positions of a session, the carry has
accumulated OOD magnitude that the model wasn't trained for. Either
retrain with matched session length or add `carried_decay < 1` to
bound stored magnitude.

### Per-paper summary

Token-weighted:
`log(ppl_per_paper) = sum(log(ppl_slice) * n_tok_slice) / sum(n_tok_slice)`.

## Streaming state diagnostic

For the chat/streaming path:

```python
from inplace_ttt import stateful_state_norms, stream_pending_progress

norms = stateful_state_norms(model)      # per-layer streaming state ratio
pending, chunk_size = stream_pending_progress(model)
```

Combined interpretation:
- `state_ratio_mean > 0`, growing → carry is engaged and accumulating.
- `state_ratio_mean == 0`, `pending < chunk_size` → mechanism is alive
  but hasn't accumulated a full chunk yet.
- `state_ratio_mean == 0`, `pending == 0` after many tokens → carry is
  actually dead (evolve was off, or fast weights were reset).

Exposed in the chat REPL after every turn.

## Related docs

- [architecture.md](architecture.md)
- [mechanism.md](mechanism.md) — scan vs stream path implementations
- [chat.md](chat.md) — streaming path + turn boundary rules
- [checkpoints.md](checkpoints.md) — ckpt path resolution
