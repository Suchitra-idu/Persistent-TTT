# Inference and Evaluation

The three eval paths (holdout / single-paper / custom), the two
generation entrypoints, the streaming chat memory model, and every
knob that changes what gets measured.
Reference: [`infer_modal.py`](../infer_modal.py).

## Contents

1. [The three inference paths, in one picture](#the-three-inference-paths-in-one-picture)
2. [The `TTTInference` class](#the-ttinference-class)
3. [`perplexity` — one-shot, no session](#perplexity--one-shot-no-session)
4. [`session_perplexity` — carry across items](#session_perplexity--carry-across-items)
5. [`generate` — autoregressive with fast-weight persistence](#generate--autoregressive-with-fast-weight-persistence)
6. [Chat: `chat_reset` and `chat_turn`](#chat-chat_reset-and-chat_turn)
7. [Snapshots: `save_session` and `import_fast_weights`](#snapshots-save_session-and-import_fast_weights)
8. [Local entrypoints](#local-entrypoints)
9. [Three-way eval: BASE / LORA-ONLY / FULL](#three-way-eval-base--lora-only--full)
10. [Reading the eval output](#reading-the-eval-output)
11. [Streaming state diagnostics](#streaming-state-diagnostics)
12. [Related docs](#related-docs)

---

## The three inference paths, in one picture

The mechanism has two execution paths (see
[mechanism.md](mechanism.md)); inference uses them in three ways:

```
┌─────────────────────────┬────────────────┬─────────────────────────────┐
│ Path                    │ Uses           │ Purpose                     │
├─────────────────────────┼────────────────┼─────────────────────────────┤
│ Scan, no session        │ _scan_forward  │ perplexity() one-shot ppl   │
│ Scan, session-mode      │ _scan_forward  │ session_perplexity() eval   │
│ Stream                  │ _stream_forward│ generate() and chat_turn()  │
└─────────────────────────┴────────────────┴─────────────────────────────┘
```

**Scan-no-session:** each forward starts with `S = 0`. No cross-call
memory. Used for isolated ppl on a single text.

**Scan-with-session:** each forward starts with `S = carried_delta`
from the previous call in the session. Between calls,
`advance_session_state` promotes `_next_carried` into `carried_delta`.
Used for `session_perplexity` — the eval workhorse.

**Stream:** each call feeds tokens through `_stream_forward`, which
maintains a `state.delta` + partial-chunk buffer across calls within
the same TTT module. Used for autoregressive `.generate()`, and
especially for chat where cross-turn memory is the whole point.

These three are wired via three flags on each TTT module — `session_mode`,
`stateful`, `ttt_evolve` — set inline via `iter_ttt_modules(model)`.
`session_mode=True + stateful=False` → scan-with-session.
`session_mode=False + stateful=True` → stream. `ttt_evolve=False` in
either mode freezes the fast-weight state (applies current, doesn't
update it).

---

## The `TTTInference` class

Modal-hosted class with GPU attached. Warm-loaded on first call and
reused across method invocations. See
[architecture.md#inference-dataflows](architecture.md#inference-dataflows).

### Class parameters

Set at instantiation:

- `ckpt: str` (default `""`) — checkpoint name.
  - `""` → base Qwen3, no adapter, no TTT weights.
  - `"step_400"` → `<CKPT_MOUNT>/<run_name>/step_400/`.
  - `"other_run/step_400"` → `<CKPT_MOUNT>/other_run/step_400/`.
- `load_ttt: bool` (default `True`) — when `False`, loads the LoRA
  adapter but skips `ttt_params.pt`. `W_target` stays at zero,
  `output_gate` at init, TTT-layer `down_proj` at pretrained. This is
  the LORA-ONLY ablation used by `_three_way_eval`.

### Loading (`@modal.enter`)

```python
adapter, ttt_ckpt = _ckpt_paths(self.ckpt)
if not self.load_ttt:
    ttt_ckpt = None
self.model, self.tokenizer = build_model(
    adapter_path=adapter, ttt_ckpt_path=ttt_ckpt,
    trainable=False, attn_impl="sdpa"
)
self.model.eval()
self.model.config.use_cache = True
```

`attn_impl="sdpa"` (scaled-dot-product attention) instead of flash
attention. Simpler dependency footprint at inference — no flash-attn
wheel needed. Slightly slower on very long sequences; acceptable
tradeoff for a much smaller Modal image.

`use_cache=True` enables the standard HuggingFace KV cache, used by
`.generate()`. In training we set it False (grad checkpointing needs
recompute-friendly forward).

### `_set_mode(evolve, stateful, fresh=True)`

Internal helper that resets fast weights and sets the mode flags:

```python
if fresh:
    reset_fast_weights(self.model)              # full clear
self.model._ttt_tap.stateful = stateful
for m in iter_ttt_modules(self.model):
    m.stateful = stateful
    m.ttt_evolve = evolve
```

`fresh=True` (the default) does a full reset. Chat's `chat_turn`
explicitly does **not** call `_set_mode` because it needs `state.delta`
to survive across turns.

---

## `perplexity` — one-shot, no session

```python
@modal.method()
def perplexity(self, text: str, evolve: bool = True) -> float:
    self._set_mode(evolve=evolve, stateful=False)
    ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=TRAIN_CFG.max_seq_len).input_ids.cuda()
    with torch.no_grad():
        loss = self.model(input_ids=ids, labels=ids).loss
    return math.exp(loss.item())
```

Whole-sequence CE loss on one text, returned as `exp(loss)`.

- **Not** session-mode. Each call starts from `S = 0`.
- The reset makes this method idempotent.
- `evolve=False`: `_scan_forward` returns `base_out` early (see
  [mechanism.md#the-scan-path](mechanism.md#the-scan-path)); the TTT
  path never fires. That's the ablation form.
- `evolve=True`: TTT fires within the forward, but there's no carry to
  or from other calls.

Use this for one-shot ppl on a single text without any session
semantics.

---

## `session_perplexity` — carry across items

The workhorse of the eval flows. Per-item ppl with fast weights
carrying across the session.

```python
def session_perplexity(self, texts, evolve=True, slice_papers=True,
                       slice_seed=0, equal_n_slices=0,
                       reset_between_items=False) -> list:
    self._set_mode(evolve=evolve, stateful=False)
    for m in iter_ttt_modules(self.model):
        m.session_mode = True
    reset_session_state(self.model)

    # ... build items ...

    for pos, item in enumerate(items):
        ids = torch.tensor([paper_token_ids[item.doc_idx][s:e]], device="cuda")
        with torch.no_grad():
            loss = self.model(input_ids=ids, labels=ids).loss
        advance_session_state(self.model)
        # state_ratio read here reflects THIS item's accumulated fast weight
        if reset_between_items:
            reset_session_state(self.model)  # carry-off mode
        # record ppl, state_ratio, session_pos, ...

    # finally:
    for m in iter_ttt_modules(self.model):
        m.session_mode = False
    reset_session_state(self.model)

    return per_item_results
```

### Three eval modes

Combining `evolve` and `reset_between_items` gives three well-defined
regimes; the eval entrypoints run all three back-to-back:

| Mode        | `evolve` | `reset_between_items` | What adapts                                    |
|-------------|:--------:|:---------------------:|------------------------------------------------|
| `carry`     | True     | False                 | chunk-scan within a forward + carry between items |
| `carry-off` | True     | True                  | chunk-scan within a forward only                  |
| `fresh`     | False    | (irrelevant)          | nothing (fast weight = 0 throughout)              |

Two natural gaps:
- `Δwithin = fresh - carry-off` — chunk-scan benefit within one forward
- `Δbetween = carry-off - carry` — cross-item carry benefit on top
- Total TTT benefit = `fresh - carry = Δwithin + Δbetween`

### Everlasting-carry eval

`session_perplexity` gains three parameters for evaluating a checkpoint
that was trained with `--mode everlasting` (see
[training.md#everlasting-carry-mode](training.md#everlasting-carry-mode)):

| param | default | what it does |
|---|---|---|
| `use_everlasting_carry` | `False` | Before each doc's first item install `per_source_carries[doc.source]` as the fast-weight seed. After each `reset_between_items` (in `carry-off` mode) reinstall the seed so within-item chunk-scan still starts from the trained state. Cold sources (label not in the loaded dict) start from zero — matches training. |
| `force_source` | `""` | If non-empty, install THAT source's carrier for every doc regardless of the doc's actual label. **Swap test:** a domain-specific carrier should degrade when mismatched. If ppls are indistinguishable across choices of `force_source`, the carrier is generic conditioning, not per-domain adaptation. |
| `sources` | `None` | Per-doc source labels (aligned with `texts`). `holdout_eval` fills this in automatically from the dataset's `source` column. |

When `use_everlasting_carry=True` **and** `include_cold_carry=True`,
`_three_way_eval` runs **two extra passes** (cold-carry and
cold-carry-off, both with the seed NOT installed). Together with the
three seeded modes this gives a 5-column table with **regime-clean
deltas** that additively decompose the total TTT benefit.

**The 5 modes:**

| column | starts from | reset between items |
|---|---|---|
| `carry` | trained seed | no |
| `cold-c` (cold-carry) | zero | no |
| `carry-off` | trained seed, reinstalled each item | yes |
| `cold-co` (cold-carry-off) | zero, reset each item | yes |
| `fresh` | (evolve=False, TTT off entirely) | n/a |

**Clean deltas — each measures ONE mechanism, no cross-contamination:**

| delta | formula | measures |
|---|---|---|
| `Δwithin` | `fresh − cold-carry-off` | pure within-item chunk-scan (from zero) |
| `Δbetween` | `cold-carry-off − cold-carry` | pure cross-item accumulation (from zero) |
| `Δseed` | `cold-carry − carry` | trained per-source seed's isolated contribution |
| `Δtotal` | `fresh − carry` | = `Δwithin + Δbetween + Δseed` |

**Why this decomposition matters — the old Δs conflated regimes.**
Before adding cold-carry-off, `Δwithin = fresh − carry-off` in the
seeded regime silently contained the seed's baseline lift (because
`carry-off` had the seed installed on every item). It looked like
"within-item chunk-scan is doing great" when what you were actually
measuring was "the seed is helping". Now `Δwithin` always compares
against **cold-carry-off** — a genuinely seed-free within-item
measurement — regardless of whether the seed is installed. `Δbetween`
always compares two cold states, so it always means "pure cross-item
accumulation from zero". And `Δseed` cleanly isolates the seed's
value. The three add up to `Δtotal`.

**Non-everlasting eval (no `--use-everlasting-carry`)** keeps the old
3-column table with `Δwithin = fresh − carry-off` and `Δbetween =
carry-off − carry` — both computed on zero-seed passes anyway
(since no seed is installed), so the formula coincides with the clean
version there. No behavior change on that path.

### Slicing modes (precedence order)

1. **`equal_n_slices > 0`**: each paper cut into `n` equal-token
   slices (deterministic, no RNG). Used by `single_paper_eval`.
2. **`slice_papers=True`** (default): each paper randomly sliced per
   `TRAIN_CFG.slice_prob`, `slice_min`, `slice_max`. Same `slice_seed`
   passed to both `evolve=True` and `evolve=False` calls ensures
   byte-identical inputs — only the TTT toggle differs.
3. **`slice_papers=False`**: one whole-paper item per input text.

### Returned list

```python
[
    {
        "paper_idx": int,
        "slice_in_paper": int,      # 0-indexed within its paper
        "session_pos": int,          # 0-indexed within the session
        "start": int, "end": int,    # token range within the paper
        "n_tokens": int,
        "ppl": float,                # exp(item mean CE)
        "state_ratio_mean": float,   # AFTER advance_session_state
    },
    ...
]
```

`ppl` is `exp(loss)`, the token-uniform mean of CE within the item.
Per-paper token-weighted ppl is
`exp(sum(log(ppl_slice) * n_tok_slice) / sum(n_tok_slice))` across the
paper's slices — computed by the print helpers.

### Determinism guarantee

The same `slice_seed` on the same texts and the same `slice_papers`
setting produces identical items (`SessionItem` tuples). This is what
makes the carry-vs-fresh comparison meaningful: the two runs see the
same tokens in the same order.

`equal_n_slices > 0` is deterministic without an RNG — pure integer
math from `equal_token_slices`.

---

## `generate` — autoregressive with fast-weight persistence

```python
def generate(self, prompt, evolve=True, max_new_tokens=512,
             fast_weight_snapshot=None,
             temperature=0.7, top_p=0.9, seed=None,
             do_sample=True) -> dict:
    self._set_mode(evolve=evolve, stateful=True)
    if fast_weight_snapshot:
        import_fast_weights(self.model, fast_weight_snapshot)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    snapshot = export_fast_weights(self.model)
    return {"text": text, "fast_weights": snapshot}
```

Uses the streaming path — fast weights evolve chunk-by-chunk over
prompt + generated tokens.

- **`fast_weight_snapshot`** (optional): restore a prior `state.delta`
  from a dict returned by a previous `generate()` call. This enables
  cross-call persistence *without* using session mode (session mode is
  scan-based; this is stream-based).
- **`seed`** (optional): `torch.manual_seed(seed)` before generation
  so two runs with the same prompt + seed produce comparable outputs.
  Used by `holdout_generate` for A/B carry-on vs carry-off with only
  the carry differing.
- **`do_sample=False`** (via `--greedy`): argmax decoding. Only sane
  choice for a strict A/B: samples produce different tokens even with
  identical logits.

Returned `snapshot` is a `dict[layer_idx, Tensor]` snapshot of every
TTT layer's final `state.delta` — usable as input to a subsequent
`generate` call or to `chat_reset(from_snapshot_name=...)` after saving.

---

## Chat: `chat_reset` and `chat_turn`

The chat path is the only workflow where the streaming
`_stream_forward` gets used for cross-turn memory. See
[chat.md](chat.md) for the full memory model.

### `chat_reset(evolve=True, from_snapshot_name="")`

```python
def chat_reset(self, evolve=True, from_snapshot_name=""):
    self._set_mode(evolve=evolve, stateful=True, fresh=True)
    seeded = False
    if from_snapshot_name:
        path = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name,
                            "sessions", f"{from_snapshot_name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(...)
        snapshot = torch.load(path, map_location="cuda")
        import_fast_weights(self.model, snapshot)
        seeded = True
    return {"ready": True, "evolve": evolve, "seeded_from_snapshot": seeded}
```

- Full reset (`fresh=True`) clears `state.delta`, pending, conv left
  context on every TTT module + the tap.
- Optionally seeds `state.delta` from a saved snapshot — so a chat
  session can start "as if it had read paper X first."

### `chat_turn(user_message, ..., max_new_tokens=512, ...)`

One conversation turn. Not idempotent with respect to fast weights —
each turn updates `state.delta` (when `evolve=True`).

Key invariant (see [chat.md](chat.md)):

- The turn's prefill sees only the current user message wrapped in the
  Qwen3 chat template — **no** past turns are re-fed as input tokens.
- Attention KV cache is used within the turn only; discarded at turn
  end.
- The conv left-context buffer is reset at turn end via
  `reset_v_left_context`.
- **TTT `state.delta` + pending PERSIST across turns** — the sole
  cross-turn memory carrier.

If TTT is the memory mechanism it claims to be, it must keep the
conversation coherent through only the fast weight it accumulated
during the previous turn's generation. Re-feeding prior turns would
break the test.

Returned dict:

```python
{
    "text": str,               # stripped answer (post-<think>...</think>)
    "thinking_text": str,      # extracted thinking trace if present
    "raw": str,                # raw text incl. specials
    "token_ids": list[int],    # generated token ids
    "stop_reason": str,        # "stop_token" or "max_tokens"
    "stop_token_id": int|None,
    "state_ratio_mean": float, # mean streaming state ratio at end
    "pending_tokens": int,     # partial chunk buffer size
    "chunk_size": int,         # cfg.chunk_size for reference
}
```

The last three fields are the diagnostics for whether TTT is actually
engaging — see
[Streaming state diagnostics](#streaming-state-diagnostics) below.

---

## Snapshots: `save_session` and `import_fast_weights`

### `save_session(name)`

Persist the current streaming fast-weight state:

```python
def save_session(self, name):
    path = os.path.join(CKPT_MOUNT, TRAIN_CFG.run_name,
                        "sessions", f"{name}.pt")
    torch.save(export_fast_weights(self.model), path)
    ckpt_vol.commit()
    return path
```

`export_fast_weights` returns a `dict[layer_idx, Tensor(cpu)]` for
every layer with non-None `state.delta`. Layers with `state.delta =
None` are skipped, so a partial snapshot is well-defined.

Volume is committed so the write is visible to future container starts
(including reloads after cold-start).

### `import_fast_weights` (used internally)

Called by `generate(fast_weight_snapshot=...)` and
`chat_reset(from_snapshot_name=...)`. Restores `state.delta` on each
TTT layer whose index is in the snapshot dict. Layers not in the dict
are unchanged.

Snapshot files are `torch.save`d dicts — not backward-compat across
`layer_indices` changes. Rebuild snapshots after a
`TTT_LAYER_STRIDE` change.

---

## Local entrypoints

Local entrypoints run on your laptop; they call `TTTInference.method.remote(...)`
which runs on Modal.

### `holdout_eval` — three-way holdout comparison

> **⚠ Correct command for a SlimPajama everlasting-carry checkpoint.**
> The eval that shows the per-source everlasting-carry signal requires
> BOTH `--use-everlasting-carry` AND `--include-cold-carry`. Without
> them, the "FULL" section runs the trained model but does NOT install
> the trained per-source seeds — so `carry` starts from zero and the
> whole point of the experiment is missed. Silent skip: just passing
> `--include-cold-carry` alone is a no-op (a warning now prints).
>
> ```bash
> TTT_DATASET=slimpajama-6b modal run infer_modal.py::holdout_eval \
>     --ckpt step_200 \
>     --n-papers-per-source 2 --min-tokens-est 2048 \
>     --equal-n-slices 4 \
>     --use-everlasting-carry --include-cold-carry
> ```
>
> **Without** `--use-everlasting-carry`, `carry` behaves exactly like
> the in-loop training-side eval (zero-seed chunk-scan accumulation)
> and Δbetween measures "zero→accumulate vs zero→reset-each" — that
> gap can be large but is unrelated to the everlasting mechanism.

```
# Single-source dataset (arxiv): plain N-paper draw
modal run infer_modal.py::holdout_eval --n-papers 5 --ckpt step_400

# Multi-source dataset (SlimPajama): stratified per-source draw
TTT_DATASET=slimpajama-6b modal run infer_modal.py::holdout_eval \
    --ckpt step_400 --n-papers-per-source 2 --min-tokens-est 2048 \
    --equal-n-slices 4
```

Runs `_three_way_eval` on holdout papers. Each of BASE / LORA-ONLY /
FULL runs the same texts three times: `carry` (TTT + cross-item carry),
`carry-off` (TTT within item, reset between items), `fresh` (TTT
silent). The printed table has three ppl columns and two Δ columns
(`Δwithin`, `Δbetween`). When the dataset has a source column, a
`per-source (token-weighted)` summary is printed underneath. See
[Three-way eval](#three-way-eval-base--lora-only--full) and
[Per-source eval table](#per-source-eval-table) below.

**Args:**
- `--n-papers N` (default 5) — number of holdout papers when the
  dataset has no source column (arxiv). Ignored when
  `--n-papers-per-source > 0` and the dataset has a source column.
- `--n-papers-per-source N` (default 0) — with a multi-source
  dataset, pick exactly N papers per source. Total picked =
  `N × n_sources`. Every domain contributes; the per-source table
  becomes statistically usable.
- `--min-tokens-est N` (default 0) — skip docs shorter than this
  (uses `tokens_est_column` if present, else 4-chars-per-token
  proxy). Use `2048` on SlimPajama so eval doesn't pick tiny
  StackExchange posts that can't be sliced meaningfully.
- `--equal-n-slices N` (default 0) — force each paper into N equal
  slices (session_pos becomes `1.1, 1.2, ..., 1.N, 2.1, ...`).
  Without it, each paper is one item (labels `1.1, 2.1, 3.1, ...`)
  since `TRAIN_CFG.slice_prob = 0` by default. Set 4-8 to see
  within-paper adaptation ramp up slice by slice.
- `--seed S` (default 0) — deterministic paper selection.
- `--ckpt X` (default `""`) — checkpoint. `""` prints only BASE.
- `--slice-papers` / `--no-slice-papers` — whether to slice papers
  within sessions per `TRAIN_CFG.slice_prob`. Overridden by
  `--equal-n-slices > 0`.
- `--use-everlasting-carry` (default False) — only affects the FULL
  block. Before each doc's first item install `per_source_carries[src]`
  as the fast-weight seed; after each `reset_between_items` reinstall
  the seed. No-op if the checkpoint has no `per_source_carries.pt`.
  See [Everlasting-carry eval](#everlasting-carry-eval).
- `--force-source LABEL` (default `""`) — with `--use-everlasting-carry`
  on, install THAT source's carrier for every doc regardless of the
  doc's actual source. The domain-specificity swap test: if the trained
  carrier is domain-specific, mismatched installs should be worse than
  matched. If ppls are the same regardless of source, the carrier is
  generic conditioning, not per-domain adaptation.
- `--include-cold-carry` (default False) — adds **two** extra eval
  passes on FULL (`cold-carry` and `cold-carry-off`) when combined
  with `--use-everlasting-carry`, giving a 5-column table with
  regime-clean deltas: `Δwithin = fresh − cold-carry-off` (pure
  within-item, seed-free), `Δbetween = cold-carry-off − cold-carry`
  (pure cross-item, seed-free), `Δseed = cold-carry − carry` (seed
  contribution). Additive: `Δtotal = Δwithin + Δbetween + Δseed`.
  Prints a warning and skips the extra passes when
  `--use-everlasting-carry` is off (`carry` already IS cold-carry
  when no seed is installed).

### `single_paper_eval` — clean within-paper carry

```
modal run infer_modal.py::single_paper_eval --n-slices 8 --ckpt step_400
```

One paper, cut into `n_slices` equal-token pieces, run as one session.
Three-way structure. Best for isolating the within-paper carry from
paper-to-paper ppl variance (the biggest source of noise in
`holdout_eval`).

### `session_eval` — custom local papers

```
modal run infer_modal.py::session_eval --papers-dir ./papers --ckpt step_400
```

Feed every `.txt` file in `papers-dir` (sorted alphabetically) as one
session. Three-way structure. Useful for evaluating on non-holdout
inputs.

### `compare_ppl` — single-text on/off

```
modal run infer_modal.py::compare_ppl --text-path paper.txt --ckpt step_400
```

Two whole-sequence ppl scores (evolve=True vs evolve=False), **no**
session mode, **no** comparison to LORA-ONLY or BASE. Fast sanity
check on one text.

### `generate_cli` — one-shot generation

```
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_400
modal run infer_modal.py::generate_cli --prompt "..." --ckpt step_400 --no-evolve
```

### `holdout_generate` — side-by-side carry comparison

```
modal run infer_modal.py::holdout_generate --n-papers 1 --ckpt step_400 --greedy
```

For each held-out paper:
1. Take the first `--prefix-chars` (default 1200) as prompt.
2. Print the tail of the prompt (for context).
3. Generate `--max-new-tokens` (default 120) with `evolve=True`.
4. Generate again with `evolve=False`, same seed, same params.
5. Print both, side-by-side.

Pass `--greedy` for `do_sample=False`. That way the only difference
between the two continuations is the carry state — same logits →
different output only when carry actually alters the logits.

Args: `--prefix-chars`, `--max-new-tokens`, `--temperature`,
`--top-p`, `--seed`, `--greedy`.

---

## Three-way eval: BASE / LORA-ONLY / FULL

`_three_way_eval(base_engine, ckpt, texts, session_kwargs)` runs the
same texts through three configurations and prints three tables:

### 1. BASE

`TTTInference(ckpt="")` — pure Qwen3, no adapter, no TTT weights. The
pretraining floor.

- Every row: `carry == carry-off == fresh`, `state == 0.00e+00`.
- This is the wiring check: BASE with any TTT toggle should behave
  identically because `W_target = 0` means the TTT path contributes
  nothing. If they differ, the TTT patch is broken.

### 2. LORA-ONLY

`TTTInference(ckpt=ckpt, load_ttt=False)` — trained LoRA loaded, TTT
tensors (including the TTT-layer `down_proj`) deliberately NOT loaded.
`W_target = 0` and `output_gate` at init.

- Isolates the LoRA contribution.
- Same wiring property: `carry == carry-off == fresh`, `state == 0`.
- At 0.6B on arxiv ML: BASE ppl ≈ 37 → LoRA-ONLY ppl ≈ 31.
- **Note:** LORA-ONLY `fresh` ≠ FULL `fresh`. FULL loads the trained
  TTT-layer `down_proj` (a slow weight in the `wdown` param group); the
  mechanism itself is silent when the fast weight = 0 but the trained
  `down_proj` still shifts ppl. This delta is "what SGD-updated
  `down_proj` on TTT layers buys you, mechanism aside."

### 3. FULL

`TTTInference(ckpt=ckpt, load_ttt=True)` — LoRA + TTT tensors both
loaded.

- `state_ratio` grows across positions in the `carry` run; is reset to
  zero between items in `carry-off`; stays at 0 in `fresh`.
- `Δwithin` = `fresh - carry-off` = within-item chunk-scan benefit
- `Δbetween` = `carry-off - carry` = cross-item persistence benefit
- Total = `fresh - carry` ≈ +3 to +7 ppl at 0.6B best.

### Why 3 model loads

Each of the three uses a separate `TTTInference` Modal instance. Three
cold starts. This is deliberate: we want the three configurations to
be exactly what they say they are, not "same instance with flags
toggled." Modal caches images across the three, so the second and
third load are much faster than the first.

### Failure isolation

Each of the three sub-runs is wrapped in try/except; a failure in one
prints an error and skips to the next. This is important because
LORA-ONLY sometimes fails on very short checkpoints (adapter
malformed, TTT ckpt required for shapes to match, etc.), and we want
BASE and FULL to still print.

---

## Reading the eval output

The per-item table columns:

```
pos    p.s    n_tok    carry   carry-off   fresh    Δwithin   Δbetween    state
```

- `pos`: session-position index (0-indexed).
- `p.s`: paper.slice, e.g. `2.3` = paper 2, slice 3. Note: with
  `--equal-n-slices 0` (default) each paper is one item and you see
  `1.1, 2.1, 3.1, ...`. Pass `--equal-n-slices 4` for `1.1..1.4, 2.1..2.4, ...`.
- `n_tok`: token count of this item.
- `carry`: ppl with `evolve=True`, fast weight carries across items.
- `carry-off`: ppl with `evolve=True`, fast weight reset between items
  (chunk-scan still fires inside each forward).
- `fresh`: ppl with `evolve=False`, fast weight = 0 throughout.
- `Δwithin`: `fresh - carry-off`. Within-item chunk-scan benefit.
- `Δbetween`: `carry-off - carry`. Cross-item persistence benefit.
- `state`: mean `||eta · carried_delta||_F / ||W_down||_F` across TTT
  layers, captured **after** the item's forward. In `carry-off` mode
  the state is read before the reset, so it still reflects what got
  accumulated during that item's chunks.

### The `state` column, interpreted

`session_perplexity` resets the fast weight at every document boundary
by default (`reset_between_docs=True`), matching training-side
semantics. So state should **grow within a paper's slices, then drop
back to a small value** at each `p.s = X.1` row.

- `0.00e+00` throughout → carry never staged. In BASE and LORA-ONLY
  this is expected. In FULL this indicates the mechanism is broken.
- Grows within paper, resets at doc boundary → mechanism engaged and
  working as intended.
- Grows monotonically **across** papers (state at paper 2 slice 1 ≫
  state at paper 1 slice 1) → doc-boundary reset was disabled; you're
  measuring cross-paper carry, which is usually not what you want.
- **Growth slope within a paper:** linear = pure accumulation
  (`carried_decay=1.0`). Plateauing/asymptoting = EMA taking effect
  (`carried_decay < 1.0`).
- **Big magnitudes (state > 5) even within one paper's slices:** clip is
  active. Applied signal is direction-only. See
  [mechanism.md#the-frobenius-clip](mechanism.md#the-frobenius-clip)
  and
  [failure-modes.md#state-saturation](failure-modes.md#state-saturation).

### The Δ columns, interpreted

Both Δs are positive when TTT helps. The decomposition tells you *how*:

- **Δwithin dominates, Δbetween ≈ 0**: chunk-scan wakes up but nothing
  useful survives across item boundaries. Common early in training and
  in short-item eval. Not necessarily broken — chunk-scan alone gives
  meaningful improvement on long documents.
- **Both positive, similar magnitude**: healthy full-TTT operation.
  Cross-item carry adds on top of within-item adaptation.
- **Δbetween ≥ Δwithin**: the carry state is doing heavy lifting —
  usually seen on tightly-related sequential items (same paper sliced,
  multi-turn conversation).
- **Δwithin < 0** (carry-off *worse* than fresh): chunk-scan is
  actively harmful within an item. Usually means the trained state was
  poorly regularized; suspect very high `state_ratio` and clip
  saturation.
- **Δbetween < 0** (carry *worse* than carry-off): accumulated carry
  drifts OOD by the end of the session. Retrain with matched session
  length, or add `carried_decay < 1`.

If both Δs are near zero throughout while `state_ratio > 0`, mechanism
is active but its direction isn't useful. Suspect gate stuck near zero
(check `health/gate_mean_L<i>`) or `W_target` insufficiently trained.

### Per-paper summary block

Below the per-item table, a token-weighted per-paper table:

```
paper     n_tok    carry   carry-off   fresh    Δwithin   Δbetween
1          8532     22.4      24.1      25.7     +1.6       +1.7
2         14001     18.9      20.0      21.1     +1.1       +1.1
...
```

Aggregation: `log(ppl_per_paper) = sum(log(ppl_slice) · n_tok_slice) /
sum(n_tok_slice)`. Geometric mean weighted by slice token count — the
correct aggregation for perplexity.

### Per-source eval table

When the active dataset carries a source label (SlimPajama's
`meta.redpajama_set_name`, etc.), `_print_per_source_summary`
appends a per-source token-weighted table under the per-paper table:

```
per-source (token-weighted):
source                   n_papers    n_tok    carry   carry-off   fresh    Δwithin   Δbetween
RedPajamaArXiv                  2    42019    19.844   20.577    21.310    +0.733    +0.733
RedPajamaBook                   2    31842    27.402   27.706    28.011    +0.305    +0.305
RedPajamaC4                     3    27441    42.318   42.434    42.550    +0.116    +0.116
RedPajamaGithub                 2    18022    12.911   14.377    15.844    +1.467    +1.467
RedPajamaStackExchange          2    21001    31.415   31.698    31.982    +0.284    +0.284
RedPajamaWikipedia              2    15804    26.113   26.336    26.559    +0.223    +0.223
```

Read the Δ columns: `Δtotal = Δwithin + Δbetween` is where "which
domain benefits most from TTT" lives. Splitting into `Δwithin` and
`Δbetween` tells you whether each domain benefits from within-item
adaptation, cross-item persistence, or both. Interpret with the number
of papers and total tokens per source — a big gap on a tiny sample
isn't statistically solid.

`fetch_holdout_texts` samples the holdout stratified by source
(round-robin one per source until `n_papers` is hit) so no domain is
starved of representation in the eval. When your dataset has no
source column, this table is silently omitted.

### `holdout_generate` output

Not a table — free-form generations. What to look for:

- **Content coherence:** does the CARRY ON continuation stay on-topic
  vs the CARRY OFF continuation drifting? At 0.6B, this signal is
  subtle; per-token perplexity differences don't always translate to
  visibly different text.
- **Consistency with prompt:** does either continuation cite specific
  terms/methods the prompt mentioned?
- **Sanity check:** with `--greedy` and no ckpt, both continuations
  should be **identical** — same base model, same seed, same
  everything. If they differ, TTT wiring is wrong.

---

## Streaming state diagnostics

For the chat / streaming path, use:

```python
from inplace_ttt import state_norms, stream_pending_progress, mean_state_ratio

norms = state_norms(model, source="stream")   # per-layer streaming state ratio
mean = mean_state_ratio(norms)
pending, chunk_size = stream_pending_progress(model)
```

Combined interpretation:

| `mean_state_ratio` | `pending / chunk_size` | Interpretation |
|---|---|---|
| `> 0`, growing | any | Carry is engaged and accumulating. |
| `== 0` | `< 1.0` | Alive; hasn't accumulated a full chunk yet. |
| `== 0` | `== 0.0` after many tokens | Carry dead. Either `evolve=False` or fast weights were reset. |
| `> 0`, static | any | Evolve is on but `_commit_chunk` isn't firing — check if chunk boundaries are being crossed. |

Exposed in the chat REPL after every turn:

```
bot> …
  [state_ratio=1.234e+00  pending=17/50]
```

`chat_turn` returns `state_ratio_mean`, `pending_tokens`, and
`chunk_size` in its result dict — these are what the REPL prints.

### The `evolve` flag at chat time

- `chat_reset(evolve=True)`: normal operation. Fast weights update
  during each turn's generation, accumulate across turns.
- `chat_reset(evolve=False)`: pure baseline. The fast weights don't
  update. Turns are still generated (via `_stream_forward` with
  `ttt_evolve=False`), but each turn's fast weight is whatever it was
  when reset was called — no updating.

`evolve=False` is useful for showing what "no carry" looks like:
compare responses to a factual question after reading N paragraphs
about the topic. `evolve=True` should improve the response;
`evolve=False` should not.

---

## Related docs

- [architecture.md](architecture.md) — how these entrypoints fit into the whole
- [mechanism.md](mechanism.md) — scan vs stream path implementations
- [chat.md](chat.md) — streaming path + turn boundary rules
- [checkpoints.md](checkpoints.md) — ckpt path resolution
- [observability.md](observability.md) — what to log during eval
- [failure-modes.md](failure-modes.md) — what to do when carry gap is flat or negative
