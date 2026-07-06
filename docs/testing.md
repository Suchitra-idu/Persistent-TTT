# Testing

The test suite structure, what each file covers, and the invariants
being tested.

## Running the suite

```
python -m pytest tests/ -q
```

Requirements: torch, numpy, pytest. No Modal, no GPU, no downloads.
Runs in ~3 seconds on any machine.

**When to run:**
- After any change to `inplace_ttt.py`, `ttt_wiring.py`,
  `train_utils.py`, or `observability.py`.
- Before every `modal run train_modal.py::train`.
- Before opening any PR.

## Suite layout

```
tests/
├── conftest.py            shared tiny-module fixtures
├── test_scan_math.py      scan vs sequential reference implementation
├── test_mechanism.py      identity, causality, stream/scan, evolve, clip
├── test_wiring.py         LoRA regex, param groups, checkpoint I/O
├── test_session.py        carry lifecycle, staging idempotence, schedule, slicing, hybrid
├── test_loss_mask.py      loss-mask build, protect helpers, reference-count loader
├── test_chat_utils.py     sampling, prompt format, stop-token assembly
├── test_observability.py  telemetry safety, metric collectors
├── test_dataset_spec.py   DatasetSpec registry, source extraction, meta handling
└── test_train_modal_utils.py  resume-path resolver, ppl math, stratified sampling, per-source eval
```

## Fixtures (`conftest.py`)

Shared tiny modules used across tests:
- Small `TTTConfig` with 2-4 layer indices
- Tiny `hidden_size`, `chunk_size`
- Fake tokenizer (dict-based encode/decode) for loss-mask tests
- CPU-only, deterministic seeds

These fixtures let tests exercise the full mechanism in milliseconds
without needing an actual LLM.

## `test_scan_math.py` — mathematical equivalence

**Purpose:** verify that `_scan_forward` (parallel chunk scan) is
bit-exact identical to a naive sequential reference implementation.

Tests:
- Non-session mode (per-forward reset)
- Session mode with prior carry
- With and without `normalize_delta_by_chunk`
- Ragged last chunk (seq_len not divisible by chunk_size)

Invariant: the fancy einsum/cumsum machinery in `_scan_forward` must
produce the same numbers as the obvious per-chunk apply-then-update
loop. If it doesn't, either the reference is wrong or the fast path
has a bug — either way, the mechanism is not what the paper describes.

## `test_mechanism.py` — behavioral invariants

**Purpose:** high-level behavioral checks.

Tests include:
- **Identity at init.** Zero W_target + pass-through conv → TTT layer
  output equals the plain gated MLP output bit-for-bit.
- **Chunk causality.** Modifying token at position `t` doesn't change
  outputs at positions `< t` (respecting chunk boundaries).
- **Stream vs scan equivalence.** Feeding the same token stream
  through both paths produces the same outputs.
- **`evolve=False` freezes updates.** `state.delta` doesn't change
  when `evolve=False`, but the current delta is still applied.
- **Clip is inactive at zero delta.** With W_target zero, clip doesn't
  fire.

Invariant: the mechanism's user-facing behaviors match its documented
contract. Breaking any of these breaks something visible to callers.

## `test_wiring.py` — LoRA + checkpoint I/O

**Purpose:** verify the wiring assumptions in `ttt_wiring.py`.

Tests:
- **LoRA target regex.** `build_lora_target_regex(num_layers, cfg)`
  produces a regex that matches attention + gate/up on every layer,
  down_proj on non-TTT layers, but NEVER down_proj on TTT layers.
- **Param group classification.** Every trainable parameter is
  classified into exactly one of `lora` / `wdown` / `new`; unclassified
  params raise.
- **Save/load round-trip.** `save_ttt_state_dict` → `load_ttt_state_dict`
  restores exact values.
- **Peft-prefix stripping.** Keys are saved without the `base_model.model.`
  prefix, so load works whether or not the model is PEFT-wrapped.
- **Mismatch guard.** Loading a ckpt with wrong number of tensors
  raises `RuntimeError` with a clear message.

Invariant: LoRA never targets TTT-layer down_proj (silent failure
mode: LoRA overwrites the fast-weight initial state).

## `test_session.py` — session lifecycle

**Purpose:** verify the TBPTT-style carry logic.

Tests:
- **Reset clears `carried_delta` and `_next_carried`** everywhere.
- **Advance is a no-op when nothing was staged** (idempotent to skip).
- **Advance is idempotent under gradient checkpointing recompute.**
  Calling forward twice (once "real," once as recompute) shouldn't
  double-stage.
- **Session schedule generation.** `make_session_schedule` produces
  the expected number of sessions and papers, respecting min/max.
- **`build_session_items` slicing.** Papers are optionally sliced per
  `slice_prob` into `slice_min..slice_max` pieces respecting
  `min_slice_tokens`.
- **`make_single_paper_sessions`** produces one session per doc with
  `k ∼ U[single_paper_slices_min, single_paper_slices_max]` pieces.
- **`carried_decay` correctness.** `carried_delta` after N items equals
  `sum(decay^(N-i-1) * per_item_delta_i)`.

Invariant: gradient never crosses the item boundary (via `detach()`);
carry lifecycle helpers do exactly what their names say.

## `test_loss_mask.py` — content-token masking

**Purpose:** verify the loss-mask primitives.

Tests:
- **`apply_loss_mask`** — disabled (None mask + first_tokens=0 → no
  change), input mutation (doesn't clobber `ids`), masking behavior,
  batched input, `first_tokens` masks the first N.
- **`build_common_token_mask`** — `keep_fraction` bounds respected
  (0.0 → keep nothing, 1.0 → keep everything), vocab-size validation
  (out-of-range ids raise), deterministic given same inputs.
- **`common_mask_from_counts`** — threshold logic correct.
- **`protect_token_ids`** — single-piece token variants, multi-piece
  length gate (min_piece_chars=3), all variants (with/without leading
  space, capitalization).
- **`protect_numeric_tokens`** — digits protected, non-digit-only
  skipped, multi-digit accepted.
- **`protect_symbol_tokens`** — listed symbols protected, leading-space
  variants, tuple/set inputs both work.
- **`protect_by_predicate`** — base helper for numeric/symbols; walks
  only masked indices.
- **`apply_protect_passes`** — aggregation of terms + numeric +
  symbols, correct freed counts returned.
- **`load_reference_counts`** — empty path, missing file, valid file,
  vocab-size mismatch (raises `RuntimeError`).
- **Default protect list regression** — the default list is well-formed
  and expands as expected.

Invariant: numbers and math symbols are never accidentally masked
(they're content in ML papers); domain terms are never accidentally
masked (via the protect list).

## `test_chat_utils.py` — chat primitives

**Purpose:** verify chat helpers on CPU.

Tests:
- **`sample_top_p`** — nucleus sampling produces exactly one token;
  respects temperature (T=0.001 → basically argmax); top_k caps
  candidates; distribution is normalized.
- **Chat template application** — Qwen3 template applied correctly
  with/without thinking; system prompt inserted when non-empty.
- **`chat_stop_token_ids`** — assembles the expected set from the
  tokenizer.
- **`strip_chat_specials`** — removes chat-template scaffolding but
  preserves content.
- **`split_thinking`** — extracts `<think>...</think>` content when
  present; returns empty thinking if opener/closer missing or in the
  wrong order.

Invariant: sampling is deterministic given a torch seed; template
application never leaks unrelated tokens.

## `test_observability.py` — telemetry safety

**Purpose:** verify `observability.py` collectors are safe and
correct.

Tests:
- **Telemetry disabled path** — no wandb import; all methods no-op.
- **Telemetry with wandb missing/failing** — degrades to console; no
  exception propagates.
- **`gpu_stats`** returns empty on no-CUDA machines (test env).
- **`param_health`** correctly computes drift and gate stats for a
  fake TTT module tree.
- **Metric collectors do not mutate model state.**

Invariant: telemetry can never crash training, and metric collectors
are read-only.

## `test_train_modal_utils.py` — train_modal.py helpers

**Purpose:** pin the pure helpers inside `train_modal.py` that don't
need Modal or a GPU.

Tests:
- **`_resolve_resume`** — empty string returns `(None, None)`; same-run
  and cross-run path forms resolve correctly; missing files raise
  `FileNotFoundError`.
- **`_token_weighted_ppl`** — matches the token-weighted geometric
  mean of the per-slice perplexities; empty input returns NaN.
- **`_stratified_sample_indices`** — round-robin covers every source
  when there are enough rows; undersized buckets fall through to random
  fill without dropping the target count; determinism per RNG seed.
- **`_per_source_eval_metrics`** — token-weighted per-source PPL for
  the multi-domain eval table; skips empty source labels; matches the
  geometric mean formula.

Invariant: resume paths never silently point at a nonexistent ckpt;
per-slice ppls aggregate to a token-count-weighted geometric mean, not
a simple average.

## `test_dataset_spec.py` — dataset abstraction

**Purpose:** pin the `DatasetSpec` registry and pure source-extraction
logic without pulling in the `datasets` library.

Tests:
- **Registry** — `arxiv` and `slimpajama-6b` are registered;
  `get_dataset_spec` raises on unknown names; SlimPajama spec
  explicitly excludes `RedPajamaCommonCrawl` and includes the other
  six source names.
- **`extract_source_from_row`** — struct-typed meta, JSON-string
  meta, missing meta column, malformed JSON, wrong meta type, missing
  key — all handled without raising, returning `""` on failure.
- **`DatasetSpec` is frozen and hashable.**

Invariant: extracting a source label never crashes on unexpected row
shapes; the extractor tolerates both dict and JSON-string forms
because HF `datasets` returns whichever the parquet schema declared.

## What's NOT tested (deliberate)

- **End-to-end training run.** Requires GPU + dataset + Modal.
- **Actual model forward.** The tests use tiny fake modules, not
  Qwen3. If Qwen3 has a specific quirk (e.g., unusual `d_ff`), we
  don't catch it locally — that's the point of `sanity_check` on
  Modal.
- **LoRA numerical behavior.** Requires PEFT installed and a real
  model.

## Adding new tests

**When to add a test:**
- You added a new pure function (no Modal / GPU deps) — write a unit
  test for it.
- You fixed a bug — write a regression test that would have caught it.
- You changed a shape/behavior contract — update the invariant tests.

**When NOT to add a test:**
- You changed something that requires a GPU or the full model. Use
  `sanity_check` on Modal instead.
- You added a new Modal entrypoint. These aren't unit-testable.

## Related docs

- [architecture.md](architecture.md) — the pure-Python modules being tested
- [development.md](development.md) — pre-commit checklist
