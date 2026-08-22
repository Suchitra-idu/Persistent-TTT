# The mechanism

## The idea

A normal MLP down-projection is fixed at inference:

```
out = z @ W0ᵀ
```

In-Place TTT adds a **fast weight** `S` that the model updates *while reading*,
gated by a learned, per-token scalar:

```
out = z @ W0ᵀ + gate(h) · (η · z @ Sᵀ)
```

`S` starts at zero and is not trained by gradient descent — it is built during
the forward pass, by one of three interchangeable **update rules**
(`TTTConfig.update_rule`). The research question is whether keeping `S`
across a whole session — instead of throwing it away — makes the model better
at the text it is currently working on. See
[research.md](research.md) for how that question is actually measured.

## The update rules

Tokens are processed in chunks of `C` (default 50). `Z` is the MLP's hidden
activation — the pattern the state reads and writes with. `V =
Conv1d(hidden_states) @ W_target` is a learned target — what the state is
trying to reconstruct. `W_target` is zero-initialised, so at training step 0
every TTT layer is bit-identical to the base MLP, regardless of rule.

Both rules below build the same object, `S`, a `[d_model, d_ff]` matrix
functioning as a second, temporary `down_proj`. They disagree on *how* `S`
should change when a new `(z, v)` pair arrives, and that disagreement is the
whole reason two rules exist.

### `hebbian` (default) — `ttt_math.scan`

Named for the classical rule it literally is: Hebb's "cells that fire
together wire together," expressed as an outer product write with no error
term at all:

```
ΔS_k = V_kᵀ Z_k / C_k
S_k  = S_carried + Σ_{j<k} ΔS_j        (chunk k reads only strictly earlier chunks)
out_k = Z_k W0ᵀ + η · Z_k S_kᵀ
```

An outer product `v ⊗ z` binds a target pattern to an activation pattern:
it's a rank-1 (rank-`C` for a whole chunk) associative memory write. Reading
it back with some new activation `z'` recovers a blend of every `v` ever
written, each weighted by how much `z'` resembles the `z` it was written
with (`z'·z_j`, via the same matmul that applies `S`). This is exactly the
content-addressable memory model behind classical Hopfield networks and the
"fast weight programmers" line of work (Schmidhuber, 1992) that linear
attention is now understood to be an instance of — `S` is memory, `z` is the
address, `v` is the value.

Nothing in that write asks whether `S` already predicts `V` well. Every
chunk's contribution is added at full strength regardless of whether it
reinforces what's already stored or contradicts it — there is no mechanism
that could even detect the difference, only accumulation. Bindings can
superpose cleanly when their addresses (`z`'s) are close to orthogonal, or
blur into each other when they overlap, the same interference a Hopfield
memory shows well past its capacity. That's also exactly what makes
`hebbian` embarrassingly **chunk-parallel**: chunk `k`'s own contribution
`ΔS_k` needs nothing from any other chunk to compute — it only depends on
that chunk's own `Z_k`, `V_k`. The whole document's deltas can be computed
independently, all at once, and then combined causally with a single
exclusive prefix-sum (`exclusive_cumsum`) — the same primitive that gives
this function its name, `scan`. That exclusive sum is also the entire
causality guarantee: chunk `k` is built only from chunks strictly before it,
so a token is never predicted using a state that already saw it.

### `delta` / `delta_chunk` — `ttt_math.delta_scan` / `chunked_delta_scan`

The delta rule is the classical Widrow-Hoff / least-mean-squares update
(1960): "delta" is literally the error term, `v − prediction`. Sun et al.
2024's contribution was recognizing that this — not the Hebbian write above
— is the update that makes a fast weight behave like a small model being
*trained*, online, one step per token, during inference:

```
pred_t = η̃(z_t) · (S_t @ z_t)          (η̃ = adaptive step, see below)
δ_t    = (v_t − pred_t) ⊗ z_t          (the residual — this is the gradient)
S_{t+1}= decay · S_t + δ_t
```

Read `S_t @ z_t` as `S`'s current prediction for what `v_t` should be, given
`z_t`. `δ_t` is that prediction's error, turned back into an update the same
way it always is in a one-layer linear model trained by gradient descent —
because that's exactly what this is: `δ_t` is `-∇_S L(S_t)` for the per-token
squared loss `L(S) = ‖v_t − S·z_t‖²`, up to the constant folded into `η`. The
"inner loop" is a training loop; `S` is the parameter it's training.

That framing is what explains the behaviour a pure accumulator can't have:
**as `S` gets better at predicting the kind of tokens it keeps seeing, the
residual shrinks, and so does the update.** A state that has found a good
local fit for the current input distribution stops moving — not because
something external stopped it, but because there's nothing left to correct.
If the input changes character (a new topic, a document boundary), the
residual reappears and `S` moves again to chase it. This is a converging
dynamical system seeking a moving target, not a ledger that only ever grows.

`delta_chunk` is not a cruder approximation of `delta` — it's the same
relationship mini-batch gradient descent has to fully online, per-example
gradient descent: the state is frozen for the length of one chunk, every
token in that chunk computes its residual against that same chunk-start
state, and the chunk's total write batches into a single matmul, the same
parallel shape as `hebbian`'s. This is Sun et al.'s own "dual form" of the
recurrence. What it costs against `delta`: some update quality (a chunk's
later tokens are compared to an already-slightly-stale `S`, not the freshest
possible one), traded for computing an entire chunk in parallel instead of
token by token.

The fundamental reason `delta`/`delta_chunk` can't be as parallel as
`hebbian`, even in the chunked case: `hebbian`'s write needs nothing but its
own chunk's `Z`, `V`. The delta rule's write needs the *state the chunk
started from*, because the residual is measured against a prediction that
state made — so chunk `k`'s delta cannot be computed until chunk `k−1`'s is
known. `delta_chunk` is sequential across `N/chunk_size` chunks for exactly
this reason; `delta` is sequential across all `N` tokens for the same
reason, one level finer.

Two things only `delta`/`delta_chunk` need, both consequences of being an
actual gradient step rather than a fixed write: the adaptive step
`η̃(z) = η / (1 + ‖z‖²)` (a fixed `η` tuned for `hebbian`, which has no
notion of a stability bound, is nowhere near the delta rule's own
step-size × curvature condition, `η·‖z‖² < ~2`, once `z` has outlier-scale
dims — the same divergence condition ordinary gradient descent has), and
`truncate_every` (below), needed because the recurrence's backward pass, not
its forward value, can compound instability across steps.

Code: [`ttt/core/ttt_math.py`](../ttt/core/ttt_math.py).

## Truncated BPTT

`delta`/`delta_chunk` only: the state is detached from the autograd graph
every `truncate_every` steps (tokens for `delta`, chunks for `delta_chunk`),
bounding how far backward can chain through repeated `frobenius_clip` calls.
`truncate_every=1` starves `w_target` of gradient entirely — it only ever
shapes a *later* chunk's read, never its own, so cutting the graph every
step leaves it with no path to the loss at all. `5` is the default.

## The output gate

```
gate = sigmoid(W_gate · h + b_gate)
out  = base_out + gate · ttt_out
```

`W_gate` is zero-initialised (one output unit, so the gate starts *uniform*
across every token): `sigmoid(output_gate_bias_init)`, default `-2.0` →
`≈0.12`. The TTT term starts almost entirely suppressed and has to earn its
way open through training. `gate_mean`/`gate_std`, captured under `no_grad`
every forward, are on `session_eval.ItemRow` — a gate pinned near zero means
the TTT term cannot be reaching the output, whatever the carry itself is
doing.

## Three levels of memory

| Level | Spans | Controlled by |
|---|---|---|
| Within-item scan | one slice of one document | `chunk_size`, `eta` |
| Cross-item carry | one session | `carried_decay`, the strategy |
| Per-source seed | the whole run | everlasting strategy, checkpointed |

Each level is measured separately — see [regimes](#regimes).

## The carry

At an item boundary the item's total delta folds into the session carry:

```
S ← carried_decay · S + Σ_k ΔS_k
```

`carried_decay = 1.0` is a pure sum and grows without bound.
`0.0` keeps only the last item. `0.9–0.95` bounds the magnitude while still
averaging across items. A constant per-item delta converges to `1/(1-decay)`.

Code: [`ttt/core/carry.py`](../ttt/core/carry.py), function `advance`.

## Bounding it

Two magnitude controls, easy to confuse:

| Control | Applied to | Where |
|---|---|---|
| Frobenius clip | the state used for the forward, per chunk | `ttt_math.frobenius_clip` |
| EMA decay | the state crossing an item boundary | `carry.advance` |

The clip does **not** compound: it scales the state at apply time and is never
fed back into the accumulation. The delta crossing an item boundary is
unclipped — the decay is what bounds that.

`state_ratio = ‖η·S‖_F / ‖W0‖_F` is the one number saying whether the fast
weight is doing anything. Near zero: the TTT term is inert.

## Regimes

Six ways to run the same document. A regime fixes three switches: is LoRA on,
does the fast weight evolve within an item, and what does it start from?

| Regime | LoRA | Evolves | Starts from | Reset between items |
|---|---|---|---|---|
| `fresh` | no | no | — | — |
| `lora_only` | yes | no | — | — |
| `cold_carry_off` | yes | yes | zero | yes |
| `cold_carry` | yes | yes | zero | no |
| `carry_off` | yes | yes | trained seed | yes |
| `carry` | yes | yes | trained seed | no |

`fresh` is the only regime with LoRA off — every other regime, `lora_only`
included, is otherwise the trained model as-is. A zero-seed eval measures
`fresh`, `lora_only`, and the `cold_*` trio.

## The result

Differences between regimes decompose additively, so each one measures exactly
one mechanism:

```
Δlora    = fresh          − lora_only         LoRA's own effect
Δwithin  = lora_only      − cold_carry_off    within-item scan
Δbetween = cold_carry_off − cold_carry        cross-item carry
Δseed    = cold_carry     − carry             the trained per-source seed
Δtotal   = fresh          − carry             = Δlora + Δwithin + Δbetween + Δseed
```

Positive means perplexity went down, i.e. the mechanism helped. A carry-scoped
chained eval (`single_doc_eval_v1.run_chained`) reports a different, per-slice
`Δbetween = carry_off.ppl − carry.ppl` — same name, a different question, at
finer granularity. Don't conflate the two.

Code: [`ttt/core/metrics.py`](../ttt/core/metrics.py), function
`gap_decomposition`. Additivity is a property test, not a comment.

## Two state families, not one

The old code blurred these; the rebuild does not (PLAN §9.1).

| | **carry** | **stream_state** |
|---|---|---|
| What | session-persistent fast weight | streaming buffer for generation |
| Used by | training, every eval, the pilot | chat and free-text generation only |
| Path | `scan` | `stream_chunk_delta` / `stream_apply` |

They share no state. "The carry" always means the first. No function named
`*_fast_weights` exists — that name straddled both and let two incompatible
key schemes coexist (PLAN D14 defect 2).
