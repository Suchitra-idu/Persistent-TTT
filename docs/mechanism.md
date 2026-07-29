# The mechanism

## The idea

A normal MLP down-projection is fixed at inference:

```
out = z @ W0ᵀ
```

In-Place TTT adds a **fast weight** `S` that the model updates *while reading*,
from the text it is reading:

```
out = z @ W0ᵀ + η · z @ Sᵀ
```

`S` starts at zero and is not trained by gradient descent. It is built during
the forward pass. The research question is whether keeping `S` across a whole
session — instead of throwing it away — makes the model better at the text it
is currently working on.

## The update

Tokens are processed in chunks of `C` (default 50). Each chunk produces a
rank-C update:

```
ΔS_k = V_kᵀ Z_k / C_k
```

`Z` is the MLP's hidden activation, `V = Conv1d(hidden_states) @ W_target` is a
learned target. `W_target` is zero-initialised, so at training step 0 the TTT
layer is bit-identical to the base MLP.

A token in chunk `k` is processed with the state from **strictly earlier**
chunks:

```
S_k = S_carried + Σ_{j<k} ΔS_j
out_k = Z_k W0ᵀ + η · Z_k S_kᵀ
```

That exclusive sum is the whole causality guarantee: a token is never
predicted using a state that already saw it.

Code: [`ttt/core/ttt_math.py`](../ttt/core/ttt_math.py), function `scan`.

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

Five ways to run the same document. A regime fixes two switches: does the fast
weight evolve within an item, and what does it start from?

| Regime | Evolves | Starts from | Reset between items |
|---|---|---|---|
| `fresh` | no | — | — |
| `cold_carry_off` | yes | zero | yes |
| `cold_carry` | yes | zero | no |
| `carry_off` | yes | trained seed | yes |
| `carry` | yes | trained seed | no |

A zero-seed eval measures only the `cold_*` trio.

## The result

Differences between regimes decompose additively, so each one measures exactly
one mechanism:

```
Δwithin  = fresh          − cold_carry_off    within-item scan
Δbetween = cold_carry_off − cold_carry        cross-item carry
Δseed    = cold_carry     − carry             the trained per-source seed
Δtotal   = fresh          − carry             = Δwithin + Δbetween + Δseed
```

Positive means perplexity went down, i.e. the mechanism helped.

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
