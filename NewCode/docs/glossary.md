# Glossary

## The mechanism

| Term | Means |
|---|---|
| **fast weight** `S` | A weight the model updates during the forward pass, not by gradient descent |
| **carry** | The fast weight persisted across items in a session. Always this, never the streaming buffer |
| **stream_state** | The separate buffer used by generation and chat |
| **chunk** | `chunk_size` tokens (default 50); one fast-weight update |
| **η (eta)** | Inner-loop learning rate of the fast weight (default 0.07) |
| **W0 / W_down** | The `down_proj` weight the fast weight adds to |
| **W_target** | Learned projection producing the update target `V`. Zero-init, so TTT starts as identity |
| **carried_decay** | EMA factor when an item's delta folds into the carry |
| **state ratio** | `‖η·S‖_F / ‖W0‖_F` — the carry's magnitude relative to the base weight |
| **clip / tau** | Frobenius bound on `‖η·S‖_F` at apply time |
| **TBPTT** | Truncated backprop through time: the carry crosses an item boundary detached |

## Scheduling

| Term | Means |
|---|---|
| **doc** | One document. (Was "paper" — arxiv-era vocabulary, renamed in D10) |
| **work item** | A contiguous token range of one doc: one forward, one backward |
| **session** | The carry's lifetime — the items it spans before being reset |
| **slice** | A work item cut out of a longer doc |
| **strategy** | The plugin deciding how docs become sessions. `hybrid` or `everlasting` |
| **carry scope** | `session` (reset at each boundary) or `source` (per-source carriers persist) |
| **epoch** | One pass over the document pool |

## Data

| Term | Means |
|---|---|
| **source** | The domain label on every row: `RedPajamaC4`, `RedPajamaBook`, … Never empty (D9) |
| **preset** | Per-source weights for rebalancing the training mix. Unnormalised; only ratios matter |
| **holdout** | The newest `holdout_last_n` rows, reserved for eval. Never trained on |
| **prefilter** | Dropping short docs by estimated tokens, before tokenizing |
| **drop_short** | Dropping short docs by exact token count, after tokenizing |

## Measurement

| Term | Means |
|---|---|
| **regime** | One of the five ways to run a doc — see [mechanism.md](mechanism.md#regimes) |
| **ppl** | Perplexity, `exp(mean token loss)`. Lower is better |
| **token-weighted** | Combining perplexities in log space weighted by token count. The only correct way |
| **Δwithin** | Gain from the within-item chunk scan |
| **Δbetween** | Gain from carrying across items |
| **Δseed** | Gain from starting at the trained per-source carrier |
| **Δtotal** | `Δwithin + Δbetween + Δseed` |

## Architecture

| Term | Means |
|---|---|
| **ring** | One of the six layers. Imports point inward only |
| **port** | An interface to something costly or impure (GPU, disk, network, clock, rng) |
| **adapter** | An implementation of a port. Every port has a real one and a fake one |
| **fake** | A working stand-in that passes the same conformance suite as the real adapter |
| **registry** | Name → plugin lookup for an axis that varies (datasets, strategies) |
| **contract suite** | One test parametrized over a registry; a new plugin auto-enrols |
| **conformance suite** | One test class per port that every adapter must pass |
| **oracle** | A slow, obviously-correct reimplementation used to check a fast one |
| **property test** | Asserts an invariant over random inputs, rather than one example |

## Configuration

| Term | Means |
|---|---|
| **resolved config** | The frozen object produced once, in Ring 5, from defaults + CLI + env |
| **D1–D14** | Rebuild decisions in [PLAN.md](../PLAN.md) §1. Cited in code by number |
| **RQ / SO / A1–A4** | Research questions, sub-objectives and ablations from the proposal |
