# Ring 1 — `ttt/extensions/`

The parts of the method that a study varies. Two of them are registries (adding
a strategy or a dataset is adding a file); the TTT layer has one implementation,
so it is a module (D5).

## The TTT layer — `mechanism/`

[`inplace_ttt.py`](../ttt/extensions/mechanism/inplace_ttt.py):

| Name | Does |
|---|---|
| `InPlaceTTTMLP` | The MLP replacement. Holds the pretrained projections, the target network (`v_source_norm`, `target_conv`, `w_target`), the main gate, and, with `carry_gate`, the carry gate and the scale `alpha`. `_scan_forward` is the whole forward; `_carried_term` is the gated carry's read. |
| `reset_carry`, `advance_carry`, `detach_carry` | The carry lifecycle the training loop drives through the `FastWeights` port. |
| `RecomputeAsForward`, `checkpoint_context_fn` | Make gradient checkpointing recompute a layer with the carry it actually read. |
| `pretrained_mlps(model)` | A context in which every TTT layer is the untouched base MLP (the `fresh` regime). |
| `iter_ttt_modules(model)` | Every TTT layer in a model. |

How the layer implements the paper is in [mechanism.md](mechanism.md).

## Session strategies — `strategies/`

A strategy turns the training documents into sessions. The carry is reset
between sessions.

| Name | File | Sessions |
|---|---|---|
| `hybrid` (default) | [`hybrid.py`](../ttt/extensions/strategies/hybrid.py) | one document each; shorter than `carry_min_tokens=2100` tokens it is one item, otherwise 2–6 slices of at least 1000 tokens |
| `mixed` | [`mixed.py`](../ttt/extensions/strategies/mixed.py) | `docs_per_session=5` documents, the next from the same source with probability `same_source_share=0.5`; documents sliced as in `hybrid` |

The protocol and registry are in [`_registry.py`](../ttt/extensions/strategies/_registry.py):
`build` (the epoch's sessions, deterministic given the rng), `count` (items, to
size the learning-rate schedule) and `describe` (one line for the boot log).
A strategy's fields are flags: `cli` exposes every field of every registered
strategy and rejects a field the chosen strategy does not have.

## Datasets — `datasets/`

| Name | File | Is |
|---|---|---|
| `slimpajama-6b` (default) | [`slimpajama_6b.py`](../ttt/extensions/datasets/slimpajama_6b.py) | DKYoon/SlimPajama-6B without CommonCrawl, six sources |
| `fixture` | [`fixture.py`](../ttt/extensions/datasets/fixture.py) | a tiny synthetic multi-source corpus for tests |

Select one with the `TTT_DATASET` environment variable or `dataset=` in the flags.
