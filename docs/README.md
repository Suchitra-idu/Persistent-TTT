# Documentation

Deep-dive docs for the In-Place TTT project. The top-level
[`../README.md`](../README.md) is the run protocol and setup guide;
these pages are the reference material.

## Index

**Architecture and mechanism**
- [architecture.md](architecture.md) — module boundaries, import graph, dataflow
- [mechanism.md](mechanism.md) — TTT math + code walkthrough, both execution paths
- [config.md](config.md) — every config field + env var, defaults, sensitivity notes

**Runtime**
- [training.md](training.md) — training loop, sessions, slicing, loss mask, in-loop eval
- [inference.md](inference.md) — eval paths, three-way comparison, snapshot lifecycle
- [chat.md](chat.md) — chat REPL, cross-turn memory invariant, snapshot resume
- [checkpoints.md](checkpoints.md) — save/load semantics, ckpt structure, migration rules

**Data**
- [data.md](data.md) — dataset shape, holdout split, tokenization, loss-mask reference

**Operations**
- [observability.md](observability.md) — full metric reference, healthy vs failure signatures
- [failure-modes.md](failure-modes.md) — known failure patterns and their fixes
- [scaling.md](scaling.md) — model-size scaling: gradient dilution, state saturation, memory

**Development**
- [testing.md](testing.md) — test suite structure, invariants tested, what runs where
- [development.md](development.md) — safe-change checklist, review conventions

## Reading order for a new contributor

1. Top-level `README.md` (setup + run protocol)
2. [architecture.md](architecture.md) — orient in the codebase
3. [mechanism.md](mechanism.md) — understand what TTT is doing
4. [config.md](config.md) — know the knobs
5. Then pick the runtime doc matching what you want to change
   ([training.md](training.md), [inference.md](inference.md), or
   [chat.md](chat.md))
6. Before submitting: [testing.md](testing.md) + [development.md](development.md)

## Reading order for a research user

1. Top-level `README.md`
2. [training.md](training.md) — pick session mode and knobs
3. [inference.md](inference.md) — understand what the eval numbers mean
4. [failure-modes.md](failure-modes.md) when things look wrong
5. [scaling.md](scaling.md) when moving between model sizes

## Doc conventions

- Concrete over abstract. Every claim links back to a file:line where
  possible.
- Include what SHOULD happen alongside what CAN go wrong. Failure modes
  are documented next to correct behavior, not in a separate section.
- Numbers reflect current config defaults in [`../ttt_config.py`](../ttt_config.py).
  When those change, docs referencing them need updating.
- No fictional examples. All commands are copy-pasteable and reflect
  actual entrypoint signatures.
