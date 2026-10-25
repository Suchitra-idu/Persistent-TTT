# Results index

Every folder below is named after the training run (the checkpoint folder on
the Modal volume). Inside: `step_<n>/<timestamp>/` is one `paper_eval` bundle of
that run's checkpoint at step n. Base model for all: Qwen3-0.6B, 4000 training
documents, seed 42, held-out set of 106 documents.

`paper/` is the cross-run analysis used by the thesis (tables, figures,
`FINDINGS.md`). Treat it as the summary; the folders below are its sources.

## Runs

| folder | what it is | thesis role | steps evaluated |
|---|---|---|---|
| `lora-only-matched` | **LoRA-only control.** No TTT layers, same data, schedule and learning rate as the TTT runs | the matched baseline for every "lead over LoRA" number | 200, 400, 600 |
| `ttt-v1.1` | **Original Hebbian TTT** (naive carry) | Chapters 5 to 7, the naive carry fails | 200, 400, 600 |
| `ttt-v1.1-clean-deltachunk` | **Original delta-rule TTT** (naive carry) | Chapters 5 to 7, the better update rule | 100, 200, 400, 600 |
| `delta-noclip` | Original delta rule **with the read-time clip turned off**, short probe | Chapter 7, removing the clip does not help (gate closes) | 100 |
| `hebbian-carry-b` | **Carry design**, Hebbian, first training run, stopped at step 200 | Chapter 8, early checkpoints | 100, 200 |
| `hebbian-carry-600` | **Carry design**, same configuration and seed as `hebbian-carry-b`, trained to step 600 | Chapter 8, the main result | 400, 600 |

The carry design is `ttt-v1.1` plus three changes: a separate carry path with
its own gate and a zero-initialised scale (`carry_gate`), training through the
carry in windows of two slices (`carry_backprop`), and the main gate starting
nearly closed (`output_gate_bias_init=-6`).

## Bundles that need a note

| bundle | note |
|---|---|
| `ttt-v1.1/*`, `ttt-v1.1-clean-deltachunk/step_{200,400,600}` | written before the `fresh` fix, so their `fresh` regime is not the true base model. The thesis analysis replaces it with `fresh` from `lora-only-matched` at the same step. All bundles from `step_100` of the delta run onwards are correct as written. |
| `hebbian-carry-b/step_200/20261022-031925` | a second bundle at the same step: boundary suite with 20-document chains and the context suite. The other bundle at that step has regimes and 5-document chains. |
| `hebbian-carry-600/step_600` | the only bundle with the context suite at step 600 |
| boundary chains | 5 documents per source (24 cross-document documents) for the original runs and `hebbian-carry-b/step_{100,200}/2026102202*`; 20 per source (100 documents) for `hebbian-carry-600` and `hebbian-carry-b/step_200/20261022-031925` |

## Superseded

| file | note |
|---|---|
| `step600_hebbian_vs_deltachunk.md` | an early comparison written before the `fresh` fix and without the matched control. Superseded by `paper/`. Kept for history only. |

## Runs with no bundle here

Their training logs are in `paper/inputs/`: the no-clip Hebbian probe
(`hebbian-noclip`) and the two failed carry-design attempts
(`hebbian-carrygate-bptt`, `hebbian-carryscale-bptt`).
