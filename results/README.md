# Results index

Every folder below is named after the training run (the checkpoint folder on
the Modal volume). Inside, `step_<n>/<timestamp>[-<tag>]/` is one `paper_eval`
bundle of that run's checkpoint at step n: raw rows, tables, a manifest with the
command that made it. Base model for all: Qwen3-0.6B, 4000 training documents.
The development held-out set has 106 documents; the confirmatory one (`-fresh`
bundles) 73 documents and 200 new Wikipedia cases.

`paper/` holds the cross-run tables, the figures, the analysis inputs and the
bundle pins (`paper/inputs/bundle_pins.json`). How each table is made is in
[`docs/reproducing.md`](../docs/reproducing.md).

## Runs

| folder | in the paper | seed | steps evaluated |
|---|---|---|---|
| `lora-only-matched` | LoRA-only control: no TTT layers, same data and schedule | 42 | 200, 400, 600 |
| `ttt-v1.1` | ungated carry, Hebbian rule | 42 | 200, 400, 600 |
| `ttt-v1.1-clean-deltachunk` | ungated carry, chunked delta rule | 42 | 100, 200, 400, 600 |
| `delta-noclip` | chunked delta rule with the clip off, a short failed probe | 42 | 100 |
| `hebbian-carry-b` | gated carry, first run, stopped at step 200 | 42 | 100, 200 |
| `hebbian-carry-600` | gated carry, same settings and seed, trained to step 600 | 42 | 400, 600 |
| `hebbian-carry-600-s43` | gated carry | 43 | 200, 400, 600 |
| `gated-carry-1050` | gated carry, trained to the end of the schedule | 42 | 1049 |
| `pmatched-control-600` | parameter-matched control: gated-carry architecture, fast weights never written | 42 | 200, 600 |
| `mixed-carry-600`, `mixed-carry-600-s43` | probe trained across documents (`strategy=mixed`) | 42, 43 | 600 |
| `ablate-c-only`, `ablate-c-only-s43` | ablation: closed gate only (C) | 42, 43 | 200 |
| `ablate-a-c`, `ablate-a-c-s43` | ablation: separate path and closed gate (A, C) | 42, 43 | 200 |
| `ablate-b-c`, `ablate-b-c-s43` | ablation: training through the carry and closed gate (B, C) | 42, 43 | 200 |
| `compare/` | paired comparisons of two bundles: the lead over LoRA-only | | |

The `hebbian-carry-*` folders hold the **gated carry**; the name predates it.
The gated carry is the ungated carry plus three changes: a separate carry path
with its own gate and a zero-initialised scale (`carry_gate`), training through
the carry in windows of two slices (`carry_backprop`), and the main gate starting
nearly closed (`output_gate_bias_init=-6`).

Bundle tags: `-fresh` (the preregistered confirmatory pass), `-tokengain-fresh`
and `-tokengain-generic-fresh` (per-token losses), `-short32-fresh` and
`-short128-fresh` (only the target's first tokens), `-dyneval` (dynamic
evaluation).

## Bundles that need a note

| bundle | note |
|---|---|
| `ttt-v1.1/*`, `ttt-v1.1-clean-deltachunk/step_{200,400,600}` | written before the `fresh` fix, so their `fresh` regime is not the true base model. The analysis replaces it with `fresh` from `lora-only-matched` at the same step, so the tables are correct; the bundles' own `SUMMARY.md` and `regime_gaps` tables are not. All bundles from `step_100` of the delta run onwards are correct as written. |
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
