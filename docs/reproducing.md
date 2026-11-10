# Reproducing the paper

Three stages, each a command:

1. **Train** a model on Modal (GPU): [`train_v1.py`](../ttt/experiments/train_v1.py).
2. **Evaluate** a checkpoint on Modal (GPU): [`paper_eval_v1.py`](../ttt/experiments/paper_eval_v1.py),
   which writes a result bundle under `results/` (see [evaluation.md](evaluation.md)).
3. **Analyse** the bundles locally (CPU, seconds to minutes): the scripts that
   write the paper's tables and figures.

The result bundles of every run in the paper are in `results/`, so stage 3
runs without a GPU or Modal account.

## Run names and what they are

The directory names under `results/` are the run names used at training time.
Several predate the paper's terminology.

| Directory | In the paper | Seed | Trained to |
|---|---|---|---|
| `lora-only-matched` | LoRA-only control | 42 | step 600 |
| `ttt-v1.1` | ungated carry, Hebbian rule | 42 | step 600 |
| `ttt-v1.1-clean-deltachunk` | ungated carry, chunked delta rule | 42 | step 600 |
| `hebbian-carry-b` | gated carry, steps 100 and 200 | 42 | step 200 |
| `hebbian-carry-600` | gated carry, steps 400 and 600 | 42 | step 600 |
| `hebbian-carry-600-s43` | gated carry | 43 | step 600 |
| `gated-carry-1050` | gated carry, full schedule | 42 | step 1049 |
| `pmatched-control-600` | parameter-matched control | 42 | step 600 |
| `mixed-carry-600`, `mixed-carry-600-s43` | probe trained across documents | 42, 43 | step 600 |
| `ablate-c-only`, `-s43` | ablation: closed gate only (C) | 42, 43 | step 200 |
| `ablate-a-c`, `-s43` | ablation: separate path, closed gate (A, C) | 42, 43 | step 200 |
| `ablate-b-c`, `-s43` | ablation: trained through the carry, closed gate (B, C) | 42, 43 | step 200 |
| `delta-noclip` | chunked delta rule without the clip (failed run) | 42 | step 100 |
| `compare/` | paired comparisons of two bundles (lead over LoRA-only) | | |

Inside `results/<run>/step_<n>/`, a bundle's name is its timestamp, with a tag
for special passes: `-fresh` (the preregistered confirmatory set),
`-tokengain-fresh` and `-tokengain-generic-fresh` (per-token losses),
`-short32-fresh` and `-short128-fresh` (only the target's first 32 or 128
tokens), `-dyneval` (dynamic evaluation).

## Stage 1: training

```
modal run --detach ttt/experiments/train_v1.py --flags "<flags>"
```

All of the paper's runs share these flags:

```
update_rule=hebbian,state_aware_gate=false,limit_docs=4000,eval_every=50,save_every=50,log_every=10,run_name=<run>
```

(`state_aware_gate=false` names an option that has since been removed. The CLI
still accepts it at that value, so the recorded commands keep working, and
rejects any other value.) Add, per run:

| Run | Extra flags |
|---|---|
| gated carry | `carry_gate=true,carry_backprop=true,output_gate_bias_init=-6` |
| ungated carry, Hebbian | none |
| ungated carry, delta rule | `update_rule=delta_chunk` |
| parameter-matched control | gated carry flags + `fast_weight_updates=false` |
| probe across documents | gated carry flags + `strategy=mixed` |
| ablation C | `output_gate_bias_init=-6` |
| ablation A, C | `carry_gate=true,output_gate_bias_init=-6` |
| ablation B, C | `carry_backprop=true,output_gate_bias_init=-6` |
| LoRA-only control | `num_layers=1` (gives no TTT layers) |
| seed 43 | `seed=43` |

The schedule is always 1050 optimizer steps (1048 for seed 43). The runs that
stop at step 600 or 200 were stopped once that checkpoint was saved, by
[`plan/tier1/pipeline.sh`](../plan/tier1/pipeline.sh)
`<modal profile> <run> <stop step> <flags> [eval args file]`, which also runs
the evaluations afterwards. Checkpoints go to the Modal volume `ttt-checkpoints`
under `<run>/step_<n>/`.

Before a real run, [`sanity_check_v1.py`](../ttt/experiments/sanity_check_v1.py)
checks the wiring: with `W_target` zeroed, the TTT path must add exactly nothing.

## Stage 2: evaluation

```
modal run ttt/experiments/paper_eval_v1.py --checkpoint <run>/step_<n> --suites <suites> [options] \
    --out-dir results --flags "<the flags the run was trained with>"
```

The exact lines used for each run are in `plan/tier1/evals/*.txt` and
[`plan/fresh_pass.sh`](../plan/fresh_pass.sh) (the confirmatory pass). Every
bundle's `manifest.json` also records the command that made it (`invocation`).
Typical lines, for the gated carry at seed 42:

```
--suites regimes,boundary,context --n-chain 20
--suites relatedness
--suites repeat
--suites relatedness,regimes --n-wiki-shards 2 --wiki-first-shard 2 --wiki-case-seed 2027 --n-related 200
    --unique-related --n-generic 200 --in-context --chain-lengths 1,2,5,10 --overlap-seeds 42,43
    --cases-file fresh --exclude-holdout-seed 0 --n-boot 10000 --tag fresh   (and eval_holdout_seed=1 in --flags)
```

## Stage 3: the tables and figures

Run from the repository root with the project's `.venv`:

| Command | Writes | Paper |
|---|---|---|
| `python -m ttt.experiments.paper_analysis_v1 --control lora-only-matched --run Hebbian=ttt-v1.1 --run Delta=ttt-v1.1-clean-deltachunk --steps 200,400,600 --mechanism-step 600 --w0-norms results/paper/inputs/w0_norms.json` | `results/paper/tables/T1`–`T6` | ungated carries; T6 is the clip saturation |
| `python -m ttt.experiments.cross_run_tables_v1` | `T7`, `T8`, `T11`–`T16` | T11 and T14: gains of the gated carry by step and seed (main table); T13: ablation; T12 and T15: development relatedness and dynamic evaluation; T16: repeat suite; T7 and T8: per source and the domain correlation |
| `python -m ttt.experiments.report_figures_v1` | `results/paper/report_figures/F15`, `F18`, `F19` | F15: training curve; F18: confirmatory relatedness; F19: gain per token position |
| `python -m ttt.experiments.token_gain_analysis_v1 --cases fresh.json <tokengain bundle> ...` | `tables/addendum_a.json` in each bundle | where in the target the carry acts |
| `python -m ttt.experiments.mixed_probe_v1 <gated carry bundle> <probe bundle>` | the probe's contrasts, on the same cases | training across documents |
| `python -m ttt.experiments.compare_checkpoints_v1 <LoRA-only bundle> <candidate bundle>` | a paired comparison, as in `results/compare/` | lead over the LoRA-only control |
| `python -m ttt.experiments.merge_bundles_v1 <out> <shard> ...` | one bundle from several shard bundles of one checkpoint | (used for dynamic evaluation) |

Figures are written to `results/paper/report_figures/`; the paper's copies are in
`Paper_TMLR/figures/` and are copied by hand.

The confirmatory tables in the paper (gains on the 200 preregistered cases, the
preregistered contrasts) are the `tables/relatedness.csv` of the `-fresh`
bundles. The parameter-matched control's contrasts were computed from the raw
rows of its bundles; no committed script writes them.

### Pinned bundles

A step can hold several bundles. The analysis scripts read the bundles listed in
[`results/paper/inputs/bundle_pins.json`](../results/paper/inputs/bundle_pins.json),
keyed `<run>/step_<n>/<suite>`, so a newer bundle added later does not change a
table. The keys `<run>/step_<n>/fresh` and `<run>/step_<n>/token_gain` pin the
confirmatory and per-token bundles that the figures read. A lookup that is not
pinned takes the newest development bundle at that step.
