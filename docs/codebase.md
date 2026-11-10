# How the codebase fits together

The package is `ttt/`. It is split into six layers ("rings"). Each may import
only the layers inside it, which the import contracts in `.importlinter` check
on every `make check`. The point of the split is that everything except the
outermost layers runs on a laptop CPU in milliseconds, against fake adapters,
so almost all of the code is tested without a GPU.

| Ring | Package | Holds | Imports |
|---|---|---|---|
| 0 | [`ttt/core/`](../ttt/core/) | pure math and logic: the TTT kernels, the carry, statistics, schedules, config dataclasses | torch (CPU), stdlib |
| 1 | [`ttt/extensions/`](../ttt/extensions/) | the TTT layer, the session strategies, the dataset specs | Ring 0, torch |
| 2 | [`ttt/ports/`](../ttt/ports/) | interfaces (`Protocol`s) to everything costly: model compute, fast weights, data, storage, tracking, randomness | Ring 0 |
| 3 | [`ttt/adapters/`](../ttt/adapters/) | the implementations of the ports, a real one and a fake one each, plus the model builder and checkpoint I/O | Rings 0–2, any framework |
| 4 | [`ttt/app/`](../ttt/app/) | the loops: data pipeline, training, every evaluation suite, the statistics over their rows | Rings 0–2 only, never an adapter, never torch directly |
| 5 | [`ttt/cli.py`](../ttt/cli.py), [`ttt/experiments/`](../ttt/experiments/) | the command-line surface and the entrypoints that wire real adapters into the loops | everything |

Per-package file lists: [core](core-map.md), [extensions](extensions-map.md),
[ports and adapters](ports-map.md), [app](app-map.md),
[experiments](experiments-map.md). The design decisions D1–D14 are in
[`REBUILD_PLAN.md`](../REBUILD_PLAN.md) and the layer rules in
[`ARCHITECTURE.md`](../ARCHITECTURE.md).

## One training run, end to end

```
modal run ttt/experiments/train_v1.py --flags "..."
 │
 ├─ cli.parse_flags / cli.from_flags        flags string -> Resolved (TrainConfig, TTTConfig, DatasetSpec, Strategy)
 ├─ _runtime.build                          loads Qwen3, patches the TTT layers, adds LoRA, wires real adapters
 │     model_builder.build_model             base model -> InPlaceTTTMLP on layers 1,3,...,27 -> LoRA
 │     TorchCompute, TorchFastWeights, HfTokenizer, ModalStorage, WandbTracker
 ├─ data_pipeline.load                       SlimPajama-6B -> balanced, tokenized documents
 │     stages: split_holdout → filter_sources → shuffle → prefilter → balance
 │             → tokenize → drop_short → rebalance → cap
 ├─ data_pipeline.holdout                    the held-out documents for in-training evaluation
 └─ train_loop.train                         the loop
       for each session from strategy.build (hybrid: one document, cut into slices):
           fast_weights.reset_carry()
           for each item (slice):
               compute.loss(tokens)          forward; the TTT layer writes S and stages the carry
               compute.backward(...)          graph kept inside the carry window (carry_backprop)
               fast_weights.advance_carry()   promote the staged carry
           every grad_accum_steps items: clip, optimizer step, detach_carry
           on cadence: checkpoint (engine.save) and evaluate (eval_loop on the holdout)
```

The training loop never sees torch. It talks to `Compute` and `FastWeights`,
which is why `tests/app/test_train_loop.py` can test the whole carry lifecycle on
`FakeCompute` and `FakeFastWeights`.

## One evaluation, end to end

```
modal run ttt/experiments/paper_eval_v1.py --checkpoint <run>/step_<n> --suites ... --flags "..."
 │
 ├─ cli.from_flags(resume_from=<run>/step_<n>)
 ├─ _runtime.build(trainable=False)          loads the checkpoint; EvalCompute replaces TorchCompute
 ├─ the held-out documents (data_pipeline.holdout), per source
 ├─ for each suite: an app module returns rows, one per document and condition
 │     regimes -> eval_loop      boundary -> boundary_eval     repeat -> repeat_eval
 │     context -> context_eval   relatedness -> relatedness_eval (cases from _wiki_relatedness)
 │     token_gain -> token_gain  dyneval -> dynamic_eval (on LoRA weights)
 ├─ rows saved as raw/<suite>.json (on the volume and under results/)
 └─ analyse: evidence.* -> estimates with bootstrap intervals -> tables/, figures/, SUMMARY.md
```

What each suite measures is in [evaluation.md](evaluation.md).

## The paper's numbers, end to end

Local, no GPU. The analysis scripts read the raw rows of the bundles under
`results/` (pinned in `results/paper/inputs/bundle_pins.json`), recompute the
estimates with the same code the bundles used (`ttt/app/evidence.py`,
`paper_analysis.py`, `relatedness_eval.estimates`, ...), and write the tables
under `results/paper/tables/` and the figures under
`results/paper/report_figures/`. Which script writes which table is in
[reproducing.md](reproducing.md).

## Configuration

There is one way in: a flags string, `"name=value,name=value"`, parsed by
`cli.parse_flags`. Every field of `TrainConfig`
([`ttt/core/config/train.py`](../ttt/core/config/train.py)), `TTTConfig`
([`ttt/core/config/ttt.py`](../ttt/core/config/ttt.py)) and of the chosen
strategy is a flag, typed from the field's default. Unknown names, and booleans
that are not one of `true/false/1/0/yes/no/on/off`, are errors before anything
reaches a GPU. The resolved configuration is frozen and recorded in every
manifest (`Resolved.describe()`).

## Where to change what

| To change | Edit | Tested by |
|---|---|---|
| the update rule, the clip, the chunking | `ttt/core/ttt_math.py` | `tests/core/test_ttt_math.py` (against reference oracles) |
| how the carry accumulates | `ttt/core/carry.py`, `InPlaceTTTMLP._scan_forward` | `tests/core/test_carry.py`, `tests/extensions/test_inplace_ttt.py` |
| what a session is | a new file in `ttt/extensions/strategies/`, registered in its `__init__` | `tests/extensions/test_strategies_contract.py` runs on every registered strategy |
| the corpus | a new spec in `ttt/extensions/datasets/` | `tests/extensions/test_datasets_contract.py` |
| a new measurement | a module in `ttt/app/` returning rows, a suite in `paper_eval_v1.py` | a test in `tests/app/` on fakes |
| a hyperparameter default | `TrainConfig` or `TTTConfig` | `tests/core/test_config_objects.py` |

A change that needs an inner ring to import an outer one is a sign it belongs
elsewhere; `make check` rejects it.

## Tests

`make check` runs the import contracts and the fast tests (about 1,600, under a
minute). `make test-gpu` runs the GPU tier on a real GPU. See
[testing.md](testing.md).
