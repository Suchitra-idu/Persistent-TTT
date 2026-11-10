# Ring 5 — `ttt/cli.py` and `ttt/experiments/`

The outermost layer: the command-line surface, and the entrypoints that wire the
real adapters into the loops. Nothing imports from here. An entrypoint is
append-only: a new experiment is a new `_v<N>` file, so the code behind a
recorded result is never edited afterwards.

## `cli.py` — flags in, frozen config out

| Name | Does |
|---|---|
| `parse_flags(text)` | `"carry_gate=true,seed=43"` → typed keyword arguments, each typed from its config field's default. A boolean must be one of `true/false/1/0/yes/no/on/off`. |
| `from_flags(**flags)` | → `Resolved`. Every value in the flags string is taken as given, zero included. `limit_docs` unset means the paper's 4000. |
| `resolve(...)` | Builds `TrainConfig`, `TTTConfig`, the `DatasetSpec` and the strategy. Unknown names raise and list the valid ones. |
| `RETIRED` | Removed options still accepted at the one value the recorded commands used (`state_aware_gate=false`). |
| `Resolved.describe()` | Every resolved setting, sorted; recorded in every manifest and W&B run. |
| `env_defaults()` | `TTT_DATASET`, `TTT_MODEL_SIZE`, `TTT_BASE_MODEL`, read here and nowhere else. |
| `resume_path`, `step_path` | `run/step_N` checkpoint paths. A bare `step_N` needs `run_name`. |

Modal builds an entrypoint's command line from its function signature and cannot
express `**kwargs`, so every entrypoint takes one `--flags` string.

## `_runtime.py` — the composition root

`build(resolved, storage=..., root=..., trainable=...)` seeds torch, loads the
base model, patches the TTT layers, adds LoRA, loads a checkpoint if
`resume_from` is set, and returns an `Engine` holding the real adapters
(`TorchCompute` when training, `TorchFastWeights`, `HfTokenizer`, storage, the
model). `tracker(...)` picks W&B or the console. `announce_boot` prints what a
run resolved to.

## The entrypoints

On Modal (GPU):

| File | Does |
|---|---|
| [`train_v1.py`](../ttt/experiments/train_v1.py) | Training. `modal run --detach ttt/experiments/train_v1.py --flags "..."`; needs `run_name`. |
| [`paper_eval_v1.py`](../ttt/experiments/paper_eval_v1.py) | Every evaluation suite for one checkpoint, into one bundle. Also re-renders a bundle locally: `python -m ttt.experiments.paper_eval_v1 <bundle>`. |
| [`sanity_check_v1.py`](../ttt/experiments/sanity_check_v1.py) | The wiring check: with `W_target` zeroed, the TTT path adds exactly zero. |

Local (CPU), over bundles in `results/`:

| File | Does |
|---|---|
| [`paper_analysis_v1.py`](../ttt/experiments/paper_analysis_v1.py) | Tables T1–T6 for a control and a set of runs. Holds `Bundle`, `is_development`, the bundle pins. |
| [`cross_run_tables_v1.py`](../ttt/experiments/cross_run_tables_v1.py) | Tables T7, T8 and T11–T16. |
| [`report_figures_v1.py`](../ttt/experiments/report_figures_v1.py) | The paper's figures F15, F18, F19. |
| [`token_gain_analysis_v1.py`](../ttt/experiments/token_gain_analysis_v1.py) | Addendum A on a token-gain bundle. |
| [`mixed_probe_v1.py`](../ttt/experiments/mixed_probe_v1.py) | The probe's contrasts against the gated carry, on the same cases. |
| [`compare_checkpoints_v1.py`](../ttt/experiments/compare_checkpoints_v1.py) | Two bundles paired document by document. |
| [`merge_bundles_v1.py`](../ttt/experiments/merge_bundles_v1.py) | Shard bundles of one checkpoint merged into one. |

Helpers (leading underscore, imported by the entrypoints):

| File | Does |
|---|---|
| [`_wiki_relatedness.py`](../ttt/experiments/_wiki_relatedness.py) | Loads English Wikipedia and builds the relatedness cases. |
| [`_dynamic_adapter.py`](../ttt/experiments/_dynamic_adapter.py) | Steps and restores a loaded model's LoRA weights, for dynamic evaluation. |

Which command produces which paper table or figure is in
[reproducing.md](reproducing.md).

## Modal

[`ttt/adapters/modal_runtime.py`](../ttt/adapters/modal_runtime.py) defines the
image, the `ttt-checkpoints` volume (checkpoints and raw evidence) and the
`hf-hub-cache` volume (model and dataset downloads), the secrets (`HF_TOKEN`,
W&B), and the GPU. The laptop's `TTT_DATASET`, `TTT_MODEL_SIZE` and
`TTT_BASE_MODEL` are baked into the image, so the container resolves the same
configuration. Select the Modal account with `MODAL_PROFILE`.
