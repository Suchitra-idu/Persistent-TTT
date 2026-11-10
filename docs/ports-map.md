# Rings 2 and 3 — ports and adapters

A **port** (`ttt/ports/`) is a `Protocol`: the interface through which the loops
reach something costly or nondeterministic. An **adapter** (`ttt/adapters/`)
implements it. Every port has a real adapter and a fake one, and both pass the
same conformance suite in `tests/ports/`, so a loop tested on the fake behaves
the same on the real thing.

## The ports

| Port | Interface | Real adapter | Fake adapter |
|---|---|---|---|
| [`Compute`](../ttt/ports/compute.py) | `loss`, `backward`, `zero_grad`, `clip_and_step`, `eval_loss`, `eval_token_losses`, `parameter_counts` | [`TorchCompute`](../ttt/adapters/torch_compute.py) (training: AdamW over the three groups); `EvalCompute` (evaluation: losses only) | [`FakeCompute`](../ttt/adapters/fake_compute.py) (scripted losses) |
| [`FastWeights`](../ttt/ports/fast_weights.py) | `set_mode(evolve, session)`, `reset_carry`, `advance_carry`, `detach_carry`, `snapshot`, `install`, `state_ratio`, `gate_stats` | [`TorchFastWeights`](../ttt/adapters/torch_fast_weights.py) (over the patched layers) | [`FakeFastWeights`](../ttt/adapters/fake_fast_weights.py) (the carry as 1x1 tensors, same EMA) |
| [`DataSource`](../ttt/ports/data_source.py) | `load(spec) -> Table` | [`HfDataSource`](../ttt/adapters/hf_data_source.py) (the Hub or local shards) | [`FakeDataSource`](../ttt/adapters/fake_data_source.py) |
| [`Table`](../ttt/ports/table.py) | `column`, `row`, `select`, `filter`, `with_column` | [`HfTable`](../ttt/adapters/hf_table.py) (Arrow, lazy) | [`ListTable`](../ttt/adapters/list_table.py) |
| [`Tokenizer`](../ttt/ports/tokenizer.py) | `encode`, `encode_batch`, `decode` | [`HfTokenizer`](../ttt/adapters/hf_tokenizer.py) | [`FakeTokenizer`](../ttt/adapters/fake_tokenizer.py) (bytes) |
| [`Storage`](../ttt/ports/storage.py) | `exists`, `read_bytes`, `write_bytes`, `listdir`, `commit` | [`LocalStorage`](../ttt/adapters/local_storage.py), [`ModalStorage`](../ttt/adapters/modal_storage.py) (a mounted volume) | [`InMemoryStorage`](../ttt/adapters/in_memory_storage.py) |
| [`Tracker`](../ttt/ports/tracker.py) | `log`, `finish` | [`WandbTracker`](../ttt/adapters/wandb_tracker.py), [`ConsoleTracker`](../ttt/adapters/console_tracker.py) | [`InMemoryTracker`](../ttt/adapters/in_memory_tracker.py) |
| [`Rng`](../ttt/ports/rng.py) | `integers`, `permutation`, `shuffle`, `random` | [`NumpyRng`](../ttt/adapters/numpy_rng.py) | [`ScriptedRng`](../ttt/adapters/scripted_rng.py) |

`Tracker` also defines the metric budget: `outside_budget` names any logged key
that is not on the list, and the training-loop tests assert there is none.

## Adapters that answer to no port

| File | Does |
|---|---|
| [`model_builder.py`](../ttt/adapters/model_builder.py) | Loads the base model, replaces the TTT layers' MLPs with `InPlaceTTTMLP`, adds LoRA (`patch_model_with_ttt`, `build_model`). |
| [`checkpoint_io.py`](../ttt/adapters/checkpoint_io.py) | Saves and loads the TTT parameters (`ttt_params.pt`); keys are base-model-relative, so a checkpoint loads with or without the PEFT wrapper. Older checkpoints also hold a `per_source_carries.pt`, which is no longer read. |
| [`modal_runtime.py`](../ttt/adapters/modal_runtime.py) | The Modal image (Python 3.11, torch 2.8, transformers, peft, datasets), the volumes `ttt-checkpoints` and `hf-hub-cache`, secrets, and the GPU (`H100`). |

## A `Carry` is keyed by base-model layer index

`Carry.deltas` maps a layer index such as `5` to that layer's state tensor.
`TorchFastWeights` reads the index from the module path (`layers.5.mlp`), so a
carry saved from one model cannot be installed into the wrong layer of another;
an unknown index or a wrong shape raises.
