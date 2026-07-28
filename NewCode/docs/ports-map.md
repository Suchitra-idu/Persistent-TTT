# Ring 2/3 map — the ports and their adapters

Ring 2 is ten protocols in `core` types. Ring 3 implements each of them at
least twice: the real thing, and a fake fast enough to run in a unit test.
Both pass the same conformance suite — that is the whole point, and it is
mechanically checked (`tests/architecture/test_suite_completeness.py`).

Read [ARCHITECTURE.md](../ARCHITECTURE.md) for where the rings sit and
[PLAN.md](../PLAN.md) for D6 and D14, which are why the split falls this way.

## The ports

| Port | Answers | Real | Fake |
|---|---|---|---|
| [`compute`](../ttt/ports/compute.py) | loss, backward, clip, optimizer step | `torch_compute` | `fake_compute` |
| [`fast_weights`](../ttt/ports/fast_weights.py) | the carry lifecycle | `torch_fast_weights` | `fake_fast_weights` |
| [`generation`](../ttt/ports/generation.py) | next-token logits + KV cache | `torch_generation` | `fake_generation` |
| [`tracker`](../ttt/ports/tracker.py) | metric logging | `wandb_tracker` | `in_memory_tracker`, `console_tracker` |
| [`storage`](../ttt/ports/storage.py) | a flat byte store | `local_storage` | `in_memory_storage` |
| [`table`](../ttt/ports/table.py) | a columnar row store | `hf_table` | `list_table` |
| [`data_source`](../ttt/ports/data_source.py) | where a corpus comes from | `hf_data_source` | `fake_data_source` |
| [`tokenizer`](../ttt/ports/tokenizer.py) | text ↔ ids, chat template | `hf_tokenizer` | `fake_tokenizer` |
| [`clock`](../ttt/ports/clock.py) | monotonic time, run-name stamp | `system_clock` | `fake_clock` |
| [`rng`](../ttt/ports/rng.py) | every draw in the system | `numpy_rng` | `scripted_rng` |

`modal_storage` and `modal_runtime` land in Phase 5 with the rest of the
Modal surface; `local_storage` is Storage's real adapter until then.

## Why ten, when PLAN §2 lists nine

`generation` is the extra one. Prefill/step/cache is a different lifecycle
from loss/backward/step, and D14's chat switch matrix has to be exercisable
with no model loaded — the same argument that split `Compute` from
`FastWeights` in the first place (D6).

## Two things the ports deliberately keep apart

**The two state families.** `FastWeights` covers both, but they are named
separately everywhere and reset separately:

| Call | Clears | Survives |
|---|---|---|
| `reset_carry()` | the session carry | the stream delta |
| `reset_stream()` | the stream delta, pending chunk, conv context | the carry |
| `reset_v_context()` | the conv left-context only | both deltas |

`reset_v_context` is the chat turn boundary. Conflating it with
`reset_stream` would silently delete the only cross-turn memory there is.

**The attention cache and the fast weight.** `Generation.reset_cache()` drops
attention history and nothing else. If it also cleared the fast weight,
D14's `cross_turn` switch would have nothing left to switch.

## One `Carry`, keyed by base-model layer index

`core.types.Carry` is the only snapshot representation, and its keys are
base-model layer indices — `{1, 3, 5, …}` at stride 2, never `{0, 1, 2, …}`
enumeration order. `TorchFastWeights` reads the index out of the module path,
which is the single point where the two schemes could ever diverge, and
`install` rejects an index the model does not have. That is D14 defect 2 made
unrepresentable rather than merely fixed.

## The fakes are not stubs

Each fake reproduces the *observable lifecycle*, not just the signature:

- `FakeFastWeights` runs the same EMA (`core.carry.advance`) on a 1×1 delta,
  and commits the stream family a chunk at a time.
- `FakeCompute` and `FakeGeneration` call `stage()` on the fast weight where a
  real forward would have staged a delta — the same object graph as
  production, with different classes in it.
- `ScriptedRng` validates that a scripted draw is inside the range asked for,
  so a script that has drifted from the schedule under test fails loudly.

## What runs where

The default tier is all fakes plus `torch_*` against a four-layer, no-attention
stand-in model (`tests/ports/_builders.py::TinyCausalLM`) — real gradients,
real optimizer steps, CPU milliseconds. The three HF adapters need a download
and are marked `integration`, deselected by default.
