# Ring 5 map — the CLI and the experiments

Ring 5 is the composition root: the only place that decides which concrete
adapter goes into which port, and the only place that reads the environment.
Nothing imports it.

It is two layers, not one, and the import contracts say so. `ttt/cli.py` turns
arguments into frozen config and touches nothing else — the
`cli-only-resolves-config` contract forbids it from importing `ttt.app`,
`ttt.adapters` or `ttt.ports`. `ttt/experiments/` composes.

| File | Replaces | Entrypoint |
|---|---|---|
| `cli.py` | `_apply_cli_overrides` + `ttt_config`'s import-time env magic | — |
| `train_v1.py` | `train_modal.py::train` | `modal run --detach ttt/experiments/train_v1.py::train` |
| `holdout_eval_v1.py` | `infer_modal.py::holdout_eval` | `modal run ttt/experiments/holdout_eval_v1.py` |
| `single_doc_eval_v1.py` | `single_paper_eval` (D10 rename) | `modal run ttt/experiments/single_doc_eval_v1.py` |
| `holdout_generate_v1.py` | `infer_modal.py::holdout_generate` | `modal run ttt/experiments/holdout_generate_v1.py` |
| `compounding_pilot_v1.py` | `train_modal.py::compounding_pilot` | `modal run ttt/experiments/compounding_pilot_v1.py` |
| `sanity_check_v1.py` | `train_modal.py::sanity_check` | `modal run ttt/experiments/sanity_check_v1.py` |
| `chat_v1.py` | `infer_modal.py::TTTInference` (chat half) | `modal deploy ttt/experiments/chat_v1.py` |
| `chat_repl.py` | `chat_client.py` | `python -m ttt.experiments.chat_repl` |
| `plot_pilot.py` | `plot_pilot.py` | `python -m ttt.experiments.plot_pilot <json>` |

## Passing configuration

Every knob travels in one `--flags` string, as `name=value` pairs:

```
modal run --detach ttt/experiments/train_v1.py --flags "num_epochs=2,strategy=hybrid"
```

Modal builds an entrypoint's CLI from its signature and cannot express
`**kwargs`, so a per-field surface is not available to it. `cli.parse_flags`
coerces each value using that field's declared default, and `cli.resolve`
rejects an unknown name locally — before anything reaches a GPU. An entrypoint's
own arguments (`--n-docs`, `--seeded`) stay separate, because those are not
config fields. `tests/architecture/test_entrypoints.py` enforces the shape.

`session_eval` has no equivalent: it read `.txt` files from a directory, an
arxiv-era manual workflow, and D11 cut it. The session-perplexity *measurement*
it drove survives as `ttt/app/session_eval.py`, which `single_doc_eval_v1` uses.

---

## `cli.py`

Two entry points, because a Modal function signature cannot express `None`:

- **`resolve(**kwargs)`** — the real surface. `None` means "not given". Unknown
  keys raise, naming the valid ones, because a typo costs GPU-hours.
- **`from_flags(**kwargs)`** — maps the sentinels a Modal entrypoint *can*
  express (`0`, `""`, `-1`) onto `None`, then calls `resolve`. The sentinel mess
  stops here.

The settable surface is **derived from the dataclasses**, not listed:

```python
_TRAIN_FIELDS = _settable(TrainConfig, handled={"strategy", "session_training",
                                                "source_preset", "micro_batch_size"})
```

so a new config field is reachable the moment it exists. Strategy knobs are the
union over registered strategies, so a new plugin's knobs auto-enrol — which is
D4's payoff: `--carry-min-tokens` configures the `Hybrid` instance, and passing
it alongside `--strategy everlasting` is an error rather than a silent no-op.

Three things resolve through a *rule* rather than a copy, which is why they are
named arguments instead of pass-through fields:

- **`strategy`** picks the plugin, and a source-scoped one defaults `eval_every`
  to 25 — it has ~10x fewer optimizer steps per epoch, so the usual 100 fires
  too rarely to see anything. An explicit `--eval-every` still wins.
- **`session_training`** is tri-state on the command line (`-1` unset) because
  it is the A1 ablation, not a mode.
- **`source_preset` / `only_sources`** produce *weights*, not a name. The old
  path registered a synthesised `_only_*` preset into a module-level dict as a
  side effect; the new one returns the mapping, which is what let D5 delete the
  preset registry.

`env_defaults(environ)` is the only environment read in the tree. `TTT_DATASET`,
`TTT_MODEL_SIZE` and `TTT_BASE_MODEL` are defaults an explicit argument beats —
the settings the laptop and the container must agree on, which is why
`modal_runtime.FORWARDED_KEYS` bakes exactly those into the image.

---

## The Engine

`_runtime.py` is the composition root and the one file in `experiments/` that
may change; the `*_v1.py` files are append-only. An `Engine` is a bag of ports
plus the two things only a real model can do — persist itself and be resumed —
so an experiment's `run(resolved, *, engine, source)` is pure wiring and the
whole thing runs on fakes with no GPU. That is what `tests/experiments/` does.

`build()` is the real one: transformers, PEFT, the torch adapters. It takes the
storage adapter *and* its mount path separately, because PEFT writes an adapter
directory and the Storage port speaks in blobs. That is a deliberate Ring-5-only
concession rather than a `root` attribute smuggled onto the port.

---

## Modal

`modal_runtime.py` holds the image, volumes, secrets and mounts. Two
dependencies the old image carried are gone (D11): `bitsandbytes`, and a
flash-attn wheel pinned to `torch2.8+cu12+cp311+cxx11abiTRUE` — the most fragile
line in the build. Attention is `sdpa` everywhere.

`modal_storage.py` is `LocalStorage` plus one thing: `commit`. Writes inside a
container are invisible to the next container until the volume is committed,
which is the entire reason the Storage port has that method. It passes the same
conformance suite as the other two adapters, against a volume double that counts
commits.

---

## What the tests cover

| Suite | Asserts |
|---|---|
| `tests/experiments/test_cli.py` | every resolution rule, and that a typo, a bad type and an invalid combination each raise |
| `tests/experiments/test_experiments.py` | every entrypoint runs end to end on fixture data; same seed, same losses |
| `tests/experiments/test_chat_repl.py` | command parsing, switch application, the A/B's restore-on-failure |
| `tests/experiments/test_runtime.py` | resume, the boot log, tracker selection, the pilot's JSON, the sanity check |
| `tests/parity/test_cli_parity.py` | 270 assertions against the *old* `_apply_cli_overrides` |

The sanity check is worth calling out: it used to be GPU-gated, so nothing
proved the TTT identity in CI. It now runs on `TinyCausalLM` on CPU and asserts
the stronger property — at `W_target = 0` the difference is **exactly** zero,
not merely under a tolerance. The `1e-3` tolerance stays for the real model,
where bf16 makes exact zero impossible.
