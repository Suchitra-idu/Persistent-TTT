# Architecture

This tree implements **the Research Hexagon**, specified in
[`RESEARCH_ARCHITECTURE.md`](RESEARCH_ARCHITECTURE.md) at the repository root.
Read that for *why*. This file is the map: where things go, and what stops
them going anywhere else.

The rebuild contract — what gets built, in what order, and the decisions
(D1–D14) behind the deviations noted below — is [`REBUILD_PLAN.md`](REBUILD_PLAN.md).
For the mechanism itself, the vocabulary, and a file-by-file map of Ring 0,
see [`docs/`](docs/README.md).

---

## The rings

Six rings. **Imports point strictly inward.** An inner ring never knows an
outer ring exists.

```
  ┌─────────────────────────────────────────────────────────────┐
  │ Ring 5  ttt/cli.py, ttt/experiments/                        │
  │         append-only compositions. Nothing imports this.     │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │ Ring 4  ttt/app/                                      │  │
  │  │         the loops. Orchestration only, via ports —    │  │
  │  │         never a concrete adapter.                     │  │
  │  │  ┌─────────────────────────────────────────────────┐  │  │
  │  │  │ Ring 3  ttt/adapters/                           │  │  │
  │  │  │         real AND fake, side by side.            │  │  │
  │  │  │  ┌───────────────────────────────────────────┐  │  │  │
  │  │  │  │ Ring 2  ttt/ports/                        │  │  │  │
  │  │  │  │         interfaces to the costly world.   │  │  │  │
  │  │  │  │  ┌─────────────────────────────────────┐  │  │  │  │
  │  │  │  │  │ Ring 1  ttt/extensions/             │  │  │  │  │
  │  │  │  │  │         the science that varies,    │  │  │  │  │
  │  │  │  │  │         as plugins behind registries│  │  │  │  │
  │  │  │  │  │  ┌───────────────────────────────┐  │  │  │  │  │
  │  │  │  │  │  │ Ring 0  ttt/core/             │  │  │  │  │  │
  │  │  │  │  │  │   pure math & logic.          │  │  │  │  │  │
  │  │  │  │  │  │   deterministic, laptop-fast. │  │  │  │  │  │
  │  │  │  │  │  └───────────────────────────────┘  │  │  │  │  │
  │  │  │  │  └─────────────────────────────────────┘  │  │  │  │
  │  │  │  └───────────────────────────────────────────┘  │  │  │
  │  │  └─────────────────────────────────────────────────┘  │  │
  │  └───────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

| Ring | Package | May import | Tested by |
|---|---|---|---|
| 0 | `ttt/core/` | `torch`, `math`, `dataclasses`, stdlib | property tests + reference oracles |
| 1 | `ttt/extensions/` | Ring 0, `torch` | one contract suite per registry |
| 2 | `ttt/ports/` | Ring 0 | one conformance suite per port |
| 3 | `ttt/adapters/` | Rings 0–2 + any framework | the port's conformance suite (real *and* fake) |
| 4 | `ttt/app/` | Rings 0–2, **via interfaces only** | orchestration tests on all-fake adapters |
| 5 | `ttt/cli.py`, `ttt/experiments/` | everything inward | CLI snapshots + smoke + same-seed reproducibility |

### Two deviations from the spec's template, both deliberate

1. **The package root is `ttt/`, not bare `core/`** (REBUILD_PLAN.md D2). `import core`
   is a landmine in a repo that also runs on Modal. The rings are unchanged;
   they are just namespaced.
2. **Ring 0 may import `torch`** (REBUILD_PLAN.md D1). Read literally, "no framework"
   would exile the TTT scan math — the most correctness-critical code here —
   out of the ring where property tests and reference oracles live. The rule
   that carries the spec's intent is *no I/O, no GPU, no global state,
   deterministic, laptop-runnable in milliseconds*, and CPU torch satisfies
   all four. The rest of the ban stands, and both halves are machine-checked.

---

## Enforcement

Boundaries you cannot run get crossed. `make check` is the gate:

```
make check   =   make lint   +   make test
                 │                │
                 │                └─ pytest, minus the gpu/integration/slow tiers
                 └─ import-linter, the Dependency Rule
```

**`.importlinter`** — five contracts:

| Contract | What it stops |
|---|---|
| `rings` | any import that points outward |
| `app-never-touches-a-concrete-adapter` | `app -> adapters`, which a plain `layers` contract would *legalise* (it places adapters below app) |
| `frameworks-out-of-inner-rings` | `transformers` / `peft` / `modal` / `wandb` / `datasets` reaching rings 0–2 |
| `app-stays-framework-free` | Ring 4 touching any framework, `torch` included — it belongs behind the ports |
| `cli-only-resolves-config` | `cli -> app / adapters / ports`, which the layers contract would legalise. Argument resolution loads no model |

**`tests/architecture/`** — the half import-linter cannot express: a source
scan banning `os.environ`, wall-clock reads, `torch.cuda`, `torch.manual_seed`,
`random.seed` / `numpy.random`, and filesystem access from rings 0–2. Comments
and string literals are excluded, so *naming* an idiom in a docstring is not a
violation. It also asserts the structural half of the test doctrine (every
port has a conformance suite, every registry has a contract suite) and the
comment budget from [`CLAUDE.md`](CLAUDE.md): documentation lines may not
exceed half a file's code lines.

`make guard` proves the enforcement is live: it plants a banned import in each
inner ring — `core -> transformers` and `app -> torch` — expects `make lint` to
reject each, and cleans up.

---

## Where things go

```
<repo root>/
  pyproject.toml     deps, pytest config, markers
  Makefile           check / lint / test / test-gpu / guard
  .importlinter      the enforced Dependency Rule
  ARCHITECTURE.md    this file
  REBUILD_PLAN.md    the rebuild contract (phases, decisions D1-D14); historical
  docs/              the documentation; start at docs/README.md

  ttt/
    cli.py           Ring 5 — the single source of CLI args
    core/            types, config (frozen), ttt_math, carry, schedule, lr_schedule,
                     tokens, sampling, balance, stats, metrics, naming, report
      config/        TTTConfig, TrainConfig, DatasetSpec, presets, resolve
    extensions/
      datasets/      one DatasetSpec per corpus, behind DATASETS
      strategies/    one session regime per file, behind STRATEGIES
      mechanism/     InPlaceTTTMLP over the core kernels
    ports/           Compute, FastWeights, Tracker, Storage, Table,
                     DataSource, Tokenizer, Rng
    adapters/        torch_* / wandb_* / modal_* / hf_*  +  their fakes
    app/             data_pipeline, stages, train_loop, the evaluation suites
                     (eval_loop, session_eval, boundary_eval, repeat_eval,
                     context_eval, relatedness_eval, token_gain, dynamic_eval),
                     evidence, paper_analysis, token_gain_analysis
    experiments/     *_v1.py — append-only; _runtime.py is the composition root

  tests/             mirrors the rings; see tests/RULES.md
```

### The cookbook (spec §12), specialised to this tree

| Task | Do this | Edit nothing else |
|---|---|---|
| add a dataset | new file in `ttt/extensions/datasets/` with `@register` | it auto-enrols in the dataset contract suite |
| add a session regime | new file in `ttt/extensions/strategies/` with `@register` | the loop finds it by name |
| add a metric | pure function in `ttt/core/metrics.py` + a reference-oracle test | — |
| add a compute/tracking backend | new adapter in `ttt/adapters/`, subclass the port's conformance suite | — |
| add an experiment | new append-only file in `ttt/experiments/` | never modify an old one |
| change a hyperparameter | edit config data | never code |

If a task makes you edit an inner ring, stop — you are fighting the
architecture.

---

## Status

The rebuild is complete and the old flat modules it replaced are gone. Ring 0
holds the kernels, the carry, statistics and the frozen configs; Ring 1 the TTT
layer and the `strategies` (`hybrid`, `mixed`) and `datasets` registries; Rings
2/3 eight ports, each with a real and a fake adapter; Ring 4 the data pipeline,
the training loop and the evaluation suites; Ring 5 the CLI and the entrypoints
(`train_v1`, `paper_eval_v1`, `sanity_check_v1` on Modal, and the local
analysis scripts). What each file does is in [`docs/`](docs/README.md).

One file in `ttt/core/` is not in REBUILD_PLAN §2's list: `lr_schedule.py`.
Ring 4 may not import a framework, so the warmup+cosine math that used to come
from `transformers.get_cosine_schedule_with_warmup` lives in Ring 0. It
reproduces HuggingFace's curve exactly inside `[0, num_training_steps]` and
raises past the end, where HuggingFace silently oscillates back up.

`app-stays-framework-free` sets `allow_indirect_imports = True`: D1 puts torch
in Ring 0, and Ring 4 reads core types and port protocols defined in it. A
direct `import torch` in `ttt/app/` is still rejected, which `make guard` proves.
