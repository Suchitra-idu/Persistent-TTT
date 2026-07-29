# Architecture

This tree implements **the Research Hexagon** — the specification lives one
directory up, in [`../RESEARCH_ARCHITECTURE.md`](../RESEARCH_ARCHITECTURE.md).
Read that for *why*. This file is the map: where things go, and what stops
them going anywhere else.

The rebuild contract — what gets built, in what order, and the decisions
(D1–D14) behind the deviations noted below — is [`PLAN.md`](PLAN.md).
For the mechanism itself, the vocabulary, and a file-by-file map of Ring 0,
see [`docs/`](docs/README.md).

---

## The rings

Six rings. **Imports point strictly inward.** An inner ring never knows an
outer ring exists.

```
  ┌─────────────────────────────────────────────────────────────┐
  │ Ring 5  ttt/experiments/, cli.py                            │
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
| 5 | `ttt/experiments/`, `cli.py` | everything inward | smoke + same-seed reproducibility |

### Two deviations from the spec's template, both deliberate

1. **The package root is `ttt/`, not bare `core/`** (PLAN.md D2). `import core`
   is a landmine in a repo that also runs on Modal. The rings are unchanged;
   they are just namespaced.
2. **Ring 0 may import `torch`** (PLAN.md D1). Read literally, "no framework"
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

**`.importlinter`** — four contracts:

| Contract | What it stops |
|---|---|
| `rings` | any import that points outward |
| `app-never-touches-a-concrete-adapter` | `app -> adapters`, which a plain `layers` contract would *legalise* (it places adapters below app) |
| `frameworks-out-of-inner-rings` | `transformers` / `peft` / `modal` / `wandb` / `datasets` reaching rings 0–2 |
| `app-stays-framework-free` | Ring 4 touching any framework, `torch` included — it belongs behind the ports |

**`tests/architecture/`** — the half import-linter cannot express: a source
scan banning `os.environ`, wall-clock reads, `torch.cuda`, `torch.manual_seed`,
`random.seed` / `numpy.random`, and filesystem access from rings 0–2. Comments
and string literals are excluded, so *naming* an idiom in a docstring is not a
violation. It also asserts the structural half of the test doctrine (every
port has a conformance suite, every registry has a contract suite) and the
comment budget from [`/CLAUDE.md`](../CLAUDE.md): documentation lines may not
exceed half a file's code lines.

`make guard` proves the enforcement is live: it plants a `core -> transformers`
import, expects `make lint` to reject it, and cleans up.

---

## Where things go

```
NewCode/
  pyproject.toml     deps, pytest config, markers
  Makefile           check / lint / test / test-gpu / guard
  .importlinter      the enforced Dependency Rule
  ARCHITECTURE.md    this file
  PLAN.md            the rebuild contract (phases, decisions D1-D14)
  cli.py             Ring 5 — the single source of CLI args

  ttt/
    core/            types, config (frozen), ttt_math, carry, schedule, tokens,
                     sampling, balance, metrics, naming, sampling_text, report
      config/        TTTConfig, TrainConfig, DatasetSpec, presets, resolve
    extensions/
      datasets/      one DatasetSpec per corpus, behind DATASETS
      strategies/    one session regime per file, behind STRATEGIES
      mechanism/     InPlaceTTTMLP over the core kernels
    ports/           Compute, FastWeights, Tracker, Storage, Table,
                     DataSource, Tokenizer, Clock, Rng
    adapters/        torch_* / wandb_* / modal_* / hf_*  +  their fakes
    app/             data_pipeline, stages, train_loop, eval_loop,
                     session_eval, pilot, chat, generate
    experiments/     *_v1.py — append-only

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

Built side by side with the original flat modules in the parent directory
(PLAN.md D8); nothing there is touched until the parity suite in Phase 6 is
green, and retiring it is a separate explicit call.

**Phases 0–4 complete.** Ring 0 is built and property-tested, Ring 1 holds the
`strategies` and `datasets` registries and the TTT module, Rings 2/3 hold ten
ports with at least one real and one fake adapter each, and Ring 4 holds the
eight loop modules — pipeline, train, eval, session eval, pilot, generate, chat
— tested end to end against fakes. Ring 5 is an empty package awaiting its phase.

Phase 4 added three port methods, each because Ring 4 could not be written
without it: `FastWeights.install(carry, family=...)` and
`FastWeights.decay_stream(factor=...)` are D14 defects 1 and 3, and
`Table.filter(name, predicate)` is what keeps the prefilter stage from
materialising a corpus larger than memory. All three are in every adapter's
conformance suite.

Phase 4 is also the first time `app-stays-framework-free` had a Ring 4 to check.
It now sets `allow_indirect_imports = True`: D1 puts torch in Ring 0, and Ring 4
reads core types and port protocols defined in it. A direct `import torch` in
`ttt/app/` is still rejected, which `make guard` proves alongside the
`core -> transformers` guard it already ran.

`tests/parity/` opened early, with the Phase 2 schedule parity that PLAN §5
requires of this phase. It is marked `parity` and runs by default; the numeric
suite that fills it out is still Phase 6.

Two deviations from PLAN §2's Ring 2/3 list, both recorded in
[`docs/ports-map.md`](docs/ports-map.md): there are **ten** ports rather than
nine — `generation` is separate from `compute` for the reason D6 separates
`compute` from `fast_weights` — and `modal_storage` / `modal_runtime` are
deferred to Phase 5, where PLAN §5 already puts the rest of the Modal surface.
`local_storage` is Storage's real adapter until then.

`tests/adapters/` holds the two Ring 3 framework seams that answer to no port,
`model_builder` and `checkpoint_io`.

One file in `ttt/core/` is not in PLAN §2's list: `lr_schedule.py`. The old
loop got its schedule from `transformers.get_cosine_schedule_with_warmup`, and
Ring 4 may not import a framework, so the warmup+cosine math has to live in
Ring 0. It reproduces HuggingFace's curve exactly inside
`[0, num_training_steps]` and raises past the end, where HuggingFace silently
oscillates back up.
