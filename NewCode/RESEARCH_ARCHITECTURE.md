# The Research Hexagon

*A layered, plugin-based, test-driven architecture for AI/ML research code — built to be changed often, extended without fear, and navigated by both humans and LLMs.*

This is the whole specification. If you read only this document, you can build a new
research codebase in this architecture correctly, or add to an existing one without
breaking its rules.

---

## 1. Why research code needs its own architecture

Standard app architectures (Clean, Hexagonal, MVC) assume: deterministic logic,
correctness defined by examples, and a cheap edge (a database). Research code breaks
all three:

1. **The edge is expensive, not just impure.** The "framework" at the boundary is a
   GPU on a remote machine costing real money per hour. You must run most of your
   logic *without touching it*.
2. **Behavior is stochastic.** RNG, floating point, hardware variance. Determinism is
   something you *engineer*, not assume.
3. **Correctness is a property, not an example.** "Is this metric right" is a
   mathematical invariant, not a fixed expected output.
4. **Config is the program.** You change behavior by changing config, not code.
5. **Experiments are permanent artifacts, not features.** Code accretes one-off runs
   forever; old ones must stay reproducible, not be refactored away.
6. **The output is metrics and checkpoints,** not a UI response.

The Research Hexagon adapts three classic ideas to these forces:
**Ports & Adapters** (isolate the costly edge), **the Dependency Rule** from Clean
Architecture (imports point inward), and the **Microkernel/Plugin** pattern (a small
core, capabilities added at extension points).

---

## 2. The one rule that defines everything: *add, don't edit*

> Adding a capability should mean **adding a file**, never editing files that already
> work.

Every place where adding one thing forces you to edit N scattered spots is a place
that gets slower and buggier as the repo grows. The entire structure below exists to
make new datasets, models, metrics, and experiments *additive*.

Its corollary: **one concept = one file.** The unit of the codebase must equal the
unit of change. If understanding a feature means opening seven files, the architecture
has already failed.

---

## 3. The rings

Six rings. **Imports point strictly inward.** An inner ring never knows an outer ring
exists.

```
Ring 0  core/         PURE logic & math. No framework, no I/O, no global state.
                      Deterministic given its inputs. Imports: nothing.

Ring 1  extensions/   The science that VARIES, as plugins behind registries:
                      datasets, model/training strategies, metrics, presets.
                      One file per plugin. Imports: core.

Ring 2  ports/        ABSTRACT interfaces to the costly/impure world:
                      Compute, Storage, Tracker, DataSource, Clock, Rng.
                      Imports: core.

Ring 3  adapters/     CONCRETE implementations of ports — real AND fake:
                      GpuCompute / FakeCompute, WandbTracker / InMemoryTracker.
                      Imports: ports (+ frameworks like torch, the tracker SDK).

Ring 4  app/          USE-CASES: the train loop, the eval loop. Orchestration
                      only — it wires ports to plugins. Imports: 0–3, via
                      interfaces (never a concrete adapter directly).

Ring 5  experiments/  One-off compositions + the CLI. Append-only.
        cli.py        Imports inward; NOTHING imports it.
```

**Why the direction matters for you as a builder:** the ring you work in bounds the
context you must hold. Editing `core/` requires zero knowledge of GPUs, trackers, or
Modal — you can hold the entire relevant surface in your head (or an LLM's). This is
what makes the codebase LLM-friendly: the rings are context windows.

### What lives where — concretely

- **core/** — schedule/partition math, sampling, metric aggregation, config
  dataclasses, the pure stages of the data pipeline. Anything you could compute on a
  laptop with no model loaded.
- **extensions/** — a `DatasetSpec` per dataset; a `Strategy` per training regime; a
  metric function per metric. Each behind a registry (§4).
- **ports/** — the interfaces. `Compute` (run a step, get a loss), `Tracker` (log
  metrics), `Storage` (save/load checkpoints), `DataSource` (yield rows). Small, and
  defined in terms of core types, not framework types.
- **adapters/** — the real implementations *and* their test fakes, side by side. The
  fake is a first-class deliverable, not a test afterthought (§6).
- **app/** — the loops. They contain orchestration decisions (when to save, when to
  eval, how to handle a non-finite loss) and *nothing else*. Every substantive
  decision is delegated to a plugin or a port.
- **experiments/** — each experiment is a self-contained file that composes the inner
  rings. It is **append-only**: you never refactor `pilot_v1`; you add `pilot_v2`.

---

## 4. The extension mechanism: registries + protocols

For each axis that changes often, define a **protocol** (the contract) and a
**registry** (the lookup). Plugins self-register; consumers look up by name.

```python
# extensions/strategies/_registry.py
class Strategy(Protocol):
    def build(self, cfg, data, rng) -> list[WorkItem]: ...   # the schedule
    def count(self, cfg, data) -> int: ...                   # for LR schedule sizing
    def describe(self, cfg) -> str: ...                      # one boot-log line

STRATEGIES: dict[str, Strategy] = {}
def register(name):
    def deco(cls): STRATEGIES[name] = cls(); return cls
    return deco
```

```python
# extensions/strategies/my_regime.py  — EVERYTHING about one regime, in one file
@register("my_regime")
class MyRegime:
    config = MyRegimeConfig(...)          # its own knobs, colocated
    def build(self, cfg, data, rng): ...
    def count(self, cfg, data): ...
    def describe(self, cfg): ...
```

The consumer never grows an `if/elif`:

```python
strat = STRATEGIES[cfg.mode]      # one lookup replaces every scattered switch
items = strat.build(cfg, data, rng)
```

Use this at **every axis with ≥2 members**: datasets, strategies, metrics, presets,
optimizers-if-they-vary. Do **not** use it for things with one implementation (§8).

---

## 5. The data plane is a pipeline

Data loading is the classic swamp. Model it as **pipe-and-filter**: an explicit list
of pure `Data -> Data` stages.

```python
PIPELINE = [split_holdout, filter_sources, shuffle, prefilter, balance,
            tokenize, drop_short, rebalance]
```

Each stage is independently testable, reorderable, and — the payoff — **a new stage
is inserted without editing its neighbors**. No more 150-line load function that must
be read top-to-bottom to change one step.

---

## 6. Ports, adapters, and fakes (how the expensive edge stays cheap)

A **port** is an interface owned by the inner rings. Every port has **two or more
adapters**: at least one real, at least one fake.

```python
# ports/compute.py
class Compute(Protocol):
    def loss_and_backward(self, batch, *, accum: int) -> float: ...
    def step(self) -> GradStats: ...

# adapters/fake_compute.py — lets the ENTIRE train loop run in CI, no GPU
class FakeCompute:
    def __init__(self, losses): self._it = iter(losses)   # scripted losses
    def loss_and_backward(self, batch, *, accum): return next(self._it)
    def step(self): return GradStats(...)
```

Because `app/` depends only on the `Compute` *port*, the training loop runs end-to-end
against `FakeCompute` in milliseconds. **Rule of thumb: if a test needs a GPU, you
haven't pushed enough logic inward.**

---

## 7. Config and reproducibility are first-class

- Config is **typed, immutable data** (frozen dataclasses), constructed directly in
  code and tests — never read from global env vars inside the core. Env/CLI resolution
  happens once, in the outer ring, and produces a config object.
- Each plugin owns its config, colocated with it. Deleting the plugin deletes its
  knobs — no orphaned fields pile up.
- A run is reproducible from **(code version + resolved config + seed + data spec)**.
  Experiments record their resolved config so any result can be regenerated.
- RNG and clock are **injected** (they are ports). No global seeding, no wall-clock
  reads in rings 0–4.

---

## 8. Guardrails — how to not over-engineer

Research repos rot from *too much* abstraction as often as too little.

- **Build the seam on the second instance, not the first.** One implementation is a
  function, not a protocol. If you can't name two concrete plugins for a registry,
  don't build the registry yet.
- **Don't protocol-ize single-implementation things** (the model forward, the
  optimizer) just for symmetry.
- **No event buses, DI containers, or blackboards** until a concrete problem forces
  them. They buy decoupling you don't have and add indirection an LLM must trace.
- **Experiments are append-only.** Never refactor an old experiment to share code with
  a new one; copy what you need. Reproducibility beats DRY in the outer ring.

---

## 9. The architecture must be *enforced*, not documented

Documentation drifts and boundaries you can't run get crossed. Make the Dependency
Rule an automated check (e.g. `import-linter` in Python):

```ini
[importlinter:contract:rings]
name = inward-only dependencies
type = layers
layers = experiments | app | adapters | ports | extensions | core
```

A boundary violation is now a **failing test**, not a review comment. This is also the
single highest-leverage LLM-friendliness feature: an agent that reaches from `core/`
into a framework gets an immediate, mechanical "no."

---

## 10. The test doctrine

Sophisticated tests aren't about coverage numbers — they match the **kind of
correctness** each ring has.

| Ring | Test style | Why |
|---|---|---|
| 0 core | **property-based** + a **reference oracle** | invariants, not example outputs |
| 1 extensions | **one contract suite run against every registered plugin** | new plugins auto-tested |
| 2/3 ports/adapters | **one conformance suite every adapter must pass** (real + fake) | fakes must behave like prod |
| 4 app | **orchestration tests on all-fake adapters** | catch loop bugs with no GPU |
| 5 experiments | **smoke + reproducibility** on tiny fake data | schema + same-seed determinism |
| — | real hardware runs: **manual, gated**, never in CI | too slow/expensive |

**Techniques worth naming:**
- *Property tests* — assert invariants over random inputs (e.g. "a partition covers
  the whole range, pieces are disjoint, each meets the minimum").
- *Reference oracle* — a slow, obviously-correct reimplementation compared against the
  fast one on random inputs. The best defense for numeric code.
- *Contract suite* — a single parametrized test over `REGISTRY.values()`. Registering
  a plugin auto-enrolls it. This is "add, don't edit" applied to tests.
- *Conformance suite* — one abstract test class per port; every adapter (including the
  fake) subclasses and must pass it. Guarantees your fakes don't lie.

### The rules (pin these in `tests/RULES.md`)

1. Test behavior and contracts, never implementation — a test survives any refactor
   that preserves behavior.
2. One reason to fail per test; the name states the property being checked.
3. Inject rng and clock. **No** global seed, wall-clock, or network in rings 0–4 tests.
4. **Don't mock what you don't own.** Use fakes at ports; use real objects inside the
   core.
5. Every port has a conformance suite; every adapter (incl. fakes) passes it.
6. Every registry has a contract suite; plugins auto-enroll on registration.
7. Every nontrivial numeric function has a reference oracle.
8. Property tests for invariants; example tests only to pin a specific fixed
   regression (name the bug in the test).
9. Tests are production code: arrange-act-assert, data builders over literals, and
   **no branching or logic** in a test body (logic hides bugs).
10. Speed budget: the whole rings-0–4 suite runs in seconds. Anything slower is marked
    and quarantined to a separate tier.

---

## 11. Directory template

```
project/
  core/           # Ring 0 — pure. property-tested.
  extensions/     # Ring 1 — datasets/, strategies/, metrics/ + their registries
  ports/          # Ring 2 — the interfaces
  adapters/       # Ring 3 — real + fake, side by side
  app/            # Ring 4 — the loops (orchestration only)
  experiments/    # Ring 5 — append-only compositions
  cli.py          # Ring 5 — the single source of CLI args
  tests/
    core/  extensions/  ports/  app/   # mirror the rings
    RULES.md                           # the test doctrine
  ARCHITECTURE.md                      # points here; the map + the ring diagram
  .importlinter                        # the enforced dependency rule
```

---

## 12. Cookbook — how a new dev adds things

Every task is additive. If a recipe makes you edit an inner ring, stop — you're
fighting the architecture.

- **Add a dataset** → new `DatasetSpec` in `extensions/datasets/`. It auto-enrolls in
  the dataset contract suite. Zero edits elsewhere.
- **Add a training regime / model variant** → new file in `extensions/strategies/`
  with one `@register`. The loop picks it up by name; the contract suite tests it.
- **Add a metric** → pure function in `core/metrics.py` (+ a reference-oracle test),
  registered if it's selectable.
- **Add a compute or tracking backend** → new adapter in `adapters/` implementing the
  port; it must pass the existing conformance suite. Nothing else changes.
- **Add an experiment** → new append-only file in `experiments/` composing inner
  rings. Never modify an old one.
- **Change a hyperparameter** → edit config data. Never code.

If you internalize one thing: **inner rings are stable and pure; you build by adding
to the outer rings.** That is the whole architecture.
```
