# The test doctrine

Verbatim from `RESEARCH_ARCHITECTURE.md` §10. These are the rules; the table
above them says *why* each ring is tested the way it is.

| Ring | Test style | Why |
|---|---|---|
| 0 core | **property-based** + a **reference oracle** | invariants, not example outputs |
| 1 extensions | **one contract suite run against every registered plugin** | new plugins auto-tested |
| 2/3 ports/adapters | **one conformance suite every adapter must pass** (real + fake) | fakes must behave like prod |
| 4 app | **orchestration tests on all-fake adapters** | catch loop bugs with no GPU |
| 5 experiments | **smoke + reproducibility** on tiny fake data | schema + same-seed determinism |
| — | real hardware runs: **manual, gated**, never in CI | too slow/expensive |

## The rules

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

## How the rules are mechanised here

Rules 5, 6 and 10 are not left to good intentions:

| Rule | Enforced by |
|---|---|
| 3 (no global seed / clock / env in rings 0–2) | `tests/architecture/test_purity.py` — a source scan for banned idioms, with comments and docstrings excluded |
| 5 (every port has a conformance suite) | `tests/architecture/test_suite_completeness.py` |
| 6 (every registry has a contract suite) | `tests/architecture/test_suite_completeness.py` |
| 10 (speed budget) | the `gpu`, `integration` and `slow` markers, deselected by default in `pyproject.toml` |
| the Dependency Rule itself | `.importlinter`, run by `make check` |

## Directory map

```
tests/
  architecture/   ring-boundary + banned-idiom scans (this tier must stay trivial)
  core/           Ring 0 — property tests (hypothesis) + reference oracles
  extensions/     Ring 1 — one contract suite per registry axis
  ports/          Rings 2/3 — one conformance suite per port, subclassed by every adapter
  app/            Ring 4 — orchestration on all-fake adapters
  experiments/    Ring 5 — smoke + same-seed reproducibility
  parity/         OLD-vs-NEW numeric equivalence (Phase 6, then frozen)
  gpu/            marked `gpu`, deselected by default, run by hand
```

## Naming conventions the architecture tests rely on

- A port `ttt/ports/<name>.py` has a suite `tests/ports/test_<name>.py`
  defining a class whose name ends in `Conformance`. Every adapter for that
  port — real and fake — subclasses it.
- A registry `ttt/extensions/<axis>/_registry.py` has a contract suite
  `tests/extensions/test_<axis>_contract.py`, parametrized over
  `REGISTRY.values()` so registering a plugin auto-enrols it.

## Markers

```
gpu           needs a real GPU. Manual and gated, never in CI.
integration   needs network, a real download, or a live service.
slow          over ~1s. Quarantined out of the default tier.
parity        OLD-vs-NEW numeric equivalence (Phase 6).
```

`pytest` with no arguments runs everything *except* `gpu`, `integration` and
`slow`. Override with an explicit `-m`, e.g. `pytest -m gpu`.
