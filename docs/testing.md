# Testing

The doctrine is [tests/RULES.md](../tests/RULES.md). This is how to run it.

## Run

```
make check     # import contracts + the fast tier. The gate.
make test      # tests only
make test-gpu  # the gated tier (needs a GPU; empty until Phase 6)
```

`pytest` with no arguments deselects `gpu`, `integration` and `slow`. Override
with an explicit `-m`.

Current: 464 tests, ~6 seconds.

## Layout

`tests/` mirrors the rings.

| Directory | Style | Why that style |
|---|---|---|
| `core/` | property tests + reference oracles | Correctness is an invariant, not an example |
| `extensions/` | one contract suite per registry | New plugins auto-enrol |
| `ports/` | one conformance suite per port | Fakes must not lie |
| `app/` | orchestration on all-fake adapters | Catch loop bugs with no GPU |
| `experiments/` | smoke + same-seed reproducibility | Schema and determinism |
| `architecture/` | source scans | Boundaries you cannot run get crossed |
| `parity/` | old vs new numerics (Phase 6) | Makes the rewrite safe to believe |
| `gpu/` | marked `gpu`, run by hand | Too slow and expensive for CI |

## The architecture guards

These fail like any other test.

| Check | Catches |
|---|---|
| `.importlinter` `rings` | Any import pointing outward |
| `.importlinter` `app-never-touches-a-concrete-adapter` | `app → adapters`, which a plain layers contract would allow |
| `.importlinter` `frameworks-out-of-inner-rings` | `transformers` / `peft` / `modal` / `wandb` / `datasets` in rings 0–2 |
| `.importlinter` `app-stays-framework-free` | Any framework in Ring 4, `torch` included |
| `test_purity.py` | `os.environ`, wall clock, `torch.cuda`, global seeding, disk access in rings 0–2 |
| `test_suite_completeness.py` | A port with no conformance suite; a registry with no contract suite |
| `test_comment_budget.py` | A file that is more prose than code |

`make guard` proves the first one is live: it plants a `core → transformers`
import, expects rejection, and cleans up.

## Writing a test

- The name states the property. No comment explaining what it asserts.
- Arrange, act, assert — with a blank line between.
- No branching or logic in the body. Loops belong in a parametrize.
- Builders over literals: [`tests/core/_builders.py`](../tests/core/_builders.py).
- Inject rng and clock. `random.Random(seed)` and `torch.Generator()`, never
  `torch.manual_seed`.
- In a hypothesis test use `assume()` to discard a bad example. `pytest.skip`
  aborts the whole property at the first one — that silently unran five tests
  here before it was caught.

## Adding an oracle

Numeric code gets a reference oracle: a slow reimplementation written
*differently* — loops instead of einsums, one element at a time. See
[`tests/core/_oracles.py`](../tests/core/_oracles.py). If the two agree on
random inputs, the fast one's cleverness is safe.

## Naming conventions the guards depend on

```
ttt/ports/<name>.py                 -> tests/ports/test_<name>.py
                                       with a `*Conformance` class
ttt/extensions/<axis>/_registry.py  -> tests/extensions/test_<axis>_contract.py
```

## Markers

| Marker | Means |
|---|---|
| `gpu` | Needs a real GPU. Manual, never in CI |
| `integration` | Needs network, a download, or a live service |
| `slow` | Over ~1s |
| `parity` | Old vs new numeric equivalence |
