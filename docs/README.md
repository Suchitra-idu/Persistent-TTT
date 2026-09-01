# Docs

Documentation for the rebuilt tree. The root `docs/` describes the old flat
layout and is untouched until cutover (PLAN D13).

Read as files, or run `make docs` from the repo root and open
<http://127.0.0.1:8000>.

| Page | Read it for |
|---|---|
| [mechanism.md](mechanism.md) | What In-Place TTT does, with the formulas |
| [research.md](research.md) | The proposal's two components/Q1–Q2 mapped onto the code, what each required check maps to, the config knobs that matter for the write-up, and empirical findings |
| [glossary.md](glossary.md) | Every term used in the code |
| [core-map.md](core-map.md) | What is in each Ring 0 file |
| [extensions-map.md](extensions-map.md) | The Ring 1 plugins and the TTT module |
| [ports-map.md](ports-map.md) | The Ring 2 ports and their real/fake adapters |
| [app-map.md](app-map.md) | The Ring 4 loops, the pipeline, and the D14 chat switches |
| [experiments-map.md](experiments-map.md) | The CLI surface, the entrypoints, and the Modal runtime |
| [testing.md](testing.md) | How to run and add tests |

Elsewhere: [MIGRATION.md](../MIGRATION.md) for what moved and what was dropped,
[ARCHITECTURE.md](../ARCHITECTURE.md) for the rings,
[PLAN.md](../PLAN.md) for the rebuild decisions D1–D14,
[tests/RULES.md](../tests/RULES.md) for the test doctrine.

## Status

All six phases done: enforcement, Ring 0, Ring 1's two registries plus the TTT
module, Rings 2/3 (ten ports with a real and a fake adapter each), Ring 4's eight
loop modules, Ring 5's CLI and entrypoints, and the OLD-vs-NEW parity suite.
The `gpu` tier is written but has not been run — it needs a real H100.

Cutover is a separate call and is not done: see
[MIGRATION.md](../MIGRATION.md).

## Commands

Run from the repo root.

| Command | Does |
|---|---|
| `make check` | The gate: import contracts + the fast test tier |
| `make test` | Tests only, minus `gpu` / `integration` / `slow` |
| `make lint` | Import contracts only |
| `make guard` | Proves enforcement is live (plants a bad import) |
| `make test-gpu` | The gated tier. Needs a real GPU; empty until Phase 6 |
| `make docs` | Serve these pages with live reload |
| `make docs-build` | Build to `site/`; `--strict`, so a broken link fails |

## Reading order

1. [mechanism.md](mechanism.md) — the research idea.
2. [`ttt/core/ttt_math.py`](../ttt/core/ttt_math.py) and
   [`ttt/core/carry.py`](../ttt/core/carry.py) — that idea as ~130 lines.
3. [`tests/core/test_ttt_math.py`](../tests/core/test_ttt_math.py) — what is
   guaranteed about it.
4. [ARCHITECTURE.md](../ARCHITECTURE.md) — where everything else will go.
