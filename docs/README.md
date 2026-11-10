# Documentation

This repository is the code behind the paper *Session-Persistent Test-Time
Training for Language Models: Transfer and Interference of a Carried Fast Weight
Across Documents* (`Paper_TMLR/main.tex`). It adds In-Place TTT layers to
Qwen3-0.6B, carries their fast weight from one slice or document to the next
through a gated path, trains that on SlimPajama-6B, and measures the effect in
perplexity.

Read the pages as files, or run `make docs` and open <http://127.0.0.1:8000>.

## Pages

| Page | Read it for |
|---|---|
| [mechanism.md](mechanism.md) | The method: the TTT layer and the gated carry, equation by equation, with the code for each part |
| [codebase.md](codebase.md) | How the code is organised, and how a training run, an evaluation and the paper's numbers flow through it |
| [evaluation.md](evaluation.md) | Every evaluation suite: what it measures, how, and which paper section uses it |
| [reproducing.md](reproducing.md) | Run names, the exact training and evaluation commands, and which script writes which table and figure |
| [research.md](research.md) | Each finding in the paper, with the suite, runs and table it comes from |
| [glossary.md](glossary.md) | Every term used in the code and the paper |
| [core-map.md](core-map.md) | `ttt/core/`, file by file |
| [extensions-map.md](extensions-map.md) | `ttt/extensions/`: the TTT layer, strategies, datasets |
| [ports-map.md](ports-map.md) | `ttt/ports/` and `ttt/adapters/` |
| [app-map.md](app-map.md) | `ttt/app/`: the data pipeline, the training loop, the suites |
| [experiments-map.md](experiments-map.md) | `ttt/cli.py` and `ttt/experiments/`: the CLI and the entrypoints |
| [testing.md](testing.md) | How the tests are organised and run |

## Reading paths

**To understand the method:** [mechanism.md](mechanism.md), then
[`ttt/core/ttt_math.py`](../ttt/core/ttt_math.py), [`ttt/core/carry.py`](../ttt/core/carry.py)
and [`ttt/extensions/mechanism/inplace_ttt.py`](../ttt/extensions/mechanism/inplace_ttt.py)
(together about 500 lines), then [`tests/core/test_ttt_math.py`](../tests/core/test_ttt_math.py)
for what is guaranteed about them.

**To find your way around the code:** [codebase.md](codebase.md), then the map
of the package you are in.

**To check a number in the paper:** [research.md](research.md) for which output
it comes from, then [reproducing.md](reproducing.md) for the command.

**To run something:** [reproducing.md](reproducing.md).

**To change something:** the "Where to change what" table in
[codebase.md](codebase.md#where-to-change-what), then [testing.md](testing.md).

## Commands

Run from the repository root.

| Command | Does |
|---|---|
| `make install-dev` | Install the package and its development dependencies into `.venv` |
| `make check` | Import contracts and the fast tests: the gate before any change is done |
| `make test` | The fast tests only |
| `make lint` | The import contracts only |
| `make guard` | Proves the import contracts are live, by planting a forbidden import |
| `make test-gpu-modal` | The GPU tests, on a Modal H100 |
| `make docs` / `make docs-build` | Serve or build these pages |

## Also

- [`ARCHITECTURE.md`](../ARCHITECTURE.md): the layer rules and how they are enforced.
- [`REBUILD_PLAN.md`](../REBUILD_PLAN.md): the design decisions D1–D14 cited in the code.
- [`tests/RULES.md`](../tests/RULES.md): the testing rules.
- `plan/PREREGISTRATION.md`: the preregistered confirmatory evaluation.
- `results/paper/FINDINGS.md`: working notes on the results, written during the study.
