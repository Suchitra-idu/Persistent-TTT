"""Filesystem scanning helpers for the architecture tests.

These run at *collection* time so that each discovered file becomes its own
parametrized test case. That keeps the test bodies free of loops and
branching (RULES.md rule 9) and gives one reason to fail per test (rule 2).

This module is a test helper, not a test: it is the one place in the suite
allowed to read the filesystem.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "ttt"
TESTS_ROOT = REPO_ROOT / "tests"

#: Rings 0-2. Everything here must be importable with no I/O, no GPU, no
#: global state, and no wall clock.
PURE_RINGS = ("core", "extensions", "ports")


def python_files(*relative_dirs: str) -> list[Path]:
    """Every ``.py`` file under the given package-relative directories."""
    found: list[Path] = []
    for relative in relative_dirs:
        found.extend(sorted((PACKAGE_ROOT / relative).rglob("*.py")))
    return [p for p in found if "__pycache__" not in p.parts]


def package_relative(path: Path) -> str:
    """``ttt/core/carry.py`` — the form used in test ids and messages."""
    return str(path.relative_to(REPO_ROOT))


_LITERAL_TOKENS = frozenset(
    t
    for t in (
        tokenize.COMMENT,
        tokenize.STRING,
        getattr(tokenize, "FSTRING_MIDDLE", None),
    )
    if t is not None
)


def code_only(path: Path) -> str:
    """Source with every comment and string literal blanked out.

    A banned idiom named in a docstring — "this ring may not read
    ``os.environ``" — is documentation, not a violation. Blanking preserves
    line and column positions so any future line-number reporting stays
    accurate.
    """
    source = path.read_text(encoding="utf-8")
    grid = [list(line) for line in source.splitlines(keepends=True)]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _LITERAL_TOKENS:
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = grid[row - 1]
            first = start_col if row == start_row else 0
            last = end_col if row == end_row else len(line)
            for col in range(first, min(last, len(line))):
                if line[col] != "\n":
                    line[col] = " "
    return "".join("".join(line) for line in grid)


def port_modules() -> list[Path]:
    """One file per port interface (Ring 2), excluding ``__init__``."""
    return [p for p in python_files("ports") if p.stem != "__init__"]


def registry_modules() -> list[Path]:
    """One ``_registry.py`` per Ring 1 extension axis."""
    return [p for p in python_files("extensions") if p.stem == "_registry"]
