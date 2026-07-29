"""Filesystem scanning for the architecture tests.

Runs at collection time so each discovered file is its own parametrized case,
keeping loops and branching out of test bodies (RULES.md rule 9). The one
place in the suite allowed to read the filesystem.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "ttt"
TESTS_ROOT = REPO_ROOT / "tests"

PURE_RINGS = ("core", "extensions", "ports")


def python_files(*relative_dirs: str) -> list[Path]:
    found: list[Path] = []
    for relative in relative_dirs:
        found.extend(sorted((PACKAGE_ROOT / relative).rglob("*.py")))
    return [p for p in found if "__pycache__" not in p.parts]


def package_relative(path: Path) -> str:
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
    """Source with comments and string literals blanked, positions preserved.

    A banned idiom named in a docstring is documentation, not a violation.
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


def documentation_and_code_lines(path: Path) -> tuple[int, int]:
    """(documentation lines, code lines) for the comment budget.

    Documentation is docstring lines plus whole-line comments. A trailing
    comment counts as code — it is the cheapest and most useful kind — and a
    multi-line error message is code, not prose, which is why this uses `ast`
    rather than blanking every string literal.
    """
    source = path.read_text(encoding="utf-8")
    documented: set[int] = set()

    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        expression = node.body[0]
        documented.update(range(expression.lineno, expression.end_lineno + 1))

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not token.line[: token.start[1]].strip():
            documented.add(token.start[0])

    non_blank = {
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if line.strip()
    }
    return len(documented), len(non_blank - documented)


def port_modules() -> list[Path]:
    return [p for p in python_files("ports") if p.stem != "__init__"]


def registry_modules() -> list[Path]:
    return [p for p in python_files("extensions") if p.stem == "_registry"]
