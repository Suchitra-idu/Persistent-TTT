"""No excessive comments or unnecessary documentation (see /CLAUDE.md).

Documentation is docstring lines plus whole-line comments; trailing comments
and multi-line error messages count as code. The allowance is the larger of
half a file's code lines and a flat ABSOLUTE_ALLOWANCE, so a package
`__init__.py` that is a docstring and nothing else still passes — while a
small file padded with prose does not.
"""

from __future__ import annotations

import pytest

from tests.architecture import _scan

MAX_DOC_RATIO = 0.5
ABSOLUTE_ALLOWANCE = 10

CASES = [
    pytest.param(path, id=_scan.package_relative(path))
    for path in _scan.python_files("")
    + [p for p in _scan.TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
]


@pytest.mark.parametrize("path", CASES)
def test_a_file_is_not_mostly_documentation(path):
    documentation, code = _scan.documentation_and_code_lines(path)
    allowed = max(ABSOLUTE_ALLOWANCE, int(MAX_DOC_RATIO * code))

    assert documentation <= allowed, (
        f"{_scan.package_relative(path)} is {documentation} documentation lines "
        f"to {code} code lines, over its allowance of {allowed}. Cut the prose: "
        "rationale belongs in PLAN.md, a test's name is its documentation, and "
        "a field needing a paragraph needs a better name."
    )
