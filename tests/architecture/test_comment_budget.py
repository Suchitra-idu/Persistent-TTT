"""No excessive comments or unnecessary documentation (see /CLAUDE.md).

Documentation is docstring lines plus whole-line comments; trailing comments
and multi-line error messages count as code. The allowance is the larger of
half a file's code lines and a flat ABSOLUTE_ALLOWANCE, so a package
`__init__.py` that is a docstring and nothing else still passes — while a
small file padded with prose does not.
"""

from __future__ import annotations

from tests.architecture import _scan

MAX_DOC_RATIO = 0.5
ABSOLUTE_ALLOWANCE = 10

FILES = _scan.python_files("") + [
    p for p in _scan.TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts
]


def test_no_file_is_mostly_documentation():
    offenders = []
    for path in FILES:
        documentation, code = _scan.documentation_and_code_lines(path)
        allowed = max(ABSOLUTE_ALLOWANCE, int(MAX_DOC_RATIO * code))
        if documentation > allowed:
            offenders.append(f"{_scan.package_relative(path)}: {documentation} doc lines to {code} code, allowance {allowed}")

    assert not offenders, (
        "cut the prose: rationale belongs in REBUILD_PLAN.md, a test's name is its "
        "documentation, and a field needing a paragraph needs a better name.\n" + "\n".join(offenders)
    )
