"""Every port has a conformance suite; every registry has a contract suite.

    ttt/ports/<name>.py                -> tests/ports/test_<name>.py + a
                                          `*Conformance` class
    ttt/extensions/<axis>/_registry.py -> tests/extensions/test_<axis>_contract.py
"""

from __future__ import annotations

import re

from tests.architecture import _scan

def test_every_port_has_a_conformance_suite():
    missing = []
    for port in _scan.port_modules():
        suite = _scan.TESTS_ROOT / "ports" / f"test_{port.stem}.py"
        source = suite.read_text() if suite.is_file() else ""
        if not re.search(r"^class\s+\w*Conformance\b", source, re.MULTILINE):
            missing.append(_scan.package_relative(suite))

    assert not missing, f"ports without a `*Conformance` suite (RULES.md rule 5): {missing}"


def test_every_registry_has_a_contract_suite():
    missing = [
        _scan.package_relative(suite)
        for registry in _scan.registry_modules()
        if not (suite := _scan.TESTS_ROOT / "extensions" / f"test_{registry.parent.name}_contract.py").is_file()
    ]

    assert not missing, f"registries without a contract suite (RULES.md rule 6): {missing}"
