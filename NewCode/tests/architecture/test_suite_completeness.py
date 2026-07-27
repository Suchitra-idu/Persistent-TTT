"""Every port has a conformance suite; every registry has a contract suite.

Rules 5 and 6 of RULES.md are structural claims about the test tree, so they
are checked structurally. The conventions they pin:

    ttt/ports/<name>.py                  -> tests/ports/test_<name>.py
                                            defining a `*Conformance` class
    ttt/extensions/<axis>/_registry.py   -> tests/extensions/test_<axis>_contract.py

Until Phases 2 and 3 land there are no ports and no registries, so these
parametrizations are empty and pytest reports them as skipped. That is the
intended reading: nothing to enforce yet, and the enforcement arms itself the
moment the first port or registry file appears.
"""

from __future__ import annotations

import re

import pytest

from tests.architecture import _scan

PORT_CASES = [
    pytest.param(path, id=_scan.package_relative(path)) for path in _scan.port_modules()
]

REGISTRY_CASES = [
    pytest.param(path, id=_scan.package_relative(path))
    for path in _scan.registry_modules()
]


@pytest.mark.parametrize("port_module", PORT_CASES)
def test_every_port_has_a_conformance_suite(port_module):
    suite = _scan.TESTS_ROOT / "ports" / f"test_{port_module.stem}.py"

    assert suite.is_file(), (
        f"port {_scan.package_relative(port_module)} has no conformance suite; "
        f"expected {_scan.package_relative(suite)} (RULES.md rule 5)"
    )


@pytest.mark.parametrize("port_module", PORT_CASES)
def test_every_port_conformance_suite_defines_a_conformance_base(port_module):
    suite = _scan.TESTS_ROOT / "ports" / f"test_{port_module.stem}.py"

    bases = re.findall(r"^class\s+(\w*Conformance)\b", suite.read_text(), re.MULTILINE)

    assert bases, (
        f"{_scan.package_relative(suite)} defines no `*Conformance` base class, so "
        "adapters have nothing to subclass and the fake is never proven to behave "
        "like the real thing (RULES.md rule 5)"
    )


@pytest.mark.parametrize("registry_module", REGISTRY_CASES)
def test_every_registry_has_a_contract_suite(registry_module):
    axis = registry_module.parent.name
    suite = _scan.TESTS_ROOT / "extensions" / f"test_{axis}_contract.py"

    assert suite.is_file(), (
        f"registry axis {axis!r} has no contract suite; expected "
        f"{_scan.package_relative(suite)}, parametrized over the registry so new "
        "plugins auto-enrol (RULES.md rule 6)"
    )
