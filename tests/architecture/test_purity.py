"""Rings 0-2 are pure: no environment, no clock, no GPU, no global seed, no disk.

import-linter cannot express "may import torch but not touch torch.cuda",
which is where D1 draws the line, so that half is a source scan.
"""

from __future__ import annotations

import re

from tests.architecture import _scan

# (rule, pattern, why). Non-capturing groups so a failure reports the match.
BANNED_IDIOMS = [
    (
        "environment-access",
        r"\bos\.(?:environ|getenv)\b",
        "config is resolved once in Ring 5 and passed inward, frozen (D3)",
    ),
    (
        "wall-clock",
        r"\b(?:time\.(?:time|perf_counter|monotonic)"
        r"|datetime\.(?:now|today|utcnow))\s*\(",
        "the clock is a port; inject it (spec §7)",
    ),
    (
        "cuda",
        r"(?:\btorch\.cuda\b|\.cuda\s*\()",
        "devices belong behind the Compute port; rings 0-2 run on a laptop",
    ),
    (
        "global-torch-seeding",
        r"\btorch\.manual_seed\s*\(",
        "rng is a port; seeding globally makes tests order-dependent",
    ),
    (
        "global-random-seeding",
        r"(?:\brandom\.seed\s*\(|\b(?:np|numpy)\.random\b)",
        "rng is injected, never a module-level global (D1, spec §7)",
    ),
    (
        "filesystem",
        r"(?:(?<![\w.])open\s*\(|\.(?:read_text|write_text|read_bytes|write_bytes)\s*\("
        r"|\btorch\.(?:save|load)\s*\()",
        "persistence belongs behind the Storage port",
    ),
]

PURE_RING_FILES = _scan.python_files(*_scan.PURE_RINGS)



def test_no_pure_ring_file_uses_a_banned_idiom():
    offenders = [
        f"{_scan.package_relative(path)}: {matches[0]!r} ({rule}: {reason})"
        for path in PURE_RING_FILES
        for rule, pattern, reason in BANNED_IDIOMS
        if (matches := re.findall(pattern, _scan.code_only(path)))
    ]

    assert not offenders, "rings 0-2 are pure:\n" + "\n".join(offenders)


def test_the_pure_rings_are_actually_being_scanned():
    assert PURE_RING_FILES, (
        "found no Python files in rings 0-2; the purity scan would pass "
        "vacuously. Check tests/architecture/_scan.py::PACKAGE_ROOT."
    )
