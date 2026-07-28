"""Puts the pre-rebuild tree on the path so parity suites can import both sides.

Appended, never prepended: NewCode must keep winning for every name the two
trees share.
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD_ROOT = str(Path(__file__).resolve().parents[3])

if OLD_ROOT not in sys.path:
    sys.path.append(OLD_ROOT)
