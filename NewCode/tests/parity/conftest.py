"""Puts the pre-rebuild tree on the path so parity suites can import both sides.

Appended, never prepended: NewCode must keep winning for every name the two
trees share.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `ttt_config` reads TTT_DATASET at import and defaults to arxiv, which D9
# dropped. Set before any parity module imports it — collection order decides
# which one gets there first, and a stale singleton is not recoverable.
os.environ.setdefault("TTT_DATASET", "slimpajama-6b")

OLD_ROOT = str(Path(__file__).resolve().parents[3])

if OLD_ROOT not in sys.path:
    sys.path.append(OLD_ROOT)
