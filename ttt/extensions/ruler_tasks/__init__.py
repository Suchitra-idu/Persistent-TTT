"""Ring 1 — RULER task specs. Registry: RULER_TASKS.

Registration is a side effect of import, so adding a task family is adding a
file and one line here.
"""

from ttt.extensions.ruler_tasks import cwe, fwe, niah, qa, vt
from ttt.extensions.ruler_tasks._registry import RULER_TASKS, get, register

__all__ = ["RULER_TASKS", "get", "register", "cwe", "fwe", "niah", "qa", "vt"]
