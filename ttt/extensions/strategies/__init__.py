"""Ring 1 — session strategies. Registry: STRATEGIES.

Registration is a side effect of import, so adding a strategy is adding a file
and one line here (D4).
"""

from ttt.extensions.strategies._registry import (
    STRATEGIES,
    Strategy,
    get,
    register,
)
from ttt.extensions.strategies.hybrid import HYBRID, Hybrid
from ttt.extensions.strategies.mixed import MIXED, Mixed

__all__ = [
    "HYBRID",
    "MIXED",
    "STRATEGIES",
    "Hybrid",
    "Mixed",
    "Strategy",
    "get",
    "register",
]
