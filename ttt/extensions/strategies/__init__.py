"""Ring 1 — session strategies. Registry: STRATEGIES.

Registration is a side effect of import, so adding a strategy is adding a file
and one line here (D4).
"""

from ttt.extensions.strategies._registry import (
    CARRY_SCOPES,
    SESSION,
    SOURCE,
    STRATEGIES,
    Strategy,
    get,
    register,
)
from ttt.extensions.strategies.everlasting import EVERLASTING, Everlasting
from ttt.extensions.strategies.hybrid import HYBRID, Hybrid
from ttt.extensions.strategies.minilasting import MINILASTING, Minilasting
from ttt.extensions.strategies.mixed import MIXED, Mixed

__all__ = [
    "CARRY_SCOPES",
    "EVERLASTING",
    "HYBRID",
    "MINILASTING",
    "MIXED",
    "SESSION",
    "SOURCE",
    "STRATEGIES",
    "Everlasting",
    "Hybrid",
    "Minilasting",
    "Mixed",
    "Strategy",
    "get",
    "register",
]
