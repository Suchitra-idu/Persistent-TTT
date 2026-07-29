"""Ring 1 — dataset specs. Registry: DATASETS.

Registration is a side effect of import, so adding a corpus is adding a file
and one line here.
"""

from ttt.extensions.datasets._registry import DATASETS, DEFAULT, get, register
from ttt.extensions.datasets.fixture import FIXTURE
from ttt.extensions.datasets.slimpajama_6b import SLIMPAJAMA_6B

__all__ = [
    "DATASETS",
    "DEFAULT",
    "FIXTURE",
    "SLIMPAJAMA_6B",
    "get",
    "register",
]
