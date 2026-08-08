"""Ring 1 — dataset specs. Registry: DATASETS.

Registration is a side effect of import, so adding a corpus is adding a file
and one line here.
"""

from ttt.extensions.datasets._registry import DATASETS, DEFAULT, get, register
from ttt.extensions.datasets.fixture import FIXTURE
from ttt.extensions.datasets.lang_transfer import EVAL_LANGS, TRAIN_LANGS
from ttt.extensions.datasets.slimpajama_6b import SLIMPAJAMA_6B

__all__ = [
    "DATASETS",
    "DEFAULT",
    "EVAL_LANGS",
    "FIXTURE",
    "SLIMPAJAMA_6B",
    "TRAIN_LANGS",
    "get",
    "register",
]
