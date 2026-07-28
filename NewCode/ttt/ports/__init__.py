"""Ring 2 — interfaces to the costly world, in core types.

Every port has a conformance suite that real and fake adapters both pass.
"""

from ttt.ports.clock import Clock
from ttt.ports.compute import Compute
from ttt.ports.data_source import DataSource
from ttt.ports.fast_weights import FastWeights
from ttt.ports.generation import Generation
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage
from ttt.ports.table import Table
from ttt.ports.tokenizer import Tokenizer
from ttt.ports.tracker import Tracker

__all__ = [
    "Clock",
    "Compute",
    "DataSource",
    "FastWeights",
    "Generation",
    "Rng",
    "Storage",
    "Table",
    "Tokenizer",
    "Tracker",
]
