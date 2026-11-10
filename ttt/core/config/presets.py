"""Source-mixing presets: pure data, not a registry (D5).

Weights are unnormalised; `core.balance` normalises at apply time. Sources
absent from a preset are dropped from the pool.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

SOURCE_PRESETS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        # C4 downweighted so rare, structured domains produce a legible gap.
        "slim-research": MappingProxyType(
            {
                "RedPajamaC4": 25,
                "RedPajamaGithub": 20,
                "RedPajamaBook": 15,
                "RedPajamaArXiv": 20,
                "RedPajamaWikipedia": 10,
                "RedPajamaStackExchange": 10,
            }
        ),
    }
)


def get_preset(name: str) -> Mapping[str, int]:
    if name not in SOURCE_PRESETS:
        raise KeyError(f"unknown source preset {name!r}. Known: {sorted(SOURCE_PRESETS)}")
    return SOURCE_PRESETS[name]
