"""Source-balancing presets: pure data, not a registry (D5).

Weights are unnormalised; `core.balance` normalises at apply time. Sources
absent from a preset are dropped from the pool for that preset.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

ONLY_PREFIX = "_only_"
NONE = "none"

SOURCE_PRESETS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        # SlimPajama-627B's design proportions.
        "slim-paper": MappingProxyType(
            {
                "RedPajamaC4": 62,
                "RedPajamaGithub": 9,
                "RedPajamaBook": 8,
                "RedPajamaArXiv": 7,
                "RedPajamaWikipedia": 7,
                "RedPajamaStackExchange": 6,
            }
        ),
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
    if name.startswith(ONLY_PREFIX):
        raise KeyError(
            f"{name!r} is a synthesised only-sources preset; build it with "
            "only_sources_preset() rather than looking it up"
        )
    if name not in SOURCE_PRESETS:
        raise KeyError(
            f"unknown source preset {name!r}. Known: {sorted(SOURCE_PRESETS)}"
        )
    return SOURCE_PRESETS[name]


def parse_only_sources(value: str) -> tuple[str, ...]:
    picks = tuple(s.strip() for s in value.split(",") if s.strip())
    if not picks:
        raise ValueError(
            f"--only-sources parsed empty from {value!r}; expected e.g. "
            "'RedPajamaBook' or 'RedPajamaBook,RedPajamaC4'"
        )
    return picks


def only_sources_preset(
    picks: Sequence[str], allowed: Sequence[str] | None = None
) -> tuple[str, Mapping[str, int]]:
    """Equal weights across the picks. Validating against `allowed` turns a
    typo into an error before container spin-up rather than an empty pool."""
    picks = tuple(picks)
    if not picks:
        raise ValueError("only_sources_preset needs at least one source")
    if allowed:
        unknown = [s for s in picks if s not in allowed]
        if unknown:
            raise ValueError(
                f"unknown source(s) {unknown}; allowed: {sorted(allowed)}"
            )
    return ONLY_PREFIX + "_".join(sorted(picks)), MappingProxyType(
        {s: 1 for s in picks}
    )


def resolve_preset_name(requested: str, spec_default: str | None) -> str:
    """Explicit wins, then the spec default, and "none" beats both."""
    if requested.lower() == NONE:
        return ""
    return requested or (spec_default or "")
