"""Core vocabulary. Ports and plugins are defined in these, not framework types.

`carry` means the session-persistent fast weight and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

# Regimes, the ways a held-out document is read: fresh = the untouched base
# model; lora_only = the trained model with fast weights off; cold_carry_off =
# fast weights on, reset at every slice; cold_carry = fast weights and the carry.
FRESH = "fresh"
LORA_ONLY = "lora_only"
COLD_CARRY_OFF = "cold_carry_off"
COLD_CARRY = "cold_carry"

REGIMES = (FRESH, LORA_ONLY, COLD_CARRY_OFF, COLD_CARRY)


@dataclass(frozen=True)
class WorkItem:
    """A contiguous token range of one document: one forward, one backward."""

    doc_idx: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"empty work item [{self.start}, {self.end}) on doc {self.doc_idx}"
            )

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Session:
    """The carry's lifetime: items sharing one fast-weight accumulation."""

    items: tuple[WorkItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("a session must contain at least one work item")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_tokens(self) -> int:
        return sum(item.n_tokens for item in self.items)

    @property
    def doc_indices(self) -> tuple[int, ...]:
        return tuple(item.doc_idx for item in self.items)


@dataclass(frozen=True, eq=False)
class Carry:
    """fp32 fast-weight deltas keyed by BASE-MODEL layer index.

    One key scheme, no enumeration-keyed twin: that is what makes D14 defect 2
    unrepresentable. Not eq-comparable — tensor equality is a tolerance
    question.
    """

    deltas: Mapping[int, torch.Tensor]

    def __post_init__(self) -> None:
        bad = [k for k in self.deltas if not isinstance(k, int) or isinstance(k, bool)]
        if bad:
            raise TypeError(
                f"Carry keys must be base-model layer indices (int), got {bad!r}"
            )
        object.__setattr__(self, "deltas", dict(self.deltas))

    @classmethod
    def empty(cls) -> "Carry":
        return cls(deltas={})

    def __len__(self) -> int:
        return len(self.deltas)

    @property
    def is_empty(self) -> bool:
        return not self.deltas

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.deltas))

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self.deltas.get(layer_idx)


@dataclass(frozen=True)
class GradStats:
    """Per-group gradient norms, measured before clipping."""

    total_norm: float
    lora: float
    wdown: float
    new: float


@dataclass(frozen=True)
class PplRow:
    """One measured slice. `state_ratio` is ||eta*S||_F / ||W0||_F at its end."""

    n_tokens: int
    ppl: float
    state_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {self.n_tokens}")
        if self.ppl <= 0.0:
            raise ValueError(f"ppl must be positive, got {self.ppl}")


@dataclass(frozen=True)
class SliceRow:
    """One measured slice, keeping its position within its document — `EvalRow`
    already collapses this away, which is exactly what a distance-into-the-
    document breakdown needs back (arXiv 2410.23771 on plain perplexity)."""

    doc_idx: int
    source: str
    regime: str
    slice_index: int
    n_tokens: int
    ppl: float

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(
                f"unknown regime {self.regime!r}; expected one of {list(REGIMES)}"
            )
        if self.slice_index < 0:
            raise ValueError(f"slice_index must be >= 0, got {self.slice_index}")
        if self.ppl <= 0.0:
            raise ValueError(f"ppl must be positive, got {self.ppl}")


@dataclass(frozen=True)
class EvalRow:
    """One document measured under one regime."""

    doc_idx: int
    source: str
    regime: str
    n_tokens: int
    ppl: float
    state_ratio_final: float = 0.0

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(
                f"unknown regime {self.regime!r}; expected one of {list(REGIMES)}"
            )
        if self.ppl <= 0.0:
            raise ValueError(f"ppl must be positive, got {self.ppl}")
