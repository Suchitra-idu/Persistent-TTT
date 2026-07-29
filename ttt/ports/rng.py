"""Rng — every draw in the system, injected (RULES.md rule 3)."""

from __future__ import annotations

from typing import MutableSequence, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class Rng(Protocol):
    def integers(self, low: int, high: int) -> int:
        """Uniform over [low, high). Empty ranges are the caller's bug."""

    def permutation(self, n: int) -> list[int]: ...

    def shuffle(self, items: MutableSequence) -> None: ...

    def random(self) -> float: ...

    def torch_generator(self, seed: int) -> torch.Generator:
        """Token sampling is reproducible without Ring 4 importing torch (D14)."""


def is_permutation(values: Sequence[int], n: int) -> bool:
    return sorted(values) == list(range(n))
