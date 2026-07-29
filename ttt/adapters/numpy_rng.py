"""NumpyRng — the production Rng, over numpy's PCG64."""

from __future__ import annotations

from typing import MutableSequence

import numpy as np
import torch


class NumpyRng:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def integers(self, low: int, high: int) -> int:
        return int(self._rng.integers(low, high))

    def permutation(self, n: int) -> list[int]:
        return [int(i) for i in self._rng.permutation(n)]

    def shuffle(self, items: MutableSequence) -> None:
        items[:] = [items[i] for i in self.permutation(len(items))]

    def random(self) -> float:
        return float(self._rng.random())

    def torch_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator
