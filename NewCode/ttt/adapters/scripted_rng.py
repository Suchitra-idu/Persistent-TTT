"""ScriptedRng — draws you wrote down, so a test names the schedule it wants.

Scripts cycle, so three integers drive a thirty-item loop. An unscripted draw
takes the neutral end — `low`, `0.0`, identity — which keeps a test that does
not care about a particular axis from having to script it.
"""

from __future__ import annotations

from typing import MutableSequence, Sequence

import torch


class ScriptedRng:
    def __init__(
        self,
        *,
        integers: Sequence[int] = (),
        randoms: Sequence[float] = (),
        permutations: Sequence[Sequence[int]] = (),
    ) -> None:
        self._integers = tuple(integers)
        self._randoms = tuple(randoms)
        self._permutations = tuple(tuple(p) for p in permutations)
        self._counts = {"integers": 0, "randoms": 0, "permutations": 0}

    def _next(self, axis: str, script: Sequence):
        index = self._counts[axis]
        self._counts[axis] = index + 1
        if not script:
            return None
        return script[index % len(script)]

    def integers(self, low: int, high: int) -> int:
        if high <= low:
            raise ValueError(f"empty range [{low}, {high})")
        drawn = self._next("integers", self._integers)
        if drawn is None:
            return int(low)
        if not low <= drawn < high:
            raise ValueError(
                f"scripted draw {drawn} is outside [{low}, {high}); the script "
                "does not match the schedule under test"
            )
        return int(drawn)

    def permutation(self, n: int) -> list[int]:
        drawn = self._next("permutations", self._permutations)
        if drawn is None:
            return list(range(n))
        if sorted(drawn) != list(range(n)):
            raise ValueError(
                f"scripted permutation {drawn} is not a permutation of range({n})"
            )
        return [int(i) for i in drawn]

    def shuffle(self, items: MutableSequence) -> None:
        order = self.permutation(len(items))
        reordered = [items[i] for i in order]
        items[:] = reordered

    def random(self) -> float:
        drawn = self._next("randoms", self._randoms)
        return 0.0 if drawn is None else float(drawn)

    def torch_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator
