"""Data builders for the core tests. Deterministic per explicit seed."""

from __future__ import annotations

import random

import torch

from ttt.core.types import DocRef, EvalRow, PplRow, SliceRow, WorkItem

FP64 = torch.float64


def generator(seed: int) -> torch.Generator:
    """A private torch RNG. Never `torch.manual_seed`."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def randn(*shape: int, seed: int, dtype: torch.dtype = FP64) -> torch.Tensor:
    return torch.randn(*shape, generator=generator(seed), dtype=dtype)


def scan_inputs(
    *,
    seed: int,
    n_tokens: int = 13,
    d_model: int = 6,
    d_ff: int = 8,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        randn(batch, n_tokens, d_ff, seed=seed),
        randn(batch, n_tokens, d_model, seed=seed + 1_000),
    )


def ppl_rows(*pairs: tuple[int, float]) -> list[PplRow]:
    return [PplRow(n_tokens=n, ppl=p) for n, p in pairs]


def eval_row(
    *,
    doc_idx: int = 0,
    source: str = "RedPajamaC4",
    regime: str = "cold_carry",
    n_tokens: int = 1000,
    ppl: float = 10.0,
    state_ratio_final: float = 0.0,
) -> EvalRow:
    return EvalRow(
        doc_idx=doc_idx,
        source=source,
        regime=regime,
        n_tokens=n_tokens,
        ppl=ppl,
        state_ratio_final=state_ratio_final,
    )


def slice_row(
    *,
    doc_idx: int = 0,
    source: str = "RedPajamaC4",
    regime: str = "cold_carry",
    slice_index: int = 0,
    n_tokens: int = 1000,
    ppl: float = 10.0,
) -> SliceRow:
    return SliceRow(
        doc_idx=doc_idx,
        source=source,
        regime=regime,
        slice_index=slice_index,
        n_tokens=n_tokens,
        ppl=ppl,
    )


def doc_refs(*specs: tuple[str, int]) -> list[DocRef]:
    return [
        DocRef(index=i, source=source, n_tokens=n)
        for i, (source, n) in enumerate(specs)
    ]


def work_items(*spans: tuple[int, int, int]) -> list[WorkItem]:
    return [WorkItem(doc_idx=d, start=s, end=e) for d, s, e in spans]


class FakeRng:
    """The Rng port's shape, backed by a private `random.Random`.

    The real adapter and its conformance suite arrive in Phase 3.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def integers(self, low: int, high: int) -> int:
        """Half-open [low, high), like numpy's."""
        return self._rng.randrange(low, high)

    def shuffle(self, items: list) -> None:
        self._rng.shuffle(items)

    def permutation(self, n: int) -> list[int]:
        order = list(range(n))
        self._rng.shuffle(order)
        return order
