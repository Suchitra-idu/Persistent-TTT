"""Data builders for the core tests. Deterministic per explicit seed."""

from __future__ import annotations


import torch

from ttt.core.types import EvalRow, PplRow, WorkItem

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


def work_items(*spans: tuple[int, int, int]) -> list[WorkItem]:
    return [WorkItem(doc_idx=d, start=s, end=e) for d, s, e in spans]


