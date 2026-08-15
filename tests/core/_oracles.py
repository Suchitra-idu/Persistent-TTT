"""Slow, obviously-correct reimplementations to test the fast ones against.

Written differently on purpose: loops, no einsums, no padding, no batching.
Test helpers — nothing in `ttt/` may import them.
"""

from __future__ import annotations

import torch


def sequential_scan(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int,
    eta: float,
    normalize: bool = True,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply-then-update, one chunk at a time, for a single batch element.

    The clip goes on the state used for the forward, never fed back into the
    accumulator; accumulating post-clip would compound the bound. `decay`
    only touches the intra-session `accumulated` term, matching
    `ttt_math.exclusive_decayed_cumsum` — `carried` still enters at full
    strength every chunk.
    """
    n_tokens, d_ff = z.shape
    d_model = v.shape[1]
    accumulated = torch.zeros((d_model, d_ff), dtype=z.dtype)
    if carried is not None:
        base = carried.to(z.dtype)
    else:
        base = torch.zeros((d_model, d_ff), dtype=z.dtype)

    out = torch.zeros((n_tokens, d_model), dtype=z.dtype)
    item_delta = torch.zeros((d_model, d_ff), dtype=z.dtype)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        z_chunk = z[start:end]
        v_chunk = v[start:end]

        state = base + accumulated
        if clip_tau is not None:
            norm = float((eta * state).norm(p="fro"))
            state = state * min(1.0, clip_tau / max(norm, 1e-12))
        out[start:end] = eta * (z_chunk @ state.transpose(0, 1))

        delta = v_chunk.transpose(0, 1) @ z_chunk
        if normalize:
            delta = delta / max(end - start, 1)
        accumulated = decay * accumulated + delta
        item_delta = item_delta + delta

    return out, item_delta


def sequential_delta_rule(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    eta: float,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`ttt_math.delta_scan`'s oracle: one token at a time, single batch element.

    Written with `@`/`torch.outer` rather than `delta_scan`'s einsums, so a
    broadcasting bug in the batched kernel has an independent reference.
    Unlike `sequential_scan`, the clip also bounds the accumulator itself —
    see `ttt_math.delta_scan`'s docstring for why, and for `_adaptive_eta`,
    reimplemented inline here rather than imported.
    """
    n_tokens, d_ff = z.shape
    d_model = v.shape[1]
    state = carried.to(z.dtype).clone() if carried is not None else torch.zeros(
        (d_model, d_ff), dtype=z.dtype
    )
    item_delta = torch.zeros_like(state)
    out = torch.zeros((n_tokens, d_model), dtype=z.dtype)

    for t in range(n_tokens):
        read_state = state
        if clip_tau is not None:
            norm = float((eta * state).norm(p="fro"))
            read_state = state * min(1.0, clip_tau / max(norm, 1e-12))
        eta_eff = eta / (1.0 + float(z[t].pow(2).sum()))
        pred = eta_eff * (read_state @ z[t])
        out[t] = pred
        delta = torch.outer(v[t] - pred, z[t])
        item_delta = item_delta + delta
        state = decay * state + delta
        if clip_tau is not None:
            norm = float((eta * state).norm(p="fro"))
            state = state * min(1.0, clip_tau / max(norm, 1e-12))

    return out, item_delta


def sequential_chunked_delta_rule(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int,
    eta: float,
    normalize: bool = True,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`ttt_math.chunked_delta_scan`'s oracle: a chunk at a time, single batch
    element, `@` instead of `chunked_delta_scan`'s einsums."""
    n_tokens, d_ff = z.shape
    d_model = v.shape[1]
    state = carried.to(z.dtype).clone() if carried is not None else torch.zeros(
        (d_model, d_ff), dtype=z.dtype
    )
    item_delta = torch.zeros_like(state)
    out = torch.zeros((n_tokens, d_model), dtype=z.dtype)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        z_chunk, v_chunk = z[start:end], v[start:end]

        read_state = state
        if clip_tau is not None:
            norm = float((eta * state).norm(p="fro"))
            read_state = state * min(1.0, clip_tau / max(norm, 1e-12))

        eta_eff = eta / (1.0 + z_chunk.pow(2).sum(dim=-1, keepdim=True))
        pred = eta_eff * (z_chunk @ read_state.transpose(0, 1))
        out[start:end] = pred

        delta = (v_chunk - pred).transpose(0, 1) @ z_chunk
        if normalize:
            delta = delta / max(end - start, 1)
        item_delta = item_delta + delta
        state = decay * state + delta
        if clip_tau is not None:
            norm = float((eta * state).norm(p="fro"))
            state = state * min(1.0, clip_tau / max(norm, 1e-12))

    return out, item_delta


def sequential_carry(item_deltas: list[torch.Tensor], *, decay: float) -> torch.Tensor:
    """The carry after a session, as an explicit recurrence with decay a parameter."""
    carried = torch.zeros_like(item_deltas[0])
    for delta in item_deltas:
        carried = decay * carried + delta
    return carried


def token_weighted_ppl(pairs: list[tuple[int, float]]) -> float:
    """exp of the token-weighted mean log-ppl, one row at a time."""
    import math

    total_log = 0.0
    total_tokens = 0
    for n_tokens, ppl in pairs:
        total_log += math.log(ppl) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_log / total_tokens)
