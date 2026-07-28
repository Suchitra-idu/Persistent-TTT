"""In-Place TTT kernels: chunk deltas, exclusive cumsum, clip, apply.

    dS_k = V_k^T Z_k / C_k
    S_k  = S_carried + sum_{j<k} dS_j
    out  = Z_k W0^T + eta * Z_k S_k^T

The exclusive cumsum is the chunk-causality guarantee.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def n_chunks_for(n_tokens: int, chunk_size: int) -> int:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
    return (n_tokens + chunk_size - 1) // chunk_size


def chunk_token_counts(n_tokens: int, chunk_size: int) -> tuple[int, ...]:
    """Non-padded token count per chunk; only the last may be short."""
    k = n_chunks_for(n_tokens, chunk_size)
    if k == 0:
        return ()
    counts = [chunk_size] * k
    counts[-1] = n_tokens - chunk_size * (k - 1)
    return tuple(counts)


def to_chunks(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """[B, N, D] -> [B, K, C, D], zero-padding the final chunk.

    Padded positions contribute 0 to the delta; they are excluded from the
    normaliser separately.
    """
    b, n, d = x.shape
    k = n_chunks_for(n, chunk_size)
    pad = k * chunk_size - n
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x.view(b, k, chunk_size, d)


def chunk_deltas(
    v_chunks: torch.Tensor,
    z_chunks: torch.Tensor,
    *,
    normalize: bool,
    token_counts: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """[B,K,C,d_model] x [B,K,C,d_ff] -> [B,K,d_model,d_ff].

    Normalising by each chunk's actual token count keeps eta roughly
    independent of chunk_size and stops a short final chunk producing an
    outsized update.
    """
    deltas = torch.einsum("bkcd,bkcf->bkdf", v_chunks, z_chunks)
    if not normalize:
        return deltas
    k = deltas.shape[1]
    counts = token_counts if token_counts is not None else (v_chunks.shape[2],) * k
    if len(counts) != k:
        raise ValueError(
            f"token_counts has {len(counts)} entries but there are {k} chunks"
        )
    divisor = torch.tensor(
        counts, dtype=deltas.dtype, device=deltas.device
    ).clamp_min(1).view(1, k, 1, 1)
    return deltas / divisor


def exclusive_cumsum(deltas: torch.Tensor) -> torch.Tensor:
    """sum_{j<k} delta_j along the chunk axis. Chunk 0 gets zeros."""
    cum = deltas.cumsum(dim=1)
    return torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)


def frobenius_clip(state: torch.Tensor, *, eta: float, tau: float) -> torch.Tensor:
    """Scale each state so ||eta*S||_F <= tau, preserving direction."""
    if tau <= 0.0:
        raise ValueError(f"clip tau must be > 0, got {tau}")
    norm = (eta * state).norm(p="fro", dim=(-2, -1), keepdim=True)
    scale = (tau / norm.clamp_min(1e-12)).clamp(max=1.0)
    return state * scale


def apply_state(
    z_chunks: torch.Tensor, state: torch.Tensor, *, eta: float
) -> torch.Tensor:
    """[B,K,C,d_ff] x [B,K,d_model,d_ff] -> [B,K,C,d_model]."""
    return eta * torch.einsum("bkcf,bkdf->bkcd", z_chunks, state)


def scan(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int,
    eta: float,
    normalize_delta_by_chunk: bool = True,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ttt_out [B,N,d_model], item_delta [B,d_model,d_ff]).

    `ttt_out` is added to `z @ W0^T` by the caller, after the gate. Two things
    are deliberate: the clip applies to the state at apply time and is never
    fed back into the accumulation, so it does not compound; and `item_delta`
    is unclipped, since the EMA in `core.carry` is what bounds the carry.
    Nothing here detaches or casts — the TBPTT boundary is the caller's.
    """
    if z.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"z and v must agree on [B, N], got {tuple(z.shape)} and {tuple(v.shape)}"
        )
    n_tokens = z.shape[1]
    counts = chunk_token_counts(n_tokens, chunk_size)

    z_chunks = to_chunks(z, chunk_size)
    v_chunks = to_chunks(v, chunk_size)

    deltas = chunk_deltas(
        v_chunks, z_chunks, normalize=normalize_delta_by_chunk, token_counts=counts
    )
    state = exclusive_cumsum(deltas)
    if carried is not None:
        state = state + carried.unsqueeze(1).to(state.dtype)
    if clip_tau is not None:
        state = frobenius_clip(state, eta=eta, tau=clip_tau)

    ttt_out = apply_state(z_chunks, state, eta=eta)
    ttt_out = ttt_out.reshape(z.shape[0], -1, ttt_out.shape[-1])[:, :n_tokens, :]
    return ttt_out, deltas.sum(dim=1)


def stream_chunk_delta(
    v_chunk: torch.Tensor,
    z_chunk: torch.Tensor,
    *,
    chunk_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    """One committed chunk's delta: [B, d_model, d_ff].

    Normalises by the configured chunk_size, not an observed count: a stream
    commits only when the buffer is full, so a partial buffer is pending
    rather than short.
    """
    delta = torch.einsum("bcd,bcf->bdf", v_chunk, z_chunk)
    if not normalize:
        return delta
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return delta / chunk_size


def stream_apply(
    z_part: torch.Tensor, state: torch.Tensor, *, eta: float
) -> torch.Tensor:
    """eta * Z S^T for streamed tokens, applied before they are buffered.

    A state is required; before the first commit the caller skips this rather
    than adding a zero tensor whose width it would have to guess.
    """
    return eta * (z_part @ state.to(z_part.dtype).transpose(-1, -2))
