"""In-Place TTT kernels: chunk deltas, exclusive cumsum, clip, apply.

    dS_k = V_k^T Z_k / C_k
    S_k  = S_carried + sum_{j<k} decay**(k-1-j) dS_j
    out  = Z_k W0^T + eta * Z_k S_k^T

The exclusive cumsum (decay=1.0) is the chunk-causality guarantee; decay<1.0
is the same recurrence `core.carry.advance` uses across items, one chunk at
a time — otherwise a long enough document diverges on `clip_tau` alone.
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


def exclusive_decayed_cumsum(deltas: torch.Tensor, *, decay: float) -> torch.Tensor:
    """S_0 = 0; S_k = decay*S_{k-1} + delta_{k-1} — `core.carry.advance`'s
    recurrence, one chunk at a time instead of one item at a time. decay=1.0
    is `exclusive_cumsum`; a plain matmul against a decay-weighted lower
    triangle stays stable for decay<1 where rescaling the running sum by
    decay**-k would overflow for a long document."""
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    if decay == 1.0:
        return exclusive_cumsum(deltas)
    k = deltas.shape[1]
    row = torch.arange(k, device=deltas.device).unsqueeze(1)
    col = torch.arange(k, device=deltas.device).unsqueeze(0)
    age = (row - col - 1).clamp(min=0).to(deltas.dtype)
    weights = torch.where(row > col, decay**age, torch.zeros_like(age))
    return torch.einsum("rc,bcxy->brxy", weights, deltas)


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
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ttt_out [B,N,d_model], item_delta [B,d_model,d_ff]).

    `ttt_out` is added to `z @ W0^T` by the caller, after the gate. Two things
    are deliberate: the clip applies to the state at apply time and is never
    fed back into the accumulation, so it does not compound; and `item_delta`
    is unclipped, since the EMA in `core.carry` is what bounds the carry.
    `decay` only reaches the intra-session accumulation; `carried` (the
    cross-item state) is added on top at full strength either way. Nothing
    here detaches or casts — the TBPTT boundary is the caller's.
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
    state = exclusive_decayed_cumsum(deltas, decay=decay)
    if carried is not None:
        state = state + carried.unsqueeze(1).to(state.dtype)
    if clip_tau is not None:
        state = frobenius_clip(state, eta=eta, tau=clip_tau)

    ttt_out = apply_state(z_chunks, state, eta=eta)
    ttt_out = ttt_out.reshape(z.shape[0], -1, ttt_out.shape[-1])[:, :n_tokens, :]
    return ttt_out, deltas.sum(dim=1)


def _adaptive_eta(z: torch.Tensor, eta: float) -> torch.Tensor:
    """Normalized-LMS step size: eta / (1 + ||z||^2), per row.

    The delta rule's per-token transition is (I - eta*z(x)z^T); stable
    iteration needs eta*||z||^2 below ~2, same condition as gradient descent
    diverging when step_size*curvature is too large. A fixed eta (tuned for
    `scan`, which has no such term) is nowhere close to safe once z has any
    outlier-scale dims — real trained models produce exactly that. This
    keeps the effective step size bounded regardless of z's scale.
    """
    return eta / (1.0 + z.pow(2).sum(dim=-1, keepdim=True))


def delta_scan(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    eta: float,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
    decay: float = 1.0,
    truncate_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token online gradient descent: dS_t = (v_t - eta*S_t@z_t) (x) z_t.

    `scan`'s ablation companion (Sun et al. 2024's actual inner-loop update,
    vs. `scan`'s Hebbian write with no error term): the residual is against
    `read_state`, the same (possibly clipped) state that produced `pred`. No
    chunk-parallel closed form exists for this recurrence, so it's O(N)
    Python-level steps, not O(N/chunk_size). Three things this recurrence
    needs that `scan` doesn't, each covered where it's used: the clip feeds
    back into the accumulator (`frobenius_clip`, else an unclipped `state`
    reaches inf and clip turns that into nan); `pred` uses `_adaptive_eta`,
    not raw `eta` (`_adaptive_eta`); and `state` is detached every
    `truncate_every` tokens, bounding how much backward chain through
    `frobenius_clip` can compound — 1 is safest but starves anything that
    only shapes `v` (e.g. the target projection) of gradient entirely, since
    that's the only path it has to the loss. Forward values never depend on
    `truncate_every`, only what backward can reach through the recurrence.
    """
    if z.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"z and v must agree on [B, N], got {tuple(z.shape)} and {tuple(v.shape)}"
        )
    if truncate_every < 1:
        raise ValueError(f"truncate_every must be >= 1, got {truncate_every}")
    b, n, d_ff = z.shape
    d_model = v.shape[-1]
    state = (
        carried.clone().to(z.dtype) if carried is not None else z.new_zeros(b, d_model, d_ff)
    )
    item_delta = torch.zeros_like(state)
    outputs = z.new_empty(b, n, d_model)

    for t in range(n):
        read_state = state
        if clip_tau is not None:
            read_state = frobenius_clip(read_state.unsqueeze(1), eta=eta, tau=clip_tau).squeeze(1)
        pred = _adaptive_eta(z[:, t], eta) * torch.einsum("bf,bdf->bd", z[:, t], read_state)
        outputs[:, t] = pred
        delta = torch.einsum("bd,bf->bdf", v[:, t] - pred, z[:, t])
        item_delta = item_delta + delta
        state = decay * state + delta
        if clip_tau is not None:
            state = frobenius_clip(state.unsqueeze(1), eta=eta, tau=clip_tau).squeeze(1)
        if (t + 1) % truncate_every == 0:
            state = state.detach()

    return outputs, item_delta


def chunked_delta_scan(
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int,
    eta: float,
    normalize_delta_by_chunk: bool = True,
    carried: torch.Tensor | None = None,
    clip_tau: float | None = None,
    decay: float = 1.0,
    truncate_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mini-batch gradient descent: the state is frozen at each chunk's start.

    `delta_scan`'s cheaper sibling: Sun et al. 2024's own parallel/dual-form
    approximation to per-token GD, not a further simplification of it — every
    token in a chunk computes its residual against the same chunk-start state,
    so the chunk's write is one batched matmul, same as `scan`. Sequential
    only across the ~N/chunk_size chunks, not across all N tokens. Same
    per-chunk causality as `scan` and clip-affects-item_delta convention as
    `delta_scan` (the residual is against what chunk actually saw). Also like
    `delta_scan`, the clip feeds back into the accumulator itself,
    `truncate_every` (here counting chunks, not tokens) bounds the same
    backward-chain instability the same way, and `pred` uses `_adaptive_eta`
    per chunk-row rather than the raw `eta` — see `delta_scan`'s docstring
    for all three.
    """
    if z.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"z and v must agree on [B, N], got {tuple(z.shape)} and {tuple(v.shape)}"
        )
    if truncate_every < 1:
        raise ValueError(f"truncate_every must be >= 1, got {truncate_every}")
    n_tokens = z.shape[1]
    counts = chunk_token_counts(n_tokens, chunk_size)
    z_chunks = to_chunks(z, chunk_size)
    v_chunks = to_chunks(v, chunk_size)
    b, k, c, d_ff = z_chunks.shape
    d_model = v.shape[-1]

    state = (
        carried.clone().to(z.dtype) if carried is not None else z.new_zeros(b, d_model, d_ff)
    )
    item_delta = torch.zeros_like(state)
    out_chunks = z.new_empty(b, k, c, d_model)

    for idx in range(k):
        read_state = state
        if clip_tau is not None:
            read_state = frobenius_clip(read_state.unsqueeze(1), eta=eta, tau=clip_tau).squeeze(1)
        z_k, v_k = z_chunks[:, idx], v_chunks[:, idx]
        pred = _adaptive_eta(z_k, eta) * torch.einsum("bcf,bdf->bcd", z_k, read_state)
        out_chunks[:, idx] = pred
        delta = torch.einsum("bcd,bcf->bdf", v_k - pred, z_k)
        if normalize_delta_by_chunk:
            delta = delta / max(counts[idx], 1)
        item_delta = item_delta + delta
        state = decay * state + delta
        if clip_tau is not None:
            state = frobenius_clip(state.unsqueeze(1), eta=eta, tau=clip_tau).squeeze(1)
        if (idx + 1) % truncate_every == 0:
            state = state.detach()

    out = out_chunks.reshape(b, k * c, d_model)[:, :n_tokens, :]
    return out, item_delta


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
