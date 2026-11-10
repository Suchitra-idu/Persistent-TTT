"""The scan kernels against a sequential apply-then-update oracle."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _builders as build
from tests.core._oracles import sequential_chunked_delta_rule, sequential_scan
from ttt.core import ttt_math

ETA = 0.05
TOL = 1e-12


@given(
    n_tokens=st.integers(min_value=0, max_value=500),
    chunk_size=st.integers(min_value=1, max_value=64),
)
def test_chunk_token_counts_sum_to_the_token_count(n_tokens, chunk_size):
    counts = ttt_math.chunk_token_counts(n_tokens, chunk_size)

    assert sum(counts) == n_tokens


@given(
    n_tokens=st.integers(min_value=1, max_value=500),
    chunk_size=st.integers(min_value=1, max_value=64),
)
def test_only_the_last_chunk_may_be_short(n_tokens, chunk_size):
    counts = ttt_math.chunk_token_counts(n_tokens, chunk_size)

    assert all(c == chunk_size for c in counts[:-1])


@given(
    n_tokens=st.integers(min_value=1, max_value=500),
    chunk_size=st.integers(min_value=1, max_value=64),
)
def test_every_chunk_holds_at_least_one_token(n_tokens, chunk_size):
    counts = ttt_math.chunk_token_counts(n_tokens, chunk_size)

    assert all(1 <= c <= chunk_size for c in counts)


def test_to_chunks_zero_pads_only_the_tail():
    x = torch.arange(1, 11, dtype=torch.float64).view(1, 10, 1)

    chunked = ttt_math.to_chunks(x, 4)

    assert chunked.shape == (1, 3, 4, 1)
    assert chunked.flatten()[:10].tolist() == x.flatten().tolist()
    assert chunked.flatten()[10:].tolist() == [0.0, 0.0]


def test_exclusive_cumsum_gives_the_first_chunk_no_state():
    deltas = build.randn(1, 4, 3, 3, seed=7)

    cum = ttt_math.exclusive_cumsum(deltas)

    assert torch.equal(cum[:, 0], torch.zeros_like(cum[:, 0]))


@given(n_chunks=st.integers(min_value=1, max_value=12))
@settings(max_examples=25, deadline=None)
def test_exclusive_cumsum_sums_strictly_earlier_chunks(n_chunks):
    deltas = build.randn(1, n_chunks, 2, 2, seed=n_chunks)

    cum = ttt_math.exclusive_cumsum(deltas)

    expected = torch.stack(
        [deltas[0, :k].sum(dim=0) for k in range(n_chunks)], dim=0
    )
    assert torch.allclose(cum[0], expected, atol=TOL)


@given(scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
@settings(max_examples=25, deadline=None)
def test_clip_bounds_the_scaled_norm_at_tau(scale):
    state = build.randn(1, 1, 4, 4, seed=3) * scale

    clipped = ttt_math.frobenius_clip(state, eta=ETA, tau=5.0)

    assert float((ETA * clipped).norm(p="fro")) <= 5.0 + 1e-9


def test_clip_leaves_a_state_already_under_tau_untouched():
    state = build.randn(1, 1, 4, 4, seed=11) * 1e-6

    clipped = ttt_math.frobenius_clip(state, eta=ETA, tau=5.0)

    assert torch.allclose(clipped, state, atol=TOL)


def test_clip_preserves_direction():
    state = build.randn(1, 1, 4, 4, seed=13) * 1e6

    clipped = ttt_math.frobenius_clip(state, eta=ETA, tau=5.0)

    cosine = torch.nn.functional.cosine_similarity(
        clipped.flatten(), state.flatten(), dim=0
    )
    assert float(cosine) == pytest.approx(1.0, abs=1e-9)


@given(
    n_tokens=st.integers(min_value=1, max_value=40),
    chunk_size=st.integers(min_value=1, max_value=12),
)
@settings(max_examples=30, deadline=None)
def test_scan_matches_the_sequential_oracle(n_tokens, chunk_size):
    z, v = build.scan_inputs(seed=n_tokens * 100 + chunk_size, n_tokens=n_tokens)

    out, item_delta = ttt_math.scan(
        z, v, chunk_size=chunk_size, eta=ETA, normalize_delta_by_chunk=True
    )

    ref_out, ref_delta = sequential_scan(
        z[0], v[0], chunk_size=chunk_size, eta=ETA, normalize=True
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


def test_scan_matches_the_oracle_when_starting_from_a_carry():
    z, v = build.scan_inputs(seed=21, n_tokens=17)
    carried = build.randn(1, 6, 8, seed=99)

    out, _ = ttt_math.scan(
        z, v, chunk_size=5, eta=ETA, normalize_delta_by_chunk=True, carried=carried
    )

    ref_out, _ = sequential_scan(
        z[0], v[0], chunk_size=5, eta=ETA, normalize=True, carried=carried[0]
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_scan_matches_the_oracle_with_the_clip_engaged():
    z, v = build.scan_inputs(seed=33, n_tokens=23)
    carried = build.randn(1, 6, 8, seed=34) * 1e4

    out, _ = ttt_math.scan(
        z,
        v,
        chunk_size=4,
        eta=ETA,
        normalize_delta_by_chunk=True,
        carried=carried,
        clip_tau=5.0,
    )

    ref_out, _ = sequential_scan(
        z[0],
        v[0],
        chunk_size=4,
        eta=ETA,
        normalize=True,
        carried=carried[0],
        clip_tau=5.0,
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_scan_matches_the_oracle_without_chunk_normalisation():
    z, v = build.scan_inputs(seed=41, n_tokens=19)

    out, item_delta = ttt_math.scan(
        z, v, chunk_size=6, eta=ETA, normalize_delta_by_chunk=False
    )

    ref_out, ref_delta = sequential_scan(
        z[0], v[0], chunk_size=6, eta=ETA, normalize=False
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


def test_a_single_chunk_from_zero_produces_no_ttt_term():
    z, v = build.scan_inputs(seed=51, n_tokens=8)

    out, _ = ttt_math.scan(z, v, chunk_size=8, eta=ETA)

    assert torch.equal(out, torch.zeros_like(out))


def test_the_item_delta_is_unclipped():
    """What crosses an item boundary is raw; the EMA is what bounds it."""
    z, v = build.scan_inputs(seed=61, n_tokens=12)
    carried = build.randn(1, 6, 8, seed=62) * 1e6

    _, clipped_run = ttt_math.scan(
        z, v, chunk_size=4, eta=ETA, carried=carried, clip_tau=5.0
    )
    _, unclipped_run = ttt_math.scan(z, v, chunk_size=4, eta=ETA, carried=carried)

    assert torch.allclose(clipped_run, unclipped_run, atol=TOL)


def test_scan_rejects_mismatched_z_and_v():
    z, _ = build.scan_inputs(seed=71, n_tokens=10)
    _, v = build.scan_inputs(seed=72, n_tokens=11)

    with pytest.raises(ValueError, match=r"must agree on \[B, N\]"):
        ttt_math.scan(z, v, chunk_size=4, eta=ETA)


@given(
    n_tokens=st.integers(min_value=1, max_value=40),
    chunk_size=st.integers(min_value=1, max_value=12),
)
@settings(max_examples=30, deadline=None)
def test_chunked_delta_scan_matches_the_sequential_oracle(n_tokens, chunk_size):
    z, v = build.scan_inputs(seed=n_tokens * 100 + chunk_size + 3, n_tokens=n_tokens)

    out, item_delta = ttt_math.chunked_delta_scan(
        z, v, chunk_size=chunk_size, eta=ETA, normalize_delta_by_chunk=True
    )

    ref_out, ref_delta = sequential_chunked_delta_rule(
        z[0], v[0], chunk_size=chunk_size, eta=ETA, normalize=True
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


def test_chunked_delta_scan_matches_the_oracle_when_starting_from_a_carry():
    z, v = build.scan_inputs(seed=311, n_tokens=17)
    carried = build.randn(1, 6, 8, seed=312)

    out, _ = ttt_math.chunked_delta_scan(z, v, chunk_size=5, eta=ETA, carried=carried)

    ref_out, _ = sequential_chunked_delta_rule(
        z[0], v[0], chunk_size=5, eta=ETA, carried=carried[0]
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_chunked_delta_scan_matches_the_oracle_with_the_clip_engaged():
    z, v = build.scan_inputs(seed=321, n_tokens=23)
    carried = build.randn(1, 6, 8, seed=322) * 1e4

    out, _ = ttt_math.chunked_delta_scan(
        z, v, chunk_size=4, eta=ETA, carried=carried, clip_tau=5.0
    )

    ref_out, _ = sequential_chunked_delta_rule(
        z[0], v[0], chunk_size=4, eta=ETA, carried=carried[0], clip_tau=5.0
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_a_single_chunk_chunked_delta_scan_from_zero_produces_no_ttt_term():
    """Unlike `delta_scan`, causality here is per-chunk like `scan`: a lone
    chunk's state is frozen at zero for the whole chunk, not written token
    by token within it."""
    z, v = build.scan_inputs(seed=331, n_tokens=8)

    out, _ = ttt_math.chunked_delta_scan(z, v, chunk_size=8, eta=ETA)

    assert torch.equal(out, torch.zeros_like(out))


def test_the_chunked_delta_scan_item_delta_reflects_the_clip():
    z, v = build.scan_inputs(seed=341, n_tokens=12)
    carried = build.randn(1, 6, 8, seed=342) * 1e6

    _, clipped_run = ttt_math.chunked_delta_scan(
        z, v, chunk_size=4, eta=ETA, carried=carried, clip_tau=5.0
    )
    _, unclipped_run = ttt_math.chunked_delta_scan(
        z, v, chunk_size=4, eta=ETA, carried=carried
    )

    assert not torch.allclose(clipped_run, unclipped_run, atol=1e-6)


def test_chunked_delta_scan_rejects_mismatched_z_and_v():
    z, _ = build.scan_inputs(seed=351, n_tokens=10)
    _, v = build.scan_inputs(seed=352, n_tokens=11)

    with pytest.raises(ValueError, match=r"must agree on \[B, N\]"):
        ttt_math.chunked_delta_scan(z, v, chunk_size=4, eta=ETA)


def _outlier_inputs(*, n_tokens: int, d_model: int, d_ff: int, seed: int):
    """A few extreme-magnitude dims, like the activation outliers real
    trained LLMs are known to produce — plain N(0,1) never triggers the
    accumulator divergence this is a regression test for."""
    z = build.randn(1, n_tokens, d_ff, seed=seed)
    z[:, :, :3] *= 300
    v = build.randn(1, n_tokens, d_model, seed=seed + 1) * 5
    return z, v


def _unclipped_chunked_accumulator(z, v, *, chunk_size, eta, clip_tau):
    """The pre-fix recurrence: clip bounds `read_state` but never feeds back
    into `state`, so it can grow without bound. Reference for the regression
    test below — not something any kernel should do anymore."""
    counts = ttt_math.chunk_token_counts(z.shape[1], chunk_size)
    z_chunks = ttt_math.to_chunks(z, chunk_size)
    v_chunks = ttt_math.to_chunks(v, chunk_size)
    state = torch.zeros(z.shape[0], v.shape[-1], z.shape[-1], dtype=z.dtype)
    for idx in range(z_chunks.shape[1]):
        read_state = ttt_math.frobenius_clip(
            state.unsqueeze(1), eta=eta, tau=clip_tau
        ).squeeze(1)
        pred = eta * torch.einsum("bcf,bdf->bcd", z_chunks[:, idx], read_state)
        delta = torch.einsum(
            "bcd,bcf->bdf", v_chunks[:, idx] - pred, z_chunks[:, idx]
        ) / max(counts[idx], 1)
        state = state + delta
    return state


@pytest.mark.parametrize(
    "kernel",
    [
        lambda z, v: ttt_math.chunked_delta_scan(z, v, chunk_size=50, eta=0.07, clip_tau=5.0),
    ],
    ids=["chunked_delta_scan"],
)
def test_the_clip_keeps_the_accumulator_bounded_under_outlier_activations(kernel):
    """The regression this pins: before the clip fed back into the raw
    accumulator, growth from outlier dims (over enough chunks, real trained
    models produce exactly this) was unbounded — large enough on the real
    model to reach inf, and frobenius_clip turns that into nan (tau/inf=0,
    inf*0=nan): a finite loss with a nan gradient, not a nan forward pass.
    Asserting boundedness rather than finiteness because the unclipped
    reference here is merely huge, not literally inf, at this scale/dtype —
    the point is the two orders of magnitude the fix actually buys."""
    z, v = _outlier_inputs(n_tokens=3000, d_model=32, d_ff=96, seed=701)

    out, item_delta = kernel(z, v)
    unclipped = _unclipped_chunked_accumulator(
        z, v, chunk_size=50, eta=0.07, clip_tau=5.0
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(item_delta).all()
    assert out.abs().max() < unclipped.abs().max() / 10


def test_chunked_delta_scan_gradient_does_not_cross_the_chunk_boundary():
    z, v = build.scan_inputs(seed=712, n_tokens=20)
    z.requires_grad_(True)
    v.requires_grad_(True)

    out, _ = ttt_math.chunked_delta_scan(z, v, chunk_size=5, eta=ETA, clip_tau=5.0)
    out[:, -5:].sum().backward()

    assert torch.equal(z.grad[:, :-5], torch.zeros_like(z.grad[:, :-5]))
    assert z.grad[:, -5:].abs().sum() > 0
    assert v.grad is None


def test_chunked_delta_scan_gradient_reaches_earlier_chunks_within_the_window():
    """truncate_every>1 is exactly the fix for what full truncation breaks: v
    only ever shapes a *later* chunk's prediction, never its own chunk's, so
    with truncate_every=1 nothing downstream of v gets gradient at all — the
    real bug this pins (w_target/target_conv stuck at zero-init forever)."""
    z, v = build.scan_inputs(seed=714, n_tokens=30)
    z.requires_grad_(True)
    v.requires_grad_(True)

    out, _ = ttt_math.chunked_delta_scan(
        z, v, chunk_size=5, eta=ETA, clip_tau=5.0, truncate_every=3
    )
    out[:, -5:].sum().backward()

    # The window covering the last chunk is chunks 3,4,5 (tokens 15-29);
    # chunks 0,1,2 (tokens 0-14) are a prior, detached window.
    assert torch.equal(z.grad[:, :15], torch.zeros_like(z.grad[:, :15]))
    assert v.grad is not None
    assert torch.equal(v.grad[:, :15], torch.zeros_like(v.grad[:, :15]))
    # v in chunks 3,4 (tokens 15-24) shapes chunk 5's prediction; v in chunk
    # 5 itself (tokens 25-29) never does, same as every chunk's own v.
    assert v.grad[:, 15:25].abs().sum() > 0
    assert torch.equal(v.grad[:, 25:], torch.zeros_like(v.grad[:, 25:]))


