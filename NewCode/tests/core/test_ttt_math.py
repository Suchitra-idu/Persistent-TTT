"""The scan kernels against a sequential apply-then-update oracle."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _builders as build
from tests.core._oracles import sequential_scan
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


def test_stream_chunk_delta_matches_the_scan_delta_on_a_full_chunk():
    z, v = build.scan_inputs(seed=81, n_tokens=6)

    streamed = ttt_math.stream_chunk_delta(v, z, chunk_size=6)
    _, scanned = ttt_math.scan(z, v, chunk_size=6, eta=ETA)

    assert torch.allclose(streamed, scanned, atol=TOL)


def test_stream_apply_matches_the_scan_term_for_one_state():
    z, _ = build.scan_inputs(seed=91, n_tokens=5)
    state = build.randn(6, 8, seed=92)

    streamed = ttt_math.stream_apply(z[0], state, eta=ETA)

    expected = ETA * (z[0] @ state.transpose(0, 1))
    assert torch.allclose(streamed, expected, atol=TOL)
