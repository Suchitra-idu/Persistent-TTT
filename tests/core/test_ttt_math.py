"""The scan kernels against a sequential apply-then-update oracle."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _builders as build
from tests.core._oracles import (
    sequential_chunked_delta_rule,
    sequential_delta_rule,
    sequential_scan,
)
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


def test_exclusive_decayed_cumsum_at_decay_one_matches_exclusive_cumsum():
    deltas = build.randn(1, 5, 3, 3, seed=101)

    assert torch.equal(
        ttt_math.exclusive_decayed_cumsum(deltas, decay=1.0),
        ttt_math.exclusive_cumsum(deltas),
    )


def test_exclusive_decayed_cumsum_at_decay_zero_keeps_only_the_prior_chunk():
    deltas = build.randn(1, 4, 2, 2, seed=102)

    cum = ttt_math.exclusive_decayed_cumsum(deltas, decay=0.0)

    assert torch.equal(cum[:, 0], torch.zeros_like(cum[:, 0]))
    for k in range(1, 4):
        assert torch.allclose(cum[:, k], deltas[:, k - 1], atol=TOL)


def test_exclusive_decayed_cumsum_matches_the_recurrence_by_hand():
    deltas = build.randn(1, 3, 2, 2, seed=103)
    decay = 0.5

    cum = ttt_math.exclusive_decayed_cumsum(deltas, decay=decay)

    s1 = deltas[:, 0]
    s2 = decay * s1 + deltas[:, 1]
    assert torch.allclose(cum[:, 1], s1, atol=TOL)
    assert torch.allclose(cum[:, 2], s2, atol=TOL)


def test_exclusive_decayed_cumsum_rejects_a_decay_outside_unit_range():
    deltas = build.randn(1, 2, 2, 2, seed=104)

    with pytest.raises(ValueError, match="decay must be in"):
        ttt_math.exclusive_decayed_cumsum(deltas, decay=1.5)


def test_exclusive_decayed_cumsum_stays_finite_over_a_long_document():
    """A long enough undecayed sum is exactly what made clip_tau load-bearing
    (repeat_carry_eval_v1 at clip_tau=10); decay<1 is the structural fix."""
    deltas = build.randn(1, 2000, 4, 4, seed=105)

    cum = ttt_math.exclusive_decayed_cumsum(deltas, decay=0.9)

    assert torch.isfinite(cum).all()


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


@given(
    n_tokens=st.integers(min_value=1, max_value=40),
    chunk_size=st.integers(min_value=1, max_value=12),
    decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_scan_matches_the_sequential_oracle_with_intra_session_decay(
    n_tokens, chunk_size, decay
):
    z, v = build.scan_inputs(seed=n_tokens * 100 + chunk_size, n_tokens=n_tokens)

    out, item_delta = ttt_math.scan(
        z, v, chunk_size=chunk_size, eta=ETA, normalize_delta_by_chunk=True, decay=decay
    )

    ref_out, ref_delta = sequential_scan(
        z[0], v[0], chunk_size=chunk_size, eta=ETA, normalize=True, decay=decay
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


def test_a_lower_chunk_decay_holds_a_long_documents_output_down():
    """The point of the knob: a long document's late-chunk output stays
    bounded instead of growing with its chunk count. `item_delta` (the
    cross-item carry) is unaffected — this is the intra-session term only."""
    z, v = build.scan_inputs(seed=201, n_tokens=400)

    undamped_out, undamped_delta = ttt_math.scan(
        z, v, chunk_size=5, eta=ETA, normalize_delta_by_chunk=True, decay=1.0
    )
    damped_out, damped_delta = ttt_math.scan(
        z, v, chunk_size=5, eta=ETA, normalize_delta_by_chunk=True, decay=0.9
    )

    assert float(damped_out[:, -1].norm()) < float(undamped_out[:, -1].norm())
    assert torch.equal(damped_delta, undamped_delta)


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


@given(n_tokens=st.integers(min_value=1, max_value=40))
@settings(max_examples=30, deadline=None)
def test_delta_scan_matches_the_sequential_oracle(n_tokens):
    z, v = build.scan_inputs(seed=n_tokens * 100 + 7, n_tokens=n_tokens)

    out, item_delta = ttt_math.delta_scan(z, v, eta=ETA)

    ref_out, ref_delta = sequential_delta_rule(z[0], v[0], eta=ETA)
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


@given(
    n_tokens=st.integers(min_value=1, max_value=40),
    decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_delta_scan_matches_the_sequential_oracle_with_decay(n_tokens, decay):
    z, v = build.scan_inputs(seed=n_tokens * 100 + 11, n_tokens=n_tokens)

    out, item_delta = ttt_math.delta_scan(z, v, eta=ETA, decay=decay)

    ref_out, ref_delta = sequential_delta_rule(z[0], v[0], eta=ETA, decay=decay)
    assert torch.allclose(out[0], ref_out, atol=TOL)
    assert torch.allclose(item_delta[0], ref_delta, atol=TOL)


def test_delta_scan_matches_the_oracle_when_starting_from_a_carry():
    z, v = build.scan_inputs(seed=211, n_tokens=17)
    carried = build.randn(1, 6, 8, seed=212)

    out, _ = ttt_math.delta_scan(z, v, eta=ETA, carried=carried)

    ref_out, _ = sequential_delta_rule(z[0], v[0], eta=ETA, carried=carried[0])
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_delta_scan_matches_the_oracle_with_the_clip_engaged():
    z, v = build.scan_inputs(seed=221, n_tokens=23)
    carried = build.randn(1, 6, 8, seed=222) * 1e4

    out, _ = ttt_math.delta_scan(z, v, eta=ETA, carried=carried, clip_tau=5.0)

    ref_out, _ = sequential_delta_rule(
        z[0], v[0], eta=ETA, carried=carried[0], clip_tau=5.0
    )
    assert torch.allclose(out[0], ref_out, atol=TOL)


def test_delta_scan_is_causal_the_first_token_from_zero_produces_no_output():
    z, v = build.scan_inputs(seed=231, n_tokens=9)

    out, _ = ttt_math.delta_scan(z, v, eta=ETA)

    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))


def test_delta_scan_is_causal_later_tokens_do_produce_output():
    """Unlike `scan`, causality here is per-token, not per-chunk: token 1
    already sees a state written by token 0 — nothing needs a second chunk."""
    z, v = build.scan_inputs(seed=232, n_tokens=9)

    out, _ = ttt_math.delta_scan(z, v, eta=ETA)

    assert not torch.equal(out[:, 1:], torch.zeros_like(out[:, 1:]))


def test_the_delta_scan_item_delta_reflects_the_clip_unlike_scans():
    """Unlike `scan`, `item_delta` here is not clip-independent: the residual
    is against what was actually predicted, so clipping the read state
    changes the learning signal itself, not just the output."""
    z, v = build.scan_inputs(seed=241, n_tokens=12)
    carried = build.randn(1, 6, 8, seed=242) * 1e6

    _, clipped_run = ttt_math.delta_scan(z, v, eta=ETA, carried=carried, clip_tau=5.0)
    _, unclipped_run = ttt_math.delta_scan(z, v, eta=ETA, carried=carried)

    assert not torch.allclose(clipped_run, unclipped_run, atol=1e-6)


def test_delta_scan_rejects_mismatched_z_and_v():
    z, _ = build.scan_inputs(seed=251, n_tokens=10)
    _, v = build.scan_inputs(seed=252, n_tokens=11)

    with pytest.raises(ValueError, match=r"must agree on \[B, N\]"):
        ttt_math.delta_scan(z, v, eta=ETA)


def test_delta_scan_differs_from_the_hebbian_scan():
    """The whole point: the residual term makes this a different recurrence,
    not just a relabelling of `scan`."""
    z, v = build.scan_inputs(seed=261, n_tokens=30)

    _, delta_item = ttt_math.delta_scan(z, v, eta=ETA)
    _, hebbian_item = ttt_math.scan(z, v, chunk_size=5, eta=ETA)

    assert not torch.allclose(delta_item, hebbian_item, atol=1e-6)


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


@given(
    n_tokens=st.integers(min_value=1, max_value=40),
    chunk_size=st.integers(min_value=1, max_value=12),
    decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_chunked_delta_scan_matches_the_sequential_oracle_with_decay(
    n_tokens, chunk_size, decay
):
    z, v = build.scan_inputs(seed=n_tokens * 100 + chunk_size + 5, n_tokens=n_tokens)

    out, item_delta = ttt_math.chunked_delta_scan(
        z, v, chunk_size=chunk_size, eta=ETA, decay=decay
    )

    ref_out, ref_delta = sequential_chunked_delta_rule(
        z[0], v[0], chunk_size=chunk_size, eta=ETA, decay=decay
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


def test_chunked_delta_scan_differs_from_both_scan_and_delta_scan():
    z, v = build.scan_inputs(seed=361, n_tokens=30)

    _, chunked_item = ttt_math.chunked_delta_scan(z, v, chunk_size=5, eta=ETA)
    _, hebbian_item = ttt_math.scan(z, v, chunk_size=5, eta=ETA)
    _, token_item = ttt_math.delta_scan(z, v, eta=ETA)

    assert not torch.allclose(chunked_item, hebbian_item, atol=1e-6)
    assert not torch.allclose(chunked_item, token_item, atol=1e-6)


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
        lambda z, v: ttt_math.chunked_delta_scan(
            z, v, chunk_size=50, eta=0.07, clip_tau=5.0, decay=1.0
        ),
        lambda z, v: ttt_math.delta_scan(z, v, eta=0.07, clip_tau=5.0, decay=1.0),
    ],
    ids=["chunked_delta_scan", "delta_scan"],
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


def test_delta_scan_gradient_does_not_cross_the_token_boundary():
    """The other half of the outlier-activation fix: `read_state` is detached
    each token, so backward from the last token's output must not reach any
    earlier token's z — the recurrence itself, chained across every token,
    is what compounded into instability, not any single non-finite value.
    `v` never gets a grad at all here: it only reaches `state`, which is
    read back only through the now-detached copy, never `out` directly."""
    z, v = build.scan_inputs(seed=711, n_tokens=20)
    z.requires_grad_(True)
    v.requires_grad_(True)

    out, _ = ttt_math.delta_scan(z, v, eta=ETA, clip_tau=5.0)
    out[:, -1].sum().backward()

    assert torch.equal(z.grad[:, :-1], torch.zeros_like(z.grad[:, :-1]))
    assert z.grad[:, -1].abs().sum() > 0
    assert v.grad is None


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


def test_delta_scan_item_delta_still_carries_gradient_from_every_token():
    """The detach only truncates the intra-item state recurrence `out` reads
    from; `item_delta` — what the session carry trains on — still sees every
    token's z and v, undiminished."""
    z, v = build.scan_inputs(seed=713, n_tokens=20)
    z.requires_grad_(True)
    v.requires_grad_(True)

    _, item_delta = ttt_math.delta_scan(z, v, eta=ETA, clip_tau=5.0)
    item_delta.sum().backward()

    assert (z.grad.abs().sum(dim=-1) > 0).all()
    assert (v.grad.abs().sum(dim=-1) > 0).all()


def test_stream_chunk_delta_matches_the_scan_delta_on_a_full_chunk():
    z, v = build.scan_inputs(seed=81, n_tokens=6)

    streamed = ttt_math.stream_chunk_delta(v, z, chunk_size=6)
    _, scanned = ttt_math.scan(z, v, chunk_size=6, eta=ETA)

    assert torch.allclose(streamed, scanned, atol=TOL)


def test_a_short_stream_chunk_still_divides_by_the_configured_chunk_size():
    """A stream commits only when the buffer is full, so a short buffer is
    pending rather than short — the divisor is the configured size either way."""
    z, v = build.scan_inputs(seed=83, n_tokens=3)

    delta = ttt_math.stream_chunk_delta(v, z, chunk_size=6)

    assert torch.allclose(
        delta, ttt_math.stream_chunk_delta(v, z, chunk_size=6, normalize=False) / 6,
        atol=TOL,
    )


def test_stream_apply_matches_the_scan_term_for_one_state():
    z, _ = build.scan_inputs(seed=91, n_tokens=5)
    state = build.randn(6, 8, seed=92)

    streamed = ttt_math.stream_apply(z[0], state, eta=ETA)

    expected = ETA * (z[0] @ state.transpose(0, 1))
    assert torch.allclose(streamed, expected, atol=TOL)
