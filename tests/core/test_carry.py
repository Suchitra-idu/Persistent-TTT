"""The carry recurrence and its magnitude. Settles PLAN §7 defect 1."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.core import _builders as build
from tests.core._oracles import sequential_carry
from ttt.core import carry

ETA = 0.05
TOL = 1e-12


def _deltas(n: int, seed: int) -> list[torch.Tensor]:
    return [build.randn(4, 5, seed=seed + i) for i in range(n)]


@given(
    n_items=st.integers(min_value=1, max_value=15),
    decay=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=40, deadline=None)
def test_advance_matches_the_recurrence_oracle(n_items, decay):
    deltas = _deltas(n_items, seed=1000)

    carried = None
    for delta in deltas:
        carried = carry.advance(carried, delta, decay=decay)

    assert torch.allclose(carried, sequential_carry(deltas, decay=decay), atol=TOL)


def test_decay_of_one_is_a_pure_sum():
    """The assumption the old, stale oracle encoded — true only at decay=1."""
    deltas = _deltas(6, seed=2000)

    carried = None
    for delta in deltas:
        carried = carry.advance(carried, delta, decay=1.0)

    assert torch.allclose(carried, torch.stack(deltas).sum(dim=0), atol=TOL)


def test_decay_below_one_is_not_a_pure_sum():
    """The disagreement that made defect 1 invisible, pinned as a test."""
    deltas = _deltas(6, seed=2000)

    ema = None
    total = None
    for delta in deltas:
        ema = carry.advance(ema, delta, decay=0.9)
        total = carry.advance(total, delta, decay=1.0)

    assert not torch.allclose(ema, total, atol=1e-6)


def test_decay_of_zero_keeps_only_the_last_item():
    deltas = _deltas(4, seed=3000)

    carried = None
    for delta in deltas:
        carried = carry.advance(carried, delta, decay=0.0)

    assert torch.allclose(carried, deltas[-1], atol=TOL)


def test_the_first_item_starts_the_carry_regardless_of_decay():
    delta = build.randn(4, 5, seed=4000)

    assert torch.allclose(carry.advance(None, delta, decay=0.5), delta, atol=TOL)


@given(decay=st.floats(min_value=0.0, max_value=0.95))
@settings(max_examples=20, deadline=None)
def test_a_repeated_delta_converges_to_the_steady_state_scale(decay):
    # Capped at 0.95 — the top of the configured range — so 500 items is
    # deep enough for the truncated tail (decay**500) to be far below the
    # tolerance. At decay -> 1 convergence is arbitrarily slow, which is the
    # whole point of `steady_state_scale` rejecting 1.0.
    delta = torch.ones((2, 2), dtype=torch.float64)

    carried = None
    for _ in range(500):
        carried = carry.advance(carried, delta, decay=decay)

    expected = carry.steady_state_scale(decay)
    assert float(carried[0, 0]) == pytest.approx(expected, rel=1e-6)


def test_a_pure_sum_has_no_steady_state():
    with pytest.raises(ValueError, match="decay in"):
        carry.steady_state_scale(1.0)


@pytest.mark.parametrize("decay", [-0.1, 1.5, 2.0])
def test_advance_rejects_a_decay_outside_the_unit_interval(decay):
    with pytest.raises(ValueError, match=r"carried_decay must be in \[0, 1\]"):
        carry.advance(None, torch.zeros((2, 2)), decay=decay)


def test_no_state_means_a_zero_ratio():
    assert carry.state_ratio(None, w0_norm=3.0, eta=ETA) == 0.0


def test_state_ratio_is_the_scaled_frobenius_ratio():
    state = torch.tensor([[3.0, 4.0]], dtype=torch.float64)  # ‖S‖_F = 5

    ratio = carry.state_ratio(state, w0_norm=2.0, eta=ETA)

    assert ratio == pytest.approx(ETA * 5.0 / 2.0)


@given(scale=st.floats(min_value=1e-3, max_value=1e3))
@settings(max_examples=25, deadline=None)
def test_state_ratio_scales_linearly_with_the_state(scale):
    state = build.randn(3, 3, seed=5000)

    base = carry.state_ratio(state, w0_norm=2.0, eta=ETA)
    scaled = carry.state_ratio(state * scale, w0_norm=2.0, eta=ETA)

    assert scaled == pytest.approx(base * scale, rel=1e-9)


def test_mean_state_ratio_of_no_layers_is_zero():
    assert carry.mean_state_ratio({}) == 0.0


def test_mean_state_ratio_averages_over_layers():
    assert carry.mean_state_ratio({1: 0.2, 3: 0.4, 5: 0.6}) == pytest.approx(0.4)
