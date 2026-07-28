"""The LR curve and the step count that sizes it."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ttt.core import lr_schedule


@given(
    items=st.integers(min_value=0, max_value=100_000),
    accum=st.integers(min_value=1, max_value=64),
    epochs=st.integers(min_value=0, max_value=10),
)
def test_step_count_rounds_a_partial_accumulation_window_up(items, accum, epochs):
    steps = lr_schedule.total_optimizer_steps(items, accum, epochs)

    assert steps == math.ceil(items / accum) * epochs


def test_a_partial_final_window_still_steps():
    assert lr_schedule.total_optimizer_steps(17, 16, 1) == 2


def test_warmup_has_a_floor_so_short_runs_still_warm_up():
    assert lr_schedule.warmup_steps(100, 0.02, 10) == 10


def test_warmup_follows_the_ratio_once_the_run_is_long_enough():
    assert lr_schedule.warmup_steps(10_000, 0.02, 10) == 200


@given(
    total=st.integers(min_value=1, max_value=5000),
    warmup=st.integers(min_value=0, max_value=500),
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_the_multiplier_stays_within_the_unit_interval(total, warmup, fraction):
    step = int(fraction * total)

    assert 0.0 <= lr_schedule.multiplier(
        step, num_warmup_steps=warmup, num_training_steps=total
    ) <= 1.0


def test_the_multiplier_starts_at_zero_when_there_is_a_warmup():
    assert lr_schedule.multiplier(0, num_warmup_steps=10, num_training_steps=100) == 0.0


def test_the_multiplier_peaks_at_the_end_of_the_warmup():
    assert lr_schedule.multiplier(
        10, num_warmup_steps=10, num_training_steps=100
    ) == pytest.approx(1.0)


def test_warmup_is_linear():
    ramp = [
        lr_schedule.multiplier(s, num_warmup_steps=10, num_training_steps=100)
        for s in range(11)
    ]

    assert ramp == pytest.approx([s / 10 for s in range(11)])


def test_the_multiplier_decays_to_zero_at_the_final_step():
    assert lr_schedule.multiplier(
        100, num_warmup_steps=10, num_training_steps=100
    ) == pytest.approx(0.0, abs=1e-12)


def test_the_cosine_half_way_through_the_decay_is_a_half():
    assert lr_schedule.multiplier(
        55, num_warmup_steps=10, num_training_steps=100
    ) == pytest.approx(0.5)


@given(step=st.integers(min_value=10, max_value=99))
def test_the_decay_is_monotone(step):
    later = lr_schedule.multiplier(
        step + 1, num_warmup_steps=10, num_training_steps=100
    )
    earlier = lr_schedule.multiplier(
        step, num_warmup_steps=10, num_training_steps=100
    )

    assert later <= earlier + 1e-12


def test_stepping_past_the_end_of_the_schedule_is_an_error():
    """HuggingFace's cosine keeps oscillating here and hands back a live LR."""
    with pytest.raises(ValueError, match="past the end of the schedule"):
        lr_schedule.multiplier(101, num_warmup_steps=10, num_training_steps=100)


def test_a_run_that_is_all_warmup_ends_at_zero():
    """No decay window left means the schedule is over, not stuck at 1.0."""
    assert lr_schedule.multiplier(
        10, num_warmup_steps=10, num_training_steps=10
    ) == 0.0
