from __future__ import annotations

import math

import pytest

from tests.app import _builders
from ttt.app import token_view

PRIME = (10, 11, 12)
TARGET = (100, 101, 102, 103)


def measure(losses=(1.0,)):
    wiring = _builders.Wiring(losses=losses)
    view = token_view.measure(prime=PRIME, target=TARGET, compute=wiring.compute, fast_weights=wiring.fast_weights)
    return view, wiring


def test_the_reset_target_is_scored_before_the_prime_is_read():
    _, wiring = measure()

    assert wiring.compute.eval_forwards == [TARGET, PRIME, TARGET]


def test_the_prime_advances_the_carry_once():
    _, wiring = measure()

    assert wiring.fast_weights.events.count("advance_carry") == 1


def test_the_carry_is_left_reset():
    _, wiring = measure()

    assert wiring.fast_weights.events[-1] == "reset_carry"


def test_each_arm_has_one_loss_per_scored_target_token():
    view, _ = measure()

    assert len(view.reset) == len(view.carry) == len(TARGET) - 1


def test_gain_is_reset_loss_minus_carry_loss():
    view = token_view.TokenView(reset=(2.0, 1.0), carry=(1.5, 1.25))

    assert view.gains == (0.5, -0.25)


def test_summary_reports_the_perplexity_change_in_percent():
    view = token_view.TokenView(reset=(1.0, 1.0), carry=(0.9, 0.9))

    assert view.summary()["ppl_change_pct"] == pytest.approx(100 * (1 - math.exp(-0.1)))


def test_an_empty_prime_is_refused():
    wiring = _builders.Wiring(losses=(1.0,))
    with pytest.raises(ValueError):
        token_view.measure(prime=(), target=TARGET, compute=wiring.compute, fast_weights=wiring.fast_weights)
