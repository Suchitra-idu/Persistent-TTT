from __future__ import annotations

import torch

from tests.app import _builders
from ttt.core.types import Carry
from ttt.app import token_gain
from ttt.app.relatedness_eval import GENERIC, IN_CONTEXT, LEVELS, RELATED, RESET, Case

TARGET = tuple(range(100, 108))


def case(index=0):
    primes = {level: tuple(range(10 * (i + 1), 10 * (i + 1) + 4)) for i, level in enumerate(LEVELS)}
    return Case(
        index=index, target_title="T", target=TARGET, primes=primes,
        prime_titles={level: level for level in LEVELS},
    )


def measure(n_cases=2, losses=(1.0,)):
    wiring = _builders.Wiring(losses=losses)
    rows = token_gain.measure(
        cases=[case(i) for i in range(n_cases)],
        compute=wiring.compute, fast_weights=wiring.fast_weights,
    )
    return rows, wiring


def test_every_case_gets_a_reset_row_and_one_per_level():
    rows, _ = measure(n_cases=1)

    assert [r.arm for r in rows] == [RESET, *LEVELS]


def test_a_row_has_one_loss_per_scored_target_token():
    rows, _ = measure(n_cases=1)

    assert {len(r.losses) for r in rows} == {len(TARGET) - 1}


def test_each_prime_is_read_before_its_target_is_scored():
    _, wiring = measure(n_cases=1)

    assert wiring.compute.eval_forwards[1:3] == [case().primes[LEVELS[0]], TARGET]


def test_each_prime_advances_the_carry_before_its_target():
    _, wiring = measure(n_cases=1)

    assert wiring.fast_weights.events.count("advance_carry") == len(LEVELS)


def test_the_reset_target_is_scored_with_no_carry():
    _, wiring = measure(n_cases=1)

    assert wiring.fast_weights.events[1:3] == ["reset_carry", "reset_carry"]


def test_rows_carry_the_case_index():
    rows, _ = measure(n_cases=2)

    assert {r.doc_idx for r in rows} == {0, 1}


def test_the_carry_is_left_reset():
    _, wiring = measure()

    assert wiring.fast_weights.events[-1] == "reset_carry"


def measure_in_context(n_cases=1):
    wiring = _builders.Wiring(losses=(1.0,))
    rows = token_gain.measure(
        cases=[case(i) for i in range(n_cases)],
        compute=wiring.compute, fast_weights=wiring.fast_weights, in_context=True,
    )
    return rows, wiring


def test_in_context_adds_one_row_after_the_levels():
    rows, _ = measure_in_context()

    assert [r.arm for r in rows] == [RESET, *LEVELS, IN_CONTEXT]


def test_the_in_context_row_scores_the_same_positions_as_the_others():
    rows, _ = measure_in_context()

    assert {len(r.losses) for r in rows} == {len(TARGET) - 1}


def test_in_context_reads_the_related_prime_in_the_same_pass_as_the_target():
    _, wiring = measure_in_context()

    assert wiring.compute.eval_forwards[-1] == case().primes[RELATED] + TARGET


def test_a_generic_carry_is_installed_and_scored_as_its_own_row():
    wiring = _builders.Wiring(losses=(1.0,))
    rows = token_gain.measure(
        cases=[case()], compute=wiring.compute, fast_weights=wiring.fast_weights, generic=Carry(deltas={i: torch.zeros(2, 2) for i in _builders.LAYERS}),
    )

    assert [r.arm for r in rows] == [RESET, *LEVELS, GENERIC]
