from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.app import swap_eval
from ttt.app.swap_eval import UNPRIMED


def by_source(per_source=3):
    docs = _builders.docs(per_source * 2)
    return {"alpha": docs[0::2], "beta": docs[1::2]}


def measure(*, per_source=3, n_primes=1, n_targets=2):
    wiring = _builders.Wiring()
    rows = swap_eval.measure(
        by_source=by_source(per_source),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        n_primes=n_primes,
        n_targets=n_targets,
    )
    return rows, wiring


def test_each_target_is_read_unprimed_and_after_every_source():
    rows, _ = measure(n_targets=2)

    assert len(rows) == 2 * 2 * 3


def test_an_unprimed_read_starts_from_no_carry():
    rows, _ = measure()

    assert {r.state_ratio for r in rows if r.prime_source == UNPRIMED} == {
        rows[0].state_ratio
    }


def test_a_target_is_never_its_own_prime():
    rows, wiring = measure(per_source=2, n_primes=1, n_targets=1)

    assert len(wiring.compute.eval_forwards) == sum(
        2 if r.prime_source else 1 for r in rows
    )


def test_a_source_short_of_primes_is_skipped_for_that_pair():
    rows, _ = measure(per_source=2, n_primes=2, n_targets=1)

    assert {(r.target_source, r.prime_source) for r in rows} == {
        ("alpha", UNPRIMED), ("alpha", "beta"), ("beta", UNPRIMED), ("beta", "alpha"),
    }


def test_zero_primes_is_rejected():
    with pytest.raises(ValueError):
        measure(n_primes=0)
