from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.app import repeat_eval


def measure(*, n_docs=2, n_repeats=4):
    wiring = _builders.Wiring()
    return repeat_eval.measure(
        docs=_builders.docs(n_docs),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        n_repeats=n_repeats,
    )


def test_one_row_per_document_per_repeat():
    assert len(measure(n_docs=2, n_repeats=4)) == 8


def test_repeats_are_numbered_from_zero_within_each_document():
    rows = measure(n_docs=2, n_repeats=3)

    assert [r.repeat for r in rows] == [0, 1, 2, 0, 1, 2]


def test_the_carry_grows_across_repeats_of_one_document():
    rows = measure(n_docs=1, n_repeats=3)

    assert rows[0].state_ratio < rows[1].state_ratio < rows[2].state_ratio


def test_each_document_starts_from_a_cleared_carry():
    rows = measure(n_docs=2, n_repeats=3)

    assert rows[3].state_ratio == pytest.approx(rows[0].state_ratio)


def test_fewer_than_two_repeats_is_rejected():
    with pytest.raises(ValueError):
        measure(n_repeats=1)
