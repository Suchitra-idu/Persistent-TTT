from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.app import pilot


def run(*, n_docs=4, seed=None, losses=(1.0,)):
    wiring = _builders.Wiring(losses=losses)
    rows = pilot.run(
        docs=_builders.docs(n_docs),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        seed=seed,
    )
    return rows, wiring


def of(rows, regime):
    return [row for row in rows if row.regime == regime]


class TestRegimes:
    def test_without_a_seed_it_runs_two_regimes(self):
        rows, _ = run(n_docs=3)

        assert {row.regime for row in rows} == {pilot.COLD, pilot.PERSIST}

    def test_with_a_seed_it_runs_three(self):
        rows, _ = run(n_docs=3, seed=_builders.carry())

        assert {row.regime for row in rows} == set(pilot.REGIMES)

    def test_every_regime_sees_the_same_document_sequence(self):
        rows, _ = run(n_docs=3, seed=_builders.carry())

        assert [r.doc_idx for r in of(rows, pilot.COLD)] == [
            r.doc_idx for r in of(rows, pilot.SEEDED)
        ]

    def test_each_regime_reports_one_row_per_document(self):
        rows, _ = run(n_docs=4)

        assert len(of(rows, pilot.PERSIST)) == 4


class TestCompounding:
    def test_cold_shows_the_same_state_at_every_position(self):
        rows, _ = run(n_docs=4)

        assert len({row.state_ratio for row in of(rows, pilot.COLD)}) == 1

    def test_persist_compounds_with_position(self):
        rows, _ = run(n_docs=4)
        ratios = [row.state_ratio for row in of(rows, pilot.PERSIST)]

        assert ratios == sorted(ratios) and ratios[0] < ratios[-1]

    def test_seeded_starts_above_persist(self):
        rows, _ = run(n_docs=4, seed=_builders.carry(value=5.0))

        assert (
            of(rows, pilot.SEEDED)[0].state_ratio
            > of(rows, pilot.PERSIST)[0].state_ratio
        )

    def test_positions_number_from_zero_in_every_regime(self):
        rows, _ = run(n_docs=3, seed=_builders.carry())

        assert [row.position for row in of(rows, pilot.SEEDED)] == [0, 1, 2]


class TestRows:
    def test_a_row_carries_the_document_it_measured(self):
        rows, _ = run(n_docs=2)

        assert of(rows, pilot.COLD)[1].doc_idx == 1

    def test_a_row_carries_both_the_nll_and_its_perplexity(self):
        import math

        rows, _ = run(n_docs=1, losses=(1.5,))
        row = of(rows, pilot.COLD)[0]

        assert (row.nll, row.ppl) == pytest.approx((1.5, math.exp(1.5)))

    def test_a_row_carries_its_token_count(self):
        rows, _ = run(n_docs=1)

        assert of(rows, pilot.COLD)[0].n_tokens == 8


def test_an_empty_document_list_measures_nothing():
    rows, _ = run(n_docs=0)

    assert rows == ()
