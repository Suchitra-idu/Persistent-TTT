from __future__ import annotations

import math

import pytest

from tests.app import _builders
from ttt.app import session_eval


def measure(*, n_docs=3, losses=(1.0,), **kwargs):
    wiring = _builders.Wiring(losses=losses)
    rows = session_eval.session_perplexity(
        docs=_builders.docs(n_docs),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        **kwargs,
    )
    return rows, wiring


class TestItemRows:
    def test_one_row_per_document_at_a_single_slice(self):
        rows, _ = measure(n_docs=3)

        assert len(rows) == 3

    def test_one_row_per_slice(self):
        rows, _ = measure(n_docs=2, n_slices=4)

        assert len(rows) == 8

    def test_positions_are_sequential_across_the_whole_session(self):
        rows, _ = measure(n_docs=2, n_slices=2)

        assert [row.position for row in rows] == [0, 1, 2, 3]

    def test_a_row_carries_its_document_and_source(self):
        rows, _ = measure(n_docs=2)

        assert [(row.doc_idx, row.source) for row in rows[:2]] == [
            (0, "alpha"),
            (1, "beta"),
        ]

    def test_the_slices_partition_the_document(self):
        rows, _ = measure(n_docs=1, n_slices=4)

        assert sum(row.n_tokens for row in rows) == 8

    def test_the_perplexity_is_the_exponentiated_loss(self):
        rows, _ = measure(n_docs=1, losses=(2.0,))

        assert rows[0].ppl == pytest.approx(math.exp(2.0))

    def test_the_nll_is_the_loss_it_came_from(self):
        rows, _ = measure(n_docs=1, losses=(2.0,))

        assert rows[0].nll == pytest.approx(2.0)


class TestResetLadder:
    def test_by_default_the_carry_restarts_at_each_document(self):
        rows, _ = measure(n_docs=3, reset_between_docs=True)

        assert len({row.state_ratio for row in rows}) == 1

    def test_persisting_across_documents_compounds_the_carry(self):
        rows, _ = measure(n_docs=3, reset_between_docs=False)

        assert rows[0].state_ratio < rows[1].state_ratio < rows[2].state_ratio

    def test_resetting_between_items_isolates_within_item_adaptation(self):
        rows, _ = measure(n_docs=1, n_slices=4, reset_between_items=True)

        assert len({row.state_ratio for row in rows}) == 1

    def test_persisting_across_items_compounds_within_a_document(self):
        rows, _ = measure(n_docs=1, n_slices=4, reset_between_items=False)

        assert rows[0].state_ratio < rows[-1].state_ratio

    def test_a_frozen_fast_weight_leaves_no_state(self):
        rows, _ = measure(n_docs=2, evolve=False)

        assert [row.state_ratio for row in rows] == [0.0, 0.0]


class TestEverlastingCarriers:
    def test_a_document_starts_from_its_own_source_carrier(self):
        _, wiring = measure(n_docs=2, carries={"alpha": _builders.carry()})

        assert wiring.fast_weights.events.count("install(carry)") == 1

    def test_a_source_with_no_carrier_starts_cold(self):
        _, wiring = measure(n_docs=2, carries={})

        assert wiring.fast_weights.events.count("install(carry)") == 0

    def test_a_forced_source_overrides_the_document_label(self):
        _, wiring = measure(
            n_docs=2, carries={"beta": _builders.carry()}, force_source="beta"
        )

        assert wiring.fast_weights.events.count("install(carry)") == 2

    def test_a_carrier_lifts_the_state_the_first_slice_sees(self):
        seeded, _ = measure(n_docs=1, carries={"alpha": _builders.carry(value=5.0)})
        cold, _ = measure(n_docs=1)

        assert seeded[0].state_ratio > cold[0].state_ratio


def test_an_empty_document_list_measures_nothing():
    rows, _ = measure(n_docs=0)

    assert rows == ()


def test_it_never_leaves_a_backward_pending():
    _, wiring = measure(n_docs=3)

    assert wiring.compute.backwards == []
