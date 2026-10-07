from __future__ import annotations

from tests.app import _builders
from ttt.app import boundary_eval
from ttt.app.boundary_eval import DOC_RESET, PERSIST, SLICE_RESET


def measure(*, chain_length=3, n_slices=2, losses=(1.0,)):
    wiring = _builders.Wiring(losses=losses)
    chains = {
        "alpha": _builders.docs(chain_length * 2)[0::2],
        "beta": _builders.docs(chain_length * 2)[1::2],
    }
    rows = boundary_eval.measure(
        chains=chains,
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        n_slices=n_slices,
    )
    return rows, wiring


def test_every_arm_measures_every_slice_of_every_chained_document():
    rows, _ = measure(chain_length=3, n_slices=2)

    assert len(rows) == 3 * 2 * 3 * 2


def test_the_arms_are_the_three_reset_policies():
    rows, _ = measure()

    assert {row.arm for row in rows} == {PERSIST, DOC_RESET, SLICE_RESET}


def test_chain_position_counts_documents_within_a_source():
    rows, _ = measure(chain_length=3, n_slices=1)

    assert [r.chain_position for r in rows if r.arm == PERSIST and r.source == "alpha"] == [0, 1, 2]


def test_slice_index_restarts_at_each_document():
    rows, _ = measure(chain_length=2, n_slices=2)

    assert [r.slice_index for r in rows if r.arm == PERSIST and r.source == "alpha"] == [0, 1, 0, 1]


def test_persist_never_resets_the_carry_inside_a_chain():
    rows, _ = measure(chain_length=3, n_slices=2)

    assert all(r.state_ratio > 0.0 for r in rows if r.arm == PERSIST)


def test_slice_reset_never_accumulates_past_one_slice():
    rows, _ = measure(chain_length=3, n_slices=2)

    assert len({r.state_ratio for r in rows if r.arm == SLICE_RESET}) == 1


def test_doc_reset_starts_the_carry_over_at_each_document():
    rows, _ = measure(chain_length=2, n_slices=2)
    doc_reset = [r for r in rows if r.arm == DOC_RESET and r.source == "alpha"]
    persist = [r for r in rows if r.arm == PERSIST and r.source == "alpha"]

    assert doc_reset[2].state_ratio < persist[2].state_ratio
