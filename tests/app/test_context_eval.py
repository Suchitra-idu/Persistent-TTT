from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import context_eval
from ttt.app.context_eval import CARRIED, CONTEXT, RESET, ContextRow
from ttt.core.metrics import ALL_SOURCES

N_SLICES = 4


def measure(*, n_docs=2, n_slices=N_SLICES):
    wiring = _builders.Wiring()
    rows = context_eval.measure(
        docs=_builders.docs(n_docs), compute=wiring.compute,
        fast_weights=wiring.fast_weights, n_slices=n_slices,
    )
    return rows, wiring


def test_every_slice_after_the_first_is_scored_in_every_arm():
    rows, _ = measure(n_docs=2)

    assert len(rows) == 2 * (N_SLICES - 1) * 3
    assert {row.arm for row in rows} == {RESET, CARRIED, CONTEXT}


def test_the_first_slice_is_never_scored():
    rows, _ = measure()

    assert min(row.slice_index for row in rows) == 1


def test_the_context_arm_prepends_the_previous_slice_and_scores_only_the_current_one():
    _, wiring = measure(n_docs=1)

    contexts = [
        (ids, score_from)
        for ids, score_from in zip(wiring.compute.eval_forwards, wiring.compute.eval_score_from)
        if score_from
    ]
    previous_length = len(contexts[0][0]) // 2
    assert len(contexts) == N_SLICES - 1
    assert contexts[0][1] == previous_length + 1


def test_every_arm_reports_the_same_scored_token_count():
    rows, _ = measure(n_docs=1)

    by_slice = {}
    for row in rows:
        by_slice.setdefault(row.slice_index, set()).add(row.n_tokens)
    assert all(len(counts) == 1 for counts in by_slice.values())


def test_the_carried_arm_reads_a_carry_and_the_reset_arm_does_not():
    _, wiring = measure(n_docs=1)

    assert wiring.fast_weights.events.count("reset_carry") >= 2 * (N_SLICES - 1)
    assert wiring.fast_weights.events.count("advance_carry") == N_SLICES


def _rows(reset, carried, context, n_docs=4):
    return tuple(
        ContextRow(arm=arm, source="alpha", doc_idx=d, slice_index=1, n_tokens=10, ppl=ppl)
        for d in range(n_docs)
        for arm, ppl in ((RESET, reset), (CARRIED, carried), (CONTEXT, context))
    )


def test_the_gains_are_measured_against_the_reset_arm():
    estimates = context_eval.estimates(_rows(10.0, 9.0, 8.0), rng=NumpyRng(0), n_boot=20)
    pooled = {e.measure: e.interval.point for e in estimates if e.source == ALL_SOURCES}

    assert pooled == pytest.approx({"carry_gain": 1.0, "context_gain": 2.0, "context_over_carry": 1.0})


def test_the_carry_share_is_its_gain_over_the_contexts():
    share = context_eval.carry_share(_rows(10.0, 9.0, 8.0), rng=NumpyRng(0), n_boot=20)

    assert share.point == pytest.approx(0.5)
