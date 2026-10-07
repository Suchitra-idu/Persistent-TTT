from __future__ import annotations

import pytest

from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import evidence
from ttt.app.boundary_eval import DOC_RESET, PERSIST, SLICE_RESET, ArmRow
from ttt.app.repeat_eval import RepeatItem
from ttt.app.swap_eval import UNPRIMED, SwapRow
from ttt.core.metrics import ALL_SOURCES
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, FRESH, LORA_ONLY, EvalRow

SOURCES = ("alpha", "beta")
REGIME_PPL = {FRESH: 20.0, LORA_ONLY: 16.0, COLD_CARRY_OFF: 15.0, COLD_CARRY: 14.0}


def kw():
    return dict(rng=ScriptedRng(), n_boot=4)


def eval_rows(n_docs=3, ppl=REGIME_PPL, sources=SOURCES):
    return tuple(
        EvalRow(doc_idx=i, source=s, regime=regime, n_tokens=10, ppl=value)
        for s in sources
        for i in range(n_docs)
        for regime, value in ppl.items()
    )


def arm_rows(chain=3, ppl=None):
    ppl = ppl or {PERSIST: 9.0, DOC_RESET: 10.0, SLICE_RESET: 12.0}
    return tuple(
        ArmRow(arm=arm, source="alpha", doc_idx=pos, chain_position=pos, slice_index=0,
               n_tokens=10, ppl=value, state_ratio=0.1, gate_mean=0.5)
        for arm, value in ppl.items()
        for pos in range(chain)
    )


def repeat_rows(ppls=(10.0, 9.0, 8.5), n_docs=2):
    return tuple(
        RepeatItem(source="alpha", doc_idx=d, repeat=r, n_tokens=10, ppl=p,
                   state_ratio=0.1 * (r + 1), gate_mean=0.4)
        for d in range(n_docs)
        for r, p in enumerate(ppls)
    )


def swap_rows():
    def row(target, prime, ppl):
        return SwapRow(target_source=target, prime_source=prime, doc_idx=0,
                       n_tokens=10, ppl=ppl, state_ratio=0.0)

    return (
        row("alpha", UNPRIMED, 10.0), row("alpha", "alpha", 8.0), row("alpha", "beta", 9.0),
        row("beta", UNPRIMED, 10.0), row("beta", "beta", 8.0), row("beta", "alpha", 9.0),
    )


def by(estimates, measure, source=ALL_SOURCES):
    return next(e for e in estimates if e.measure == measure and e.source == source)


def test_resamples_draw_n_indices_n_boot_times():
    drawn = evidence.resamples(3, 2, ScriptedRng(integers=(2, 0, 1)))

    assert drawn == ((2, 0, 1), (2, 0, 1))


def test_resampling_nothing_draws_nothing():
    assert evidence.resamples(0, 5, ScriptedRng()) == ()


def test_paired_estimates_report_each_source_then_the_pool():
    reference = {("a", 0): 10.0, ("b", 0): 10.0}

    estimates = evidence.paired_estimates("m", reference, reference, **kw())

    assert [e.source for e in estimates] == ["a", "b", ALL_SOURCES]


def test_paired_estimates_pair_only_documents_both_arms_measured():
    estimates = evidence.paired_estimates(
        "m", {("a", 0): 10.0, ("a", 1): 10.0}, {("a", 1): 8.0, ("a", 2): 8.0}, **kw()
    )

    assert estimates[0].n_docs == 1


def test_per_doc_weights_slices_by_tokens():
    rows = arm_rows(chain=1) + (
        ArmRow(arm=PERSIST, source="alpha", doc_idx=0, chain_position=0, slice_index=1,
               n_tokens=30, ppl=1.0, state_ratio=0.0, gate_mean=1.0),
    )

    assert evidence.per_doc(rows, where=lambda r: r.arm == PERSIST)[("alpha", 0)] == pytest.approx(9.0 ** 0.25)


@pytest.mark.parametrize(
    ("measure", "expected"),
    [("lora", 4.0), ("within", 1.0), ("between_slices", 1.0), ("total", 6.0)],
)
def test_regime_estimates_are_the_gap_decomposition(measure, expected):
    estimates = evidence.regime_estimates(eval_rows(), **kw())

    assert by(estimates, measure).interval.point == pytest.approx(expected)


def test_the_cross_document_gap_skips_each_chains_first_document():
    estimates = evidence.boundary_estimates(arm_rows(chain=3), **kw())

    assert by(estimates, evidence.CROSS_DOC).n_docs == 2


@pytest.mark.parametrize(("measure", "expected"), [(evidence.CROSS_DOC, 1.0), (evidence.CROSS_SLICE, 2.0)])
def test_boundary_gaps_compare_adjacent_reset_policies(measure, expected):
    estimates = evidence.boundary_estimates(arm_rows(), **kw())

    assert by(estimates, measure).interval.point == pytest.approx(expected)


def test_the_repeat_curve_starts_at_the_first_replay():
    points = evidence.repeat_curve(repeat_rows(), **kw())

    assert sorted({p.repeat for p in points}) == [1, 2]


def test_the_repeat_curve_measures_against_the_cold_pass():
    points = evidence.repeat_curve(repeat_rows(ppls=(10.0, 9.0, 8.5)), **kw())

    assert [p.gap.point for p in points if p.source == ALL_SOURCES] == pytest.approx([1.0, 1.5])


def test_the_repeat_curve_carries_the_state_median_across_documents():
    points = evidence.repeat_curve(repeat_rows(), **kw())

    assert points[0].state_ratio.median == pytest.approx(0.2)


def test_a_flat_curve_has_a_zero_trend():
    estimates = evidence.repeat_trend(repeat_rows(ppls=(10.0, 9.0, 9.0, 9.0)), **kw())

    assert by(estimates, "repeat_3_vs_1").interval.point == pytest.approx(0.0)


def test_a_single_replay_has_no_trend():
    assert evidence.repeat_trend(repeat_rows(ppls=(10.0, 9.0)), **kw()) == ()


def test_swap_estimates_name_the_target_as_the_source():
    estimates = evidence.swap_estimates(swap_rows(), **kw())

    assert by(estimates, "prime:beta", "alpha").interval.point == pytest.approx(1.0)


def test_same_source_priming_is_compared_with_the_other_sources():
    estimates = evidence.swap_estimates(swap_rows(), **kw())

    assert by(estimates, evidence.SAME_VS_OTHER).interval.point == pytest.approx(1.0)


def test_domain_correlates_rank_sources_by_fresh_perplexity():
    fresh = {"a": 10.0, "b": 20.0, "c": 30.0}
    estimates = (
        evidence.paired_estimates("within", {("a", 0): 11.0}, {("a", 0): 10.0}, **kw())[:1]
        + evidence.paired_estimates("within", {("b", 0): 12.0}, {("b", 0): 10.0}, **kw())[:1]
        + evidence.paired_estimates("within", {("c", 0): 13.0}, {("c", 0): 10.0}, **kw())[:1]
    )

    assert evidence.domain_correlates(fresh, estimates)[0].rho == pytest.approx(1.0)


def test_fresh_by_source_reads_only_the_fresh_regime():
    fresh = evidence.fresh_by_source(eval_rows(sources=("a", "b")))

    assert fresh == pytest.approx({"a": 20.0, "b": 20.0})


def test_the_same_base_model_on_the_same_documents_has_no_fresh_gap():
    estimates = evidence.checkpoint_estimates(eval_rows(), eval_rows(), **kw())

    assert by(estimates, "fresh_sanity").interval.point == pytest.approx(0.0)


def test_a_candidate_lower_than_the_baselines_lora_wins():
    candidate = eval_rows(ppl={**REGIME_PPL, COLD_CARRY: 12.0})

    estimates = evidence.checkpoint_estimates(eval_rows(), candidate, **kw())

    assert by(estimates, "vs_cold_carry").interval.point == pytest.approx(4.0)


def test_unmatched_docs_counts_both_sides():
    assert evidence.unmatched_docs(eval_rows(n_docs=3), eval_rows(n_docs=2)) == 2


def test_rows_survive_a_json_round_trip_through_storage():
    storage = InMemoryStorage()

    evidence.save_json(storage, "raw/x.json", evidence.records(arm_rows()))

    assert evidence.rows_from(ArmRow, evidence.load_json(storage, "raw/x.json")) == arm_rows()


def test_domain_correlates_ignore_a_gap_that_only_scales_with_base_perplexity():
    fresh = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
    bumps = {"a": 0.02, "b": 0.01, "c": 0.04, "d": 0.03}
    estimates = tuple(
        evidence.paired_estimates(
            "within", {(s, 0): p}, {(s, 0): p * (0.9 - bumps[s])}, **kw()
        )[0]
        for s, p in fresh.items()
    )

    assert evidence.domain_correlates(fresh, estimates)[0].rho == pytest.approx(0.6)
