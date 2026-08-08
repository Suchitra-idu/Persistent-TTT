from __future__ import annotations

import math

import pytest

from tests.app import _builders
from ttt.app import eval_loop
from ttt.core.types import (
    CARRY,
    CARRY_OFF,
    COLD_CARRY,
    COLD_CARRY_OFF,
    FRESH,
    LORA_ONLY,
)

SLICES = 4

# Cycle length coprime with the slice count, so each document lands on a
# different offset and a subset mean is distinguishable from the pool mean.
SHIFTING = (1.0, 0.95, 0.9, 0.85, 0.8)


def measure(*, n_docs=2, n_slices=SLICES, losses=(1.0, 0.9, 0.8, 0.7), carries=None):
    wiring = _builders.Wiring(losses=losses)
    report = eval_loop.evaluate(
        docs=_builders.docs(n_docs),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        n_slices=n_slices,
        carries=carries,
    )
    return report, wiring


class TestZeroSeedPass:
    def test_it_measures_four_regimes_per_document(self):
        report, _ = measure(n_docs=2)

        assert len(report.rows) == 8

    def test_the_regimes_are_the_cold_quartet(self):
        report, _ = measure()

        assert {row.regime for row in report.rows} == {
            COLD_CARRY,
            COLD_CARRY_OFF,
            LORA_ONLY,
            FRESH,
        }

    def test_every_slice_is_one_evaluation_forward(self):
        _, wiring = measure(n_docs=1, n_slices=SLICES)

        assert len(wiring.compute.eval_forwards) == 4 * SLICES

    def test_only_the_fresh_regime_disables_lora(self):
        _, wiring = measure(n_docs=1, n_slices=SLICES)

        assert wiring.compute.eval_lora_flags.count(False) == SLICES

    def test_no_evaluation_forward_leaves_a_backward_pending(self):
        _, wiring = measure()

        assert wiring.compute.backwards == []

    def test_it_reports_every_budgeted_aggregate(self):
        report, _ = measure()

        assert set(report.metrics) == {
            "eval/carry_ppl",
            "eval/carry_off_ppl",
            "eval/lora_only_ppl",
            "eval/fresh_ppl",
            "eval/gap_lora",
            "eval/gap_within",
            "eval/gap_between",
            "eval/gap_total",
            "eval/state_ratio_final",
        }

    def test_the_gaps_decompose_additively(self):
        report, _ = measure()
        gaps = report.metrics

        assert gaps["eval/gap_lora"] + gaps["eval/gap_within"] + gaps[
            "eval/gap_between"
        ] == pytest.approx(gaps["eval/gap_total"])

    def test_carry_off_resets_once_per_extra_slice(self):
        _, one = measure(n_docs=1, n_slices=1)
        _, four = measure(n_docs=1, n_slices=4)
        resets = [w.fast_weights.events.count("reset_carry") for w in (one, four)]

        assert resets[1] - resets[0] == 3

    def test_carry_persists_across_slices(self):
        report, _ = measure(n_docs=1)
        carry_row = next(r for r in report.rows if r.regime == COLD_CARRY)
        off_row = next(r for r in report.rows if r.regime == COLD_CARRY_OFF)

        assert carry_row.state_ratio_final > off_row.state_ratio_final

    def test_a_frozen_fast_weight_leaves_no_state(self):
        report, _ = measure(n_docs=1)
        fresh = next(r for r in report.rows if r.regime == FRESH)

        assert fresh.state_ratio_final == 0.0

    def test_it_covers_every_token_of_a_document(self):
        report, _ = measure(n_docs=1)
        row = next(r for r in report.rows if r.regime == COLD_CARRY)

        assert row.n_tokens == 8


class TestSeededPass:
    def test_a_source_with_a_carrier_is_measured_seeded(self):
        report, _ = measure(n_docs=2, carries={"alpha": _builders.carry()})

        assert {CARRY, CARRY_OFF} <= {row.regime for row in report.rows}

    def test_a_source_without_a_carrier_is_not(self):
        report, _ = measure(n_docs=2, carries={"alpha": _builders.carry()})
        seeded = {row.source for row in report.rows if row.regime == CARRY}

        assert seeded == {"alpha"}

    def test_it_reports_the_seeded_aggregates(self):
        report, _ = measure(n_docs=2, carries={"alpha": _builders.carry()})

        assert {
            "eval_seed/carry_ppl",
            "eval_seed/carry_off_ppl",
            "eval_seed/gap_between",
            "eval_seed/gap_seed",
            "eval_seed/n_seeded_docs",
        } <= set(report.metrics)

    def test_it_counts_the_documents_it_could_seed(self):
        report, _ = measure(n_docs=4, carries={"alpha": _builders.carry()})

        assert report.metrics["eval_seed/n_seeded_docs"] == 2

    def test_no_carrier_means_no_seeded_keys(self):
        report, _ = measure(n_docs=2, carries={})

        assert not any(key.startswith("eval_seed/") for key in report.metrics)

    def test_the_seed_is_reinstalled_after_every_carry_off_reset(self):
        seed = {"alpha": _builders.carry()}
        _, one = measure(n_docs=1, n_slices=1, carries=seed)
        _, four = measure(n_docs=1, n_slices=4, carries=seed)
        installs = [w.fast_weights.events.count("install(carry)") for w in (one, four)]

        assert installs[1] - installs[0] == 3

    def test_the_seed_gap_compares_seeded_documents_against_themselves_cold(self):
        from ttt.core.metrics import geometric_mean_ppl

        report, _ = measure(n_docs=4, losses=SHIFTING, carries={"alpha": _builders.carry()})
        seeded = {row.doc_idx for row in report.rows if row.regime == CARRY}
        cold = geometric_mean_ppl(
            [
                row.ppl
                for row in report.rows
                if row.regime == COLD_CARRY and row.doc_idx in seeded
            ]
        )

        assert report.metrics["eval_seed/gap_seed"] == pytest.approx(
            cold - report.metrics["eval_seed/carry_ppl"]
        )

    def test_the_seed_gap_ignores_the_documents_it_could_not_seed(self):
        from ttt.core.metrics import geometric_mean_ppl

        report, _ = measure(n_docs=4, losses=SHIFTING, carries={"alpha": _builders.carry()})
        whole_pool = geometric_mean_ppl(
            [row.ppl for row in report.rows if row.regime == COLD_CARRY]
        )

        assert report.metrics["eval_seed/gap_seed"] != pytest.approx(
            whole_pool - report.metrics["eval_seed/carry_ppl"]
        )

    def test_fresh_is_not_measured_twice(self):
        report, _ = measure(n_docs=1, carries={"alpha": _builders.carry()})

        assert len([r for r in report.rows if r.regime == FRESH]) == 1


class TestStateRestoration:
    def test_it_puts_back_the_carry_it_found(self):
        wiring = _builders.Wiring()
        wiring.fast_weights.set_mode(evolve=True, stream=False, session=True)
        wiring.fast_weights.stage(8)
        wiring.fast_weights.advance_carry()
        before = wiring.fast_weights.state_ratio(family="carry")

        eval_loop.evaluate(
            docs=_builders.docs(2),
            compute=wiring.compute,
            fast_weights=wiring.fast_weights,
            n_slices=SLICES,
        )

        assert wiring.fast_weights.state_ratio(family="carry") == pytest.approx(before)

    def test_it_restores_even_when_a_measurement_raises(self):
        wiring = _builders.Wiring()
        wiring.fast_weights.set_mode(evolve=True, stream=False, session=True)
        wiring.fast_weights.stage(8)
        wiring.fast_weights.advance_carry()
        before = wiring.fast_weights.state_ratio(family="carry")

        with pytest.raises(ZeroDivisionError):
            eval_loop.evaluate(
                docs=_builders.docs(1),
                compute=_Exploding(),
                fast_weights=wiring.fast_weights,
                n_slices=SLICES,
            )

        assert wiring.fast_weights.state_ratio(family="carry") == pytest.approx(before)

    def test_it_leaves_the_fast_weight_thawed_for_the_training_loop(self):
        _, wiring = measure()

        assert wiring.fast_weights.evolve

    def test_it_restores_the_training_session_mode(self):
        wiring = _builders.Wiring()

        eval_loop.evaluate(
            docs=_builders.docs(1),
            compute=wiring.compute,
            fast_weights=wiring.fast_weights,
            n_slices=SLICES,
            session_training=False,
        )

        assert not wiring.fast_weights.session_mode


class _Exploding:
    def eval_loss(self, token_ids, *, lora: bool = True):
        raise ZeroDivisionError("measurement blew up")


class TestSliceRows:
    def test_it_keeps_one_row_per_doc_regime_and_slice(self):
        report, _ = measure(n_docs=2, n_slices=SLICES)

        assert len(report.slices) == 2 * 4 * SLICES

    def test_slice_indices_run_in_order_per_doc_and_regime(self):
        report, _ = measure(n_docs=1, n_slices=SLICES)
        indices = [s.slice_index for s in report.slices if s.regime == COLD_CARRY]

        assert indices == list(range(SLICES))

    def test_an_empty_holdout_has_no_slices(self):
        report, _ = measure(n_docs=0)

        assert report.slices == ()


class TestByteAccounting:
    def test_an_untracked_doc_reports_zero_bytes_throughout(self):
        report, _ = measure(n_docs=1)

        assert all(row.n_bytes == 0 for row in report.rows)
        assert all(s.n_bytes == 0 for s in report.slices)

    def test_a_docs_bytes_split_proportionally_to_its_slices_tokens(self):
        wiring = _builders.Wiring(losses=(1.0, 0.9, 0.8, 0.7))
        doc = _builders.doc(0, n_tokens=8, n_bytes=40)
        report = eval_loop.evaluate(
            docs=[doc], compute=wiring.compute, fast_weights=wiring.fast_weights,
            n_slices=SLICES,
        )
        slices = [s for s in report.slices if s.regime == COLD_CARRY]

        # 8 tokens split into 4 equal slices of 2; 40 bytes at 5 bytes/token.
        assert [s.n_bytes for s in slices] == [10, 10, 10, 10]

    def test_a_docs_slice_bytes_sum_back_to_the_row_total(self):
        wiring = _builders.Wiring(losses=(1.0, 0.9, 0.8, 0.7))
        doc = _builders.doc(0, n_tokens=8, n_bytes=40)
        report = eval_loop.evaluate(
            docs=[doc], compute=wiring.compute, fast_weights=wiring.fast_weights,
            n_slices=SLICES,
        )
        row = next(r for r in report.rows if r.regime == COLD_CARRY)

        assert row.n_bytes == doc.n_bytes


class TestPerSourceSummaries:
    def test_it_summarises_each_source(self):
        report, _ = measure(n_docs=4)

        assert [s.source for s in report.summaries] == list(_builders.SOURCES)

    def test_a_summary_counts_its_documents(self):
        report, _ = measure(n_docs=4)

        assert report.summaries[0].n_docs == 2

    def test_a_summary_carries_the_gap_decomposition(self):
        report, _ = measure(n_docs=4)

        assert report.summaries[0].gaps.is_additive


def test_an_empty_holdout_reports_nothing():
    report, _ = measure(n_docs=0)

    assert (report.rows, report.metrics, report.summaries) == ((), {}, ())


def test_a_diverged_loss_becomes_an_infinite_perplexity():
    report, _ = measure(n_docs=1, losses=(1e6,))

    assert math.isinf(report.metrics["eval/carry_ppl"])
