"""Smoke and reproducibility for every entrypoint, on fixture data and fakes."""

from __future__ import annotations

import json

import pytest

from tests.experiments import _builders
from ttt.experiments import (
    compounding_pilot_v1,
    holdout_eval_v1,
    holdout_generate_v1,
    single_doc_eval_v1,
    train_v1,
)

# High enough to flatten FakeGeneration's peaked logits, so the draw rather
# than the argmax decides the tokens and a shared generator is observable.
FLAT = 20.0


def lines():
    return [].append


class TestTrain:
    def test_a_run_takes_the_steps_its_schedule_planned(self):
        result = train_v1.run(
            _builders.resolved(),
            engine=_builders.engine(),
            source=_builders.data_source(),
            announce=lines(),
        )

        assert result.steps == result.total_steps > 0

    def test_it_visits_every_document_in_the_pool(self):
        engine = _builders.engine()

        result = train_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            announce=lines(),
        )

        assert result.micro_steps == len(engine.compute.forwards)

    def test_it_trains_a_carrier_per_source(self):
        result = train_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert set(result.carries) <= set(_builders.SOURCES)

    def test_it_announces_what_it_resolved(self):
        spoken = []

        train_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=spoken.append,
        )

        assert any("documents" in line for line in spoken)

    def test_it_checkpoints_at_the_end(self):
        engine = _builders.engine()

        train_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            announce=lines(),
        )

        assert engine.save.calls

    def test_it_resumes_from_the_carriers_the_engine_hands_it(self):
        engine = _builders.engine(carries={"FixtureProse": _builders.carry()})

        train_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            announce=lines(),
        )

        assert "install(carry)" in engine.fast_weights.events

    def test_eval_fires_when_the_cadence_is_on(self):
        resolution = _builders.resolved(eval_every=1)
        engine = _builders.engine(resolution)

        train_v1.run(
            resolution, engine=engine, source=_builders.data_source(), announce=lines()
        )

        assert engine.tracker.values_of("eval/carry_ppl")

    def test_eval_stays_silent_when_the_cadence_is_off(self):
        engine = _builders.engine()

        train_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            announce=lines(),
        )

        assert not engine.tracker.values_of("eval/carry_ppl")

    def test_the_same_seed_produces_the_same_losses(self):
        def losses():
            engine = _builders.engine()
            train_v1.run(
                _builders.resolved(), engine=engine,
                source=_builders.data_source(), announce=lines(),
            )
            return engine.tracker.values_of("train/loss")

        assert losses() == losses()

    def test_a_different_seed_visits_documents_in_a_different_order(self):
        def order(seed):
            engine = _builders.engine(_builders.resolved(seed=seed))
            train_v1.run(
                _builders.resolved(seed=seed), engine=engine,
                source=_builders.data_source(), announce=lines(),
            )
            return engine.compute.forwards

        assert order(1) != order(2)

    def test_eval_does_not_fire_on_an_empty_holdout(self):
        resolution = _builders.nothing_held_out(_builders.resolved(eval_every=1))
        spoken = []

        train_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=spoken.append,
        )

        assert not any("n_docs" in line for line in spoken)

    @pytest.mark.parametrize("strategy", ["hybrid", "everlasting"])
    def test_every_strategy_completes_a_run(self, strategy):
        resolution = _builders.resolved(strategy=strategy)

        result = train_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert result.steps > 0


class TestHoldoutEval:
    def test_it_measures_the_held_out_documents(self):
        report = holdout_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert report.rows

    def test_it_reports_the_budgeted_aggregates(self):
        report = holdout_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert "eval/gap_total" in report.metrics

    def test_it_prints_the_per_source_table(self):
        spoken = []

        holdout_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=spoken.append,
        )

        assert any("n_docs" in line for line in spoken)

    def test_a_seeded_run_measures_the_seeded_regimes(self):
        engine = _builders.engine(carries={"FixtureProse": _builders.carry()})

        report = holdout_eval_v1.run(
            _builders.resolved(), engine=engine,
            source=_builders.data_source(), announce=lines(),
        )

        assert "eval_seed/carry_ppl" in report.metrics

    def test_an_unseeded_run_measures_only_the_cold_trio(self):
        engine = _builders.engine(carries={"FixtureProse": _builders.carry()})

        report = holdout_eval_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            seeded=False, announce=lines(),
        )

        assert not any(key.startswith("eval_seed/") for key in report.metrics)

    def test_a_forced_source_seeds_every_document(self):
        engine = _builders.engine(carries={"FixtureCode": _builders.carry()})

        report = holdout_eval_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            force_source="FixtureCode", announce=lines(),
        )

        assert report.metrics["eval_seed/n_seeded_docs"] == len(
            {row.doc_idx for row in report.rows}
        )

    def test_forcing_a_source_with_no_carrier_is_rejected(self):
        engine = _builders.engine(carries={"FixtureCode": _builders.carry()})

        with pytest.raises(KeyError, match="no trained carrier"):
            holdout_eval_v1.run(
                _builders.resolved(), engine=engine,
                source=_builders.data_source(), force_source="FixtureMath",
                announce=lines(),
            )

    def test_an_empty_holdout_reports_nothing(self):
        resolution = _builders.nothing_held_out(_builders.resolved())

        report = holdout_eval_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert report.rows == ()


class TestSingleDocEval:
    def test_it_measures_one_document_in_slices(self):
        rows = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )

        assert len(rows) == 4

    def test_the_slices_partition_the_document(self):
        rows = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )

        assert sum(row.n_tokens for row in rows) == rows[0].n_tokens * 4

    def test_the_carry_compounds_across_slices(self):
        rows = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )

        assert rows[0].state_ratio < rows[-1].state_ratio

    def test_the_table_has_one_row_per_slice(self):
        engine = _builders.engine()
        rows = single_doc_eval_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            n_slices=4, announce=lines(),
        )

        assert len(single_doc_eval_v1.slice_table(rows, rows).rows) == 4

    def test_an_empty_holdout_measures_nothing(self):
        resolution = _builders.nothing_held_out(_builders.resolved())

        rows = single_doc_eval_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert rows == ()


class TestHoldoutGenerate:
    def test_it_produces_both_arms(self):
        out = holdout_generate_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert out[0].carry_on and out[0].carry_off

    def test_it_covers_the_documents_it_was_asked_for(self):
        out = holdout_generate_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_docs=2, announce=lines(),
        )

        assert len(out) == 2

    def test_the_carry_on_arm_accumulates_state(self):
        out = holdout_generate_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert out[0].state_ratio > 0.0

    def test_each_document_starts_from_a_clean_stream(self):
        engine = _builders.engine()

        out = holdout_generate_v1.run(
            _builders.resolved(), engine=engine,
            source=_builders.data_source(), n_docs=2, announce=lines(),
        )

        assert out[0].state_ratio == pytest.approx(out[1].state_ratio)

    def test_both_arms_draw_from_the_same_seeded_generator(self):
        out = holdout_generate_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), temperature=FLAT, announce=lines(),
        )

        assert out[0].carry_on == out[0].carry_off

    def test_greedy_decoding_reproduces_itself(self):
        def once():
            return holdout_generate_v1.run(
                _builders.resolved(), engine=_builders.engine(),
                source=_builders.data_source(), announce=lines(),
            )[0].carry_on

        assert once() == once()


class TestCompoundingPilot:
    def test_it_runs_the_cold_and_persist_regimes(self):
        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert {row.regime for row in rows} == {"cold", "persist"}

    def test_a_carrier_adds_the_seeded_regime(self):
        engine = _builders.engine(carries={"FixtureProse": _builders.carry()})

        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            pilot_source="FixtureProse", announce=lines(),
        )

        assert "seeded" in {row.regime for row in rows}

    def test_it_restricts_the_pool_to_one_source(self):
        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), pilot_source="FixtureCode",
            announce=lines(),
        )

        assert {row.source for row in rows} == {"FixtureCode"}

    def test_an_unknown_source_measures_nothing(self):
        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), pilot_source="Nope", announce=lines(),
        )

        assert rows == ()

    def test_persisting_compounds_with_position(self):
        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )
        persist = [row.state_ratio for row in rows if row.regime == "persist"]

        assert persist == sorted(persist) and persist[0] < persist[-1]

    def test_the_payload_is_json_with_one_record_per_row(self):
        rows = compounding_pilot_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), announce=lines(),
        )

        assert len(json.loads(compounding_pilot_v1.payload(_builders.resolved(), rows))["rows"]) == len(rows)

    def test_the_payload_records_the_config_that_produced_it(self):
        payload = json.loads(
            compounding_pilot_v1.payload(_builders.resolved(), ())
        )

        assert payload["config"]["dataset"] == "fixture"
