"""Smoke and reproducibility for every entrypoint, on fixture data and fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.experiments import _builders
from ttt.core import metrics
from ttt.core.types import COLD_CARRY, FRESH
from ttt.experiments import (
    compounding_pilot_v1,
    holdout_eval_v1,
    holdout_generate_v1,
    repeat_carry_eval_v1,
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

    def test_on_data_ready_fires_once_before_training_starts(self):
        calls = []

        def mark():
            calls.append(len(engine.compute.forwards))

        engine = _builders.engine()
        train_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            announce=lines(), on_data_ready=mark,
        )

        assert calls == [0]

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
    def test_it_measures_one_document_per_source(self):
        by_source = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )

        assert [source for source, _ in by_source] == sorted(_builders.SOURCES)

    def test_it_measures_each_documents_slices(self):
        by_source = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )

        assert all(len(rows) == 4 for _, rows in by_source)

    def test_the_slices_partition_the_document(self):
        by_source = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )
        _, rows = by_source[0]

        assert sum(row.n_tokens for row in rows) == rows[0].n_tokens * 4

    def test_the_carry_compounds_across_slices(self):
        by_source = single_doc_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, announce=lines(),
        )
        _, rows = by_source[0]

        assert rows[0].state_ratio < rows[-1].state_ratio

    def test_the_table_has_one_row_per_slice(self):
        engine = _builders.engine()
        by_source = single_doc_eval_v1.run(
            _builders.resolved(), engine=engine, source=_builders.data_source(),
            n_slices=4, announce=lines(),
        )
        _, rows = by_source[0]

        assert len(single_doc_eval_v1.slice_table(rows, rows).rows) == 4

    def test_an_empty_holdout_measures_nothing(self):
        resolution = _builders.nothing_held_out(_builders.resolved())

        by_source = single_doc_eval_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert by_source == ()


class TestSingleDocEvalChained:
    def test_it_chains_n_docs_worth_of_slices_per_source(self):
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )

        assert all(len(rows) == 4 * 3 for _, rows, _ in by_source)

    def test_n_docs_beyond_the_configured_eval_cap_still_chains_that_many(self):
        """The trap: train.eval_n_docs_per_source (default 5) caps the
        holdout pool before run_chained ever sees it, so n_docs=6 used to
        silently chain only 5 — the default cap, not what was asked for."""
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(_builders.resolved(), holdout_last_n=48),
            engine=_builders.engine(),
            source=_builders.data_source(sources=_builders.SOURCES * 8),
            n_slices=2, n_docs=6, announce=lines(),
        )

        assert all(len(rows) == 2 * 6 for _, rows, _ in by_source)

    def test_positions_run_continuously_across_documents(self):
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )
        _, rows, _ = by_source[0]

        assert [row.position for row in rows] == list(range(12))

    def test_the_carry_is_never_reset_at_a_document_boundary(self):
        """The whole point: state_ratio keeps climbing past the first
        document's own slices, into the second and third."""
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )
        _, rows, _ = by_source[0]

        assert rows[0].state_ratio < rows[4].state_ratio < rows[8].state_ratio

    def test_the_table_names_each_rows_document(self):
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )
        _, rows, rows_off = by_source[0]

        table = single_doc_eval_v1.chain_table(rows, rows_off)

        assert len(table.rows) == 12
        assert len({row[1] for row in table.rows}) == 3

    def test_it_covers_every_source_not_just_one(self):
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )

        assert {src for src, _, _ in by_source} == set(_builders.SOURCES)

    def test_each_sources_chain_never_crosses_into_another_source(self):
        """carry_scope=SOURCE keeps each source's carrier separate in
        production (train_loop._seed) — a chain must not mix sources."""
        by_source = single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )

        for src, rows, _ in by_source:
            assert {row.source for row in rows} == {src}

    def test_an_empty_holdout_measures_nothing(self):
        resolution = _builders.nothing_held_out(_builders.resolved())

        by_source = single_doc_eval_v1.run_chained(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert by_source == ()


class TestPlotChained:
    def _by_source(self):
        return single_doc_eval_v1.run_chained(
            _builders.wider_holdout(
                _builders.resolved(eval_n_docs_per_source=3), holdout_last_n=12
            ),
            engine=_builders.engine(),
            source=_builders.data_source(), n_slices=4, n_docs=3, announce=lines(),
        )

    def test_it_writes_a_png_named_with_the_resumed_step(self, tmp_path):
        pytest.importorskip("matplotlib")

        path = single_doc_eval_v1.plot_chained(
            self._by_source(), resume_from="step_600", out_dir=str(tmp_path)
        )

        assert Path(path).is_file()
        assert Path(path).name.startswith("single_doc_eval_step_600_")

    def test_an_empty_resume_from_still_names_the_file(self, tmp_path):
        pytest.importorskip("matplotlib")

        path = single_doc_eval_v1.plot_chained(
            self._by_source(), resume_from="", out_dir=str(tmp_path)
        )

        assert Path(path).name.startswith("single_doc_eval_no-resume_")


class TestRepeatCarryEval:
    def test_it_replays_every_document_n_repeats_times(self):
        summaries = repeat_carry_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_repeats=3, announce=lines(),
        )

        per_source = [s for s in summaries if s.source != metrics.ALL_SOURCES]
        assert {s.repeat for s in per_source} == {0, 1, 2}

    def test_it_covers_every_source_plus_an_all_rollup(self):
        summaries = repeat_carry_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_repeats=2, announce=lines(),
        )

        assert set(_builders.SOURCES) <= {s.source for s in summaries}
        assert metrics.ALL_SOURCES in {s.source for s in summaries}

    def test_ppl_improves_across_repeats_with_the_scripted_losses(self):
        summaries = repeat_carry_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_repeats=3, announce=lines(),
        )

        by_repeat = {
            s.repeat: s.ppl_by_regime[COLD_CARRY]
            for s in summaries
            if s.source == metrics.ALL_SOURCES
        }
        assert by_repeat[2] < by_repeat[0]

    def test_the_reference_regimes_stay_flat_across_repeats(self):
        summaries = repeat_carry_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_repeats=3, announce=lines(),
        )

        by_repeat = {
            s.repeat: s.ppl_by_regime[FRESH]
            for s in summaries
            if s.source == metrics.ALL_SOURCES
        }
        assert by_repeat[0] == pytest.approx(by_repeat[2])

    def test_it_prints_the_repeat_table(self):
        spoken = []

        repeat_carry_eval_v1.run(
            _builders.resolved(), engine=_builders.engine(),
            source=_builders.data_source(), n_repeats=2, announce=spoken.append,
        )

        assert any("Δwithin" in line for line in spoken)

    def test_an_empty_holdout_measures_nothing(self):
        resolution = _builders.nothing_held_out(_builders.resolved())

        summaries = repeat_carry_eval_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert summaries == ()


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
