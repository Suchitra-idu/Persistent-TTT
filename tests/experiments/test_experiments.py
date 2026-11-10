"""Smoke and reproducibility for every entrypoint, on fixture data and fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.experiments import _builders
from ttt.core import metrics
from ttt.core.types import COLD_CARRY, FRESH
from ttt.experiments import train_v1



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

    @pytest.mark.parametrize("strategy", ["hybrid", "mixed"])
    def test_every_strategy_completes_a_run(self, strategy):
        resolution = _builders.resolved(strategy=strategy)

        result = train_v1.run(
            resolution, engine=_builders.engine(resolution),
            source=_builders.data_source(), announce=lines(),
        )

        assert result.steps > 0


