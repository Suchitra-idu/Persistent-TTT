"""A whole run end to end on fakes: corpus in, carriers and eval numbers out."""

from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import data_pipeline, eval_loop, train_loop
from ttt.extensions.strategies import HYBRID, STRATEGIES
from ttt.ports.tracker import outside_budget

CORPUS = ("alpha", "beta") * 8


class Run:
    def __init__(self, *, strategy, cfg, limit_docs=6):
        self.wiring = _builders.Wiring(losses=(1.4, 1.2, 1.0, 0.8))
        self.cfg = cfg
        source = _builders.data_source(CORPUS)
        self.data = data_pipeline.load(
            source=source,
            spec=_builders.SPEC,
            cfg=cfg,
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
            limit_docs=limit_docs,
            weights=_builders.WEIGHTS,
        )
        self.docs = data_pipeline.documents(self.data)
        self.holdout = data_pipeline.holdout(
            source=source,
            spec=_builders.SPEC,
            cfg=cfg,
            rng=ScriptedRng(),
            tokenizer=FakeTokenizer(),
        )
        self.saves = []
        self.evals = []
        self.lines = []
        self.result = train_loop.train(
            docs=self.docs,
            compute=self.wiring.compute,
            fast_weights=self.wiring.fast_weights,
            tracker=self.wiring.tracker,
            strategy=strategy,
            cfg=cfg,
            rng=self.wiring.rng,
            checkpoint=self._save,
            evaluate=self._eval,
            announce=self.lines.append,
        )

    def _save(self, *, step):
        self.saves.append(step)

    def _eval(self, *, step):
        report = eval_loop.evaluate(
            docs=self.holdout.docs,
            compute=self.wiring.compute,
            fast_weights=self.wiring.fast_weights,
            n_slices=self.cfg.eval_n_slices,
        )
        self.evals.append(report)
        return report.metrics


@pytest.fixture(scope="module")
def run():
    return Run(
        strategy=HYBRID,
        cfg=_builders.config(
            num_epochs=2,
            grad_accum_steps=2,
            log_every=1,
            save_every=2,
            eval_every=2,
            eval_n_slices=2,
            eval_n_docs_per_source=1,
        ),
    )


def test_the_run_took_every_step_its_schedule_planned(run):
    assert run.result.steps == run.result.total_steps == 6


def test_every_document_was_visited_once_per_epoch(run):
    assert run.result.micro_steps == 12


def test_checkpoints_landed_on_cadence_and_at_the_end(run):
    assert run.saves == [2, 4, 6, 6]


def test_eval_fired_on_cadence(run):
    assert len(run.evals) == 3


def test_the_zero_seed_trio_is_measured_for_every_document(run):
    from ttt.core.types import COLD_CARRY

    cold = [row for row in run.evals[-1].rows if row.regime == COLD_CARRY]

    assert len(cold) == len(run.holdout.docs)


def test_eval_reported_per_source_summaries(run):
    assert [s.source for s in run.evals[-1].summaries] == list(_builders.SOURCES)


def test_eval_left_the_training_carry_untouched(run):
    assert run.result.steps == run.result.total_steps


def test_nothing_outside_the_logging_budget_reached_the_tracker(run):
    assert set().union(*(outside_budget(r) for r in run.wiring.tracker.records)) == set()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_registered_strategy_completes_a_run(name):
    completed = Run(
        strategy=STRATEGIES[name],
        cfg=_builders.config(strategy=name, num_epochs=1, eval_every=1),
        limit_docs=4,
    )

    assert completed.result.steps > 0


def test_the_same_seed_produces_the_same_run():
    cfg = _builders.config(num_epochs=1)
    losses = [
        Run(strategy=HYBRID, cfg=cfg).wiring.tracker.values_of("train/loss")
        for _ in range(2)
    ]

    assert losses[0] == losses[1]
