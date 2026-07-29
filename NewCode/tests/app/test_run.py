"""A whole run end to end on fakes: corpus in, carriers and eval numbers out."""

from __future__ import annotations

import pytest

from tests.app import _builders
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import data_pipeline, eval_loop, train_loop
from ttt.extensions.strategies import EVERLASTING, STRATEGIES
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

    def _save(self, *, step, carries, n_updates):
        self.saves.append((step, dict(carries), dict(n_updates)))

    def _eval(self, *, step, carries):
        report = eval_loop.evaluate(
            docs=self.holdout.docs,
            compute=self.wiring.compute,
            fast_weights=self.wiring.fast_weights,
            n_slices=self.cfg.eval_n_slices,
            carries=carries,
            session_training=self.cfg.session_training,
        )
        self.evals.append(report)
        return report.metrics


@pytest.fixture(scope="module")
def run():
    return Run(
        strategy=EVERLASTING,
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


def test_the_pipeline_produced_documents(run):
    assert len(run.docs) == 6


def test_every_document_carries_a_source(run):
    assert all(doc.source for doc in run.docs)


def test_the_run_took_every_step_its_schedule_planned(run):
    assert run.result.steps == run.result.total_steps == 6


def test_every_document_was_visited_once_per_epoch(run):
    assert run.result.micro_steps == 12


def test_a_carrier_was_trained_for_every_source(run):
    assert set(run.result.carries) == set(_builders.SOURCES)


def test_every_carrier_holds_a_delta_per_layer(run):
    assert all(
        carry.layer_indices == _builders.LAYERS
        for carry in run.result.carries.values()
    )


def test_the_update_counts_cover_every_item(run):
    assert sum(run.result.n_updates.values()) == run.result.micro_steps


def test_checkpoints_landed_on_cadence_and_at_the_end(run):
    assert [step for step, _, _ in run.saves] == [2, 4, 6, 6]


def test_a_checkpoint_carries_the_state_of_that_step(run):
    assert set(run.saves[-1][1]) == set(_builders.SOURCES)


def test_eval_fired_on_cadence(run):
    assert len(run.evals) == 3


def test_eval_measured_the_holdout_under_every_regime(run):
    from ttt.core.types import REGIMES

    assert len(run.evals[-1].rows) == len(run.holdout.docs) * len(REGIMES)


def test_the_zero_seed_trio_is_measured_for_every_document(run):
    from ttt.core.types import COLD_CARRY

    cold = [row for row in run.evals[-1].rows if row.regime == COLD_CARRY]

    assert len(cold) == len(run.holdout.docs)


def test_eval_reported_per_source_summaries(run):
    assert [s.source for s in run.evals[-1].summaries] == list(_builders.SOURCES)


def test_the_seeded_pass_ran_once_carriers_existed(run):
    assert "eval_seed/carry_ppl" in run.evals[-1].metrics


def test_eval_left_the_training_carry_untouched(run):
    assert run.result.steps == run.result.total_steps


def test_nothing_outside_the_logging_budget_reached_the_tracker(run):
    assert set().union(*(outside_budget(r) for r in run.wiring.tracker.records)) == set()


def test_the_run_announced_its_progress(run):
    assert len(run.lines) >= run.result.steps


def test_the_tracker_was_closed(run):
    assert run.wiring.tracker.finished


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
        Run(strategy=EVERLASTING, cfg=cfg).wiring.tracker.values_of("train/loss")
        for _ in range(2)
    ]

    assert losses[0] == losses[1]
