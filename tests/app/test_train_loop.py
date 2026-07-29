from __future__ import annotations

import math

import pytest

from tests.app import _builders
from ttt.app import train_loop
from ttt.extensions.strategies import EVERLASTING, Hybrid
from ttt.ports.tracker import MICRO_STEP, TRAIN_STEP, outside_budget

NAN = float("nan")

# Long enough that the length gate fires, so a session has more than one item.
SLICING_HYBRID = Hybrid(
    name="test-hybrid",
    carry_min_tokens=8,
    slice_min_tokens=2,
    slices_min=2,
    slices_max=3,
)


def run(
    *,
    n_docs: int = 4,
    losses=(1.0,),
    strategy=EVERLASTING,
    cfg=None,
    carries=None,
    checkpoint=None,
    evaluate=None,
    announce=None,
    docs=None,
):
    wiring = _builders.Wiring(losses=losses)
    result = train_loop.train(
        docs=docs if docs is not None else _builders.docs(n_docs),
        compute=wiring.compute,
        fast_weights=wiring.fast_weights,
        tracker=wiring.tracker,
        strategy=strategy,
        cfg=cfg or _builders.config(),
        rng=wiring.rng,
        carries=carries,
        checkpoint=checkpoint,
        evaluate=evaluate,
        announce=announce or (lambda _: None),
    )
    return result, wiring


class Recorder:
    def __init__(self, metrics=None):
        self.calls = []
        self._metrics = metrics or {"eval/carry_ppl": 2.0}

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._metrics


class TestAccumulation:
    def test_the_boundary_fires_at_exactly_grad_accum_steps(self):
        result, wiring = run(n_docs=4)

        assert len(wiring.compute.steps) == result.steps == 2

    def test_a_partial_final_window_does_not_step(self):
        _, wiring = run(n_docs=3)

        assert len(wiring.compute.steps) == 1

    def test_every_item_is_one_forward(self):
        _, wiring = run(n_docs=4)

        assert len(wiring.compute.forwards) == 4

    def test_the_backward_scale_is_one_over_the_window(self):
        _, wiring = run(n_docs=4)

        assert wiring.compute.backwards == [0.5] * 4

    def test_gradients_are_cleared_after_every_step(self):
        _, wiring = run(n_docs=4)

        assert wiring.compute.zero_grads == 2

    def test_it_clips_at_the_configured_norm(self):
        _, wiring = run(n_docs=4)

        assert wiring.compute.clipped_at == [_builders.config().max_grad_norm] * 2

    def test_the_token_count_covers_every_item(self):
        result, _ = run(n_docs=4)

        assert result.total_tokens == 4 * 8


class TestNonfiniteLoss:
    def test_it_skips_the_backward(self):
        _, wiring = run(n_docs=4, losses=(1.0, NAN, 1.0, 1.0))

        assert len(wiring.compute.backwards) == 3

    def test_it_still_advances_the_carry(self):
        _, wiring = run(n_docs=4, losses=(1.0, NAN, 1.0, 1.0))

        assert wiring.fast_weights.events.count("advance_carry") == 4

    def test_it_still_counts_a_micro_step(self):
        result, _ = run(n_docs=4, losses=(1.0, NAN, 1.0, 1.0))

        assert result.micro_steps == 4

    def test_it_does_not_defer_the_accumulation_boundary(self):
        result, _ = run(n_docs=4, losses=(1.0, NAN, 1.0, 1.0))

        assert result.steps == 2

    def test_it_is_counted(self):
        result, _ = run(n_docs=4, losses=(1.0, NAN, 1.0, 1.0))

        assert result.nonfinite == 1

    def test_a_window_with_no_finite_loss_does_not_step(self):
        _, wiring = run(n_docs=4, losses=(NAN,))

        assert wiring.compute.steps == []

    def test_a_window_with_no_finite_loss_still_clears_the_gradients(self):
        _, wiring = run(n_docs=4, losses=(NAN,))

        assert wiring.compute.zero_grads == 2

    def test_the_loss_is_logged_as_it_was_measured(self):
        _, wiring = run(n_docs=2, losses=(NAN,))

        assert all(math.isnan(v) for v in wiring.tracker.values_of("micro/doc_loss"))


class TestCarryLifecycle:
    def test_the_carry_resets_at_every_session_boundary(self):
        result, wiring = run(n_docs=4)

        assert wiring.fast_weights.events.count("reset_carry") == result.sessions == 4

    def test_the_carry_advances_once_per_item(self):
        _, wiring = run(n_docs=4)

        assert wiring.fast_weights.events.count("advance_carry") == 4

    def test_a_multi_item_session_advances_once_per_item(self):
        result, wiring = run(n_docs=2, strategy=SLICING_HYBRID)

        assert (result.sessions, wiring.fast_weights.events.count("advance_carry")) == (
            2,
            6,
        )

    def test_a_session_scoped_strategy_installs_no_carrier(self):
        _, wiring = run(n_docs=4, strategy=SLICING_HYBRID)

        assert "install(carry)" not in wiring.fast_weights.events

    def test_a_session_scoped_strategy_snapshots_no_carrier(self):
        result, _ = run(n_docs=4, strategy=SLICING_HYBRID)

        assert result.carries == {}


class TestEverlastingCarry:
    def test_it_snapshots_a_carrier_per_source(self):
        result, _ = run(n_docs=4)

        assert set(result.carries) == set(_builders.SOURCES)

    def test_a_source_seen_for_the_first_time_starts_from_zero(self):
        _, wiring = run(n_docs=2)

        assert wiring.fast_weights.events.count("install(carry)") == 0

    def test_a_source_seen_again_starts_from_its_carrier(self):
        _, wiring = run(n_docs=4)

        assert wiring.fast_weights.events.count("install(carry)") == 2

    def test_a_resumed_carrier_is_installed_on_the_first_session(self):
        _, wiring = run(n_docs=2, carries={"alpha": _builders.carry()})
        events = wiring.fast_weights.events

        assert events[: events.index("advance_carry")][-2:] == [
            "reset_carry",
            "install(carry)",
        ]

    def test_the_carrier_is_installed_before_the_forward(self):
        _, wiring = run(n_docs=2, carries={"alpha": _builders.carry()})
        events = wiring.fast_weights.events

        assert events.index("install(carry)") < events.index("advance_carry")

    def test_it_counts_updates_per_source(self):
        result, _ = run(n_docs=4)

        assert result.n_updates == {"alpha": 2, "beta": 2}

    def test_the_snapshot_is_taken_after_the_carry_advanced(self):
        result, _ = run(n_docs=2)

        assert not result.carries["alpha"].is_empty


class TestLearningRate:
    def test_it_follows_the_warmup_cosine(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.values_of("train/lr_lora") == pytest.approx(
            [1e-5, 0.5e-5]
        )

    def test_the_rate_logged_is_the_rate_stepped_with(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.values_of("train/lr_lora") == [
            step["lora"] for step in wiring.compute.steps
        ]

    def test_every_group_gets_its_own_rate(self):
        _, wiring = run(n_docs=4)
        cfg = _builders.config()

        assert wiring.compute.steps[0] == pytest.approx(
            {"lora": cfg.lr_lora, "wdown": cfg.lr_wdown, "new": cfg.lr_new_modules}
        )

    def test_warmup_holds_the_rate_below_its_peak(self):
        cfg = _builders.config(warmup_min_steps=2)
        _, wiring = run(n_docs=4, cfg=cfg)

        assert wiring.tracker.values_of("train/lr_lora")[0] == 0.0


class TestLogging:
    def test_nothing_outside_the_budget_is_logged(self):
        _, wiring = run(n_docs=4)

        assert [outside_budget(record) for record in wiring.tracker.records] == [
            () for _ in wiring.tracker.records
        ]

    def test_the_micro_axis_advances_once_per_item(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.values_of(MICRO_STEP) == [1, 2, 3, 4]

    def test_the_train_axis_advances_once_per_step(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.values_of(TRAIN_STEP) == [1, 2]

    def test_the_state_ratio_is_logged_every_item(self):
        _, wiring = run(n_docs=4)

        assert len(wiring.tracker.values_of("micro/state_ratio_mean")) == 4

    def test_an_unclipped_step_reports_a_ratio_of_one(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.values_of("train/grad_clip_ratio") == [1.0, 1.0]

    def test_the_tracker_is_finished(self):
        _, wiring = run(n_docs=4)

        assert wiring.tracker.finished

    def test_progress_is_announced_on_cadence(self):
        lines = []
        run(n_docs=4, cfg=_builders.config(log_every=1), announce=lines.append)

        assert len(lines) == 2


class TestCheckpointing:
    def test_it_saves_on_cadence(self):
        checkpoint = Recorder()
        run(n_docs=4, cfg=_builders.config(save_every=1), checkpoint=checkpoint)

        assert [call["step"] for call in checkpoint.calls] == [1, 2, 2]

    def test_it_saves_once_at_the_end_of_a_run_with_no_cadence_hit(self):
        checkpoint = Recorder()
        result, _ = run(n_docs=4, checkpoint=checkpoint)

        assert [call["step"] for call in checkpoint.calls] == [result.steps]

    def test_it_hands_over_the_live_carriers(self):
        checkpoint = Recorder()
        run(n_docs=4, checkpoint=checkpoint)

        assert set(checkpoint.calls[-1]["carries"]) == set(_builders.SOURCES)

    def test_it_hands_over_the_update_counts(self):
        checkpoint = Recorder()
        run(n_docs=4, checkpoint=checkpoint)

        assert checkpoint.calls[-1]["n_updates"] == {"alpha": 2, "beta": 2}


class TestInLoopEval:
    def test_it_fires_on_cadence(self):
        evaluate = Recorder()
        run(n_docs=4, cfg=_builders.config(eval_every=1), evaluate=evaluate)

        assert [call["step"] for call in evaluate.calls] == [1, 2]

    def test_it_does_not_fire_when_disabled(self):
        evaluate = Recorder()
        run(n_docs=4, cfg=_builders.config(eval_every=0), evaluate=evaluate)

        assert evaluate.calls == []

    def test_its_metrics_land_on_the_train_axis(self):
        evaluate = Recorder()
        _, wiring = run(n_docs=4, cfg=_builders.config(eval_every=2), evaluate=evaluate)

        assert {"eval/carry_ppl", TRAIN_STEP} <= set(wiring.tracker.records[-1])

    def test_it_sees_the_carriers_trained_so_far(self):
        evaluate = Recorder()
        run(n_docs=4, cfg=_builders.config(eval_every=2), evaluate=evaluate)

        assert set(evaluate.calls[-1]["carries"]) == set(_builders.SOURCES)

    def test_its_result_is_announced(self):
        lines = []
        run(
            n_docs=4,
            cfg=_builders.config(eval_every=2),
            evaluate=Recorder(),
            announce=lines.append,
        )

        assert any("[eval]" in line for line in lines)


class TestEpochs:
    def test_every_epoch_revisits_every_document(self):
        _, wiring = run(n_docs=4, cfg=_builders.config(num_epochs=2))

        assert len(wiring.compute.forwards) == 8

    def test_the_schedule_is_sized_for_every_epoch(self):
        result, _ = run(n_docs=4, cfg=_builders.config(num_epochs=2))

        assert result.total_steps == 4

    def test_the_step_count_never_outruns_the_schedule(self):
        result, _ = run(n_docs=5, cfg=_builders.config(num_epochs=3))

        assert result.steps <= result.total_steps

    def test_zero_epochs_does_nothing(self):
        result, wiring = run(n_docs=4, cfg=_builders.config(num_epochs=0))

        assert (result.steps, wiring.compute.forwards) == (0, [])


def test_an_empty_pool_trains_nothing():
    result, _ = run(docs=())

    assert (result.steps, result.sessions) == (0, 0)


@pytest.mark.parametrize("strategy", [EVERLASTING, SLICING_HYBRID])
def test_a_complete_run_stays_inside_the_logging_budget(strategy):
    cfg = _builders.config(num_epochs=2, save_every=1, eval_every=1, log_every=1)
    _, wiring = run(
        n_docs=6,
        strategy=strategy,
        cfg=cfg,
        checkpoint=Recorder(),
        evaluate=Recorder(),
    )

    assert set().union(*wiring.tracker.records) <= _budget_keys()


def _budget_keys():
    from ttt.ports.tracker import BUDGET

    return set(BUDGET)
