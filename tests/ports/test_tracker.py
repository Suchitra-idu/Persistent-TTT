from __future__ import annotations

import pytest
import wandb

from ttt.adapters.console_tracker import ConsoleTracker
from ttt.adapters.in_memory_tracker import InMemoryTracker
from ttt.adapters.wandb_tracker import WandbTracker
from ttt.ports.tracker import BUDGET, MICRO_STEP, TRAIN_STEP, Tracker, outside_budget

METRICS = {TRAIN_STEP: 4, "train/loss": 2.5}


class ExplodingRun:
    """A wandb run that fails every call, which is the case telemetry must survive."""

    def __init__(self) -> None:
        self.define_metric_calls = 0

    def define_metric(self, *args, **kwargs) -> None:
        self.define_metric_calls += 1

    def log(self, metrics) -> None:
        raise wandb.Error("wandb is down")

    def finish(self) -> None:
        raise wandb.Error("wandb is down")


class RecordingRun(ExplodingRun):
    def __init__(self) -> None:
        super().__init__()
        self.logged: list[dict] = []
        self.finished = False

    def log(self, metrics) -> None:
        self.logged.append(dict(metrics))

    def finish(self) -> None:
        self.finished = True


class TrackerConformance:
    @pytest.fixture
    def tracker(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, tracker):
        assert isinstance(tracker, Tracker)

    def test_logging_does_not_raise(self, tracker):
        tracker.log(METRICS)

    def test_logging_an_empty_dict_does_not_raise(self, tracker):
        tracker.log({})

    def test_finishing_does_not_raise(self, tracker):
        tracker.log(METRICS)
        tracker.finish()

    def test_logging_after_finishing_does_not_raise(self, tracker):
        tracker.finish()
        tracker.log(METRICS)

class TestInMemoryTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return InMemoryTracker()

    def test_it_records_in_order(self, tracker):
        tracker.log({MICRO_STEP: 0})
        tracker.log({MICRO_STEP: 1})

        assert tracker.values_of(MICRO_STEP) == [0, 1]


class TestConsoleTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return ConsoleTracker()


class TestWandbTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return WandbTracker(RecordingRun())


    def test_it_forwards_what_it_is_given(self, tracker):
        tracker.log(METRICS)

        assert tracker._run.logged == [METRICS]


def test_a_key_d12_cut_is_reported_outside_the_budget():
    assert outside_budget({"health/gate_mean_L0": 1.0}) == ("health/gate_mean_L0",)


def test_a_budgeted_metric_dict_reports_nothing_outside_it():
    assert outside_budget({key: 0.0 for key in BUDGET}) == ()
