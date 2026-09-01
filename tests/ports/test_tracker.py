from __future__ import annotations

import pytest
import wandb

from ttt.adapters.console_tracker import ConsoleTracker
from ttt.adapters.in_memory_tracker import InMemoryTracker
from ttt.adapters.wandb_tracker import WandbTracker
from ttt.ports.tracker import BUDGET, MICRO_STEP, TRAIN_STEP, Tracker, outside_budget

METRICS = {TRAIN_STEP: 4, "train/loss": 2.5}


def _make_png(tmp_path):
    from PIL import Image

    path = tmp_path / "chart.png"
    Image.new("RGB", (2, 2)).save(path)
    return path


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

    def test_logging_an_image_does_not_raise(self, tracker, tmp_path):
        path = _make_png(tmp_path)

        tracker.log_image("chart", str(path))


class TestInMemoryTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return InMemoryTracker()

    def test_it_records_in_order(self, tracker):
        tracker.log({MICRO_STEP: 0})
        tracker.log({MICRO_STEP: 1})

        assert tracker.values_of(MICRO_STEP) == [0, 1]

    def test_finish_is_observable(self, tracker):
        tracker.finish()

        assert tracker.finished

    def test_it_reports_which_keys_were_logged(self, tracker):
        tracker.log(METRICS)

        assert tracker.keys_logged == frozenset(METRICS)

    def test_it_records_images_in_order(self, tracker, tmp_path):
        path = _make_png(tmp_path)

        tracker.log_image("chart", str(path))

        assert tracker.images == [("chart", str(path))]


class TestConsoleTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return ConsoleTracker()

    def test_it_prints_the_axis_first(self, tracker, capsys):
        tracker.log(METRICS)

        assert capsys.readouterr().out.startswith(f"[{TRAIN_STEP} 4] ")

    def test_it_can_thin_what_it_prints(self, capsys):
        tracker = ConsoleTracker(every=3)

        for _ in range(6):
            tracker.log(METRICS)

        assert capsys.readouterr().out.count("train/loss") == 2

    def test_a_zero_interval_is_rejected(self):
        with pytest.raises(ValueError, match="every must be >= 1"):
            ConsoleTracker(every=0)

    def test_it_prints_the_image_key_and_path(self, tracker, tmp_path, capsys):
        path = _make_png(tmp_path)

        tracker.log_image("chart", str(path))

        assert str(path) in capsys.readouterr().out


class TestWandbTracker(TrackerConformance):
    @pytest.fixture
    def tracker(self):
        return WandbTracker(RecordingRun())

    def test_it_declares_both_x_axes(self):
        run = RecordingRun()

        WandbTracker(run)

        assert run.define_metric_calls >= 2

    def test_it_forwards_what_it_is_given(self, tracker):
        tracker.log(METRICS)

        assert tracker._run.logged == [METRICS]

    def test_an_outage_degrades_to_the_fallback(self, capsys):
        tracker = WandbTracker(ExplodingRun())

        tracker.log(METRICS)

        assert "train/loss" in capsys.readouterr().out

    def test_an_outage_at_finish_degrades_to_the_fallback(self, capsys):
        tracker = WandbTracker(ExplodingRun())

        tracker.finish()

        assert "finished" in capsys.readouterr().out

    def test_it_forwards_an_image_wrapped_for_wandb(self, tracker, tmp_path):
        path = _make_png(tmp_path)

        tracker.log_image("chart", str(path))

        [logged] = tracker._run.logged
        assert isinstance(logged["chart"], wandb.Image)

    def test_an_image_outage_degrades_to_the_fallback(self, tmp_path, capsys):
        tracker = WandbTracker(ExplodingRun())
        path = _make_png(tmp_path)

        tracker.log_image("chart", str(path))

        assert str(path) in capsys.readouterr().out


def test_a_key_d12_cut_is_reported_outside_the_budget():
    assert outside_budget({"health/gate_mean_L0": 1.0}) == ("health/gate_mean_L0",)


def test_a_budgeted_metric_dict_reports_nothing_outside_it():
    assert outside_budget({key: 0.0 for key in BUDGET}) == ()
