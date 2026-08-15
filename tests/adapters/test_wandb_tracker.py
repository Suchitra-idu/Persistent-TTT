"""WandbTracker's degrade-to-console path: silent forever was the bug."""

from __future__ import annotations

import pytest
import wandb

from ttt.adapters.console_tracker import ConsoleTracker
from ttt.adapters.wandb_tracker import WandbTracker


class FakeRun:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.defined: list[tuple] = []
        self.logged: list[dict] = []
        self.finished = False

    def define_metric(self, *args, **kwargs) -> None:
        self.defined.append((args, kwargs))

    def log(self, metrics: dict) -> None:
        if self.fails:
            raise wandb.errors.CommError("network is down")
        self.logged.append(metrics)

    def finish(self) -> None:
        if self.fails:
            raise wandb.errors.CommError("network is down")
        self.finished = True


def test_a_healthy_run_never_touches_the_fallback():
    fallback = ConsoleTracker()
    tracker = WandbTracker(FakeRun(), fallback=fallback)

    tracker.log({"train/step": 1, "train/loss": 0.5})

    assert fallback._seen == 0


def test_a_failing_run_falls_back_without_raising():
    tracker = WandbTracker(FakeRun(fails=True), fallback=ConsoleTracker())

    tracker.log({"train/step": 1, "train/loss": 0.5})


def test_a_failing_run_warns_exactly_once(capsys):
    tracker = WandbTracker(FakeRun(fails=True), fallback=ConsoleTracker())

    tracker.log({"train/step": 1})
    tracker.log({"train/step": 2})
    tracker.log({"train/step": 3})

    warnings = [
        line for line in capsys.readouterr().err.splitlines() if "wandb logging failed" in line
    ]
    assert len(warnings) == 1


def test_the_warning_names_the_exception(capsys):
    tracker = WandbTracker(FakeRun(fails=True), fallback=ConsoleTracker())

    tracker.log({"train/step": 1})

    assert "CommError" in capsys.readouterr().err


def test_a_failing_finish_also_warns_once(capsys):
    tracker = WandbTracker(FakeRun(fails=True), fallback=ConsoleTracker())

    tracker.finish()

    assert "wandb logging failed" in capsys.readouterr().err


def test_a_programmer_bug_still_surfaces():
    class BrokenRun(FakeRun):
        def log(self, metrics: dict) -> None:
            raise TypeError("not a wandb problem")

    tracker = WandbTracker(BrokenRun(), fallback=ConsoleTracker())

    with pytest.raises(TypeError):
        tracker.log({"train/step": 1})
