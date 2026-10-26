from __future__ import annotations

import math

import pytest

from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import dynamic_eval, relatedness_eval
from ttt.app.dynamic_eval import Setting
from ttt.app.relatedness_eval import LEVELS, RELATED, RESET, SELF, Case
from ttt.core.metrics import ALL_SOURCES

TARGET = (1, 2, 3)


def case(index: int) -> Case:
    return Case(
        index=index, target_title=f"t{index}", target=TARGET,
        primes={level: (10 + i,) for i, level in enumerate(LEVELS)},
        prime_titles={level: level for level in LEVELS},
    )


class FakeAdapter:
    """The target's loss drops by lr * steps after adapting, so the best setting is known."""

    def __init__(self):
        self.bonus = 0.0
        self.calls = []

    def adapt(self, token_ids, *, optimizer, lr, steps):
        self.calls.append(("adapt", tuple(token_ids), optimizer, lr, steps))
        self.bonus = lr * steps if optimizer == "adam" else 0.0

    def restore(self):
        self.calls.append(("restore",))
        self.bonus = 0.0


class FakeCompute:
    def __init__(self, adapter):
        self.adapter = adapter

    def eval_loss(self, token_ids, *, lora=True, score_from=0):
        return 2.0 - self.adapter.bonus


def wired():
    adapter = FakeAdapter()
    return adapter, FakeCompute(adapter)


def test_every_case_is_scored_under_reset_and_every_level():
    adapter, compute = wired()

    rows = dynamic_eval.measure(cases=[case(0), case(1)], compute=compute, adapter=adapter,
                                setting=Setting("adam", 0.1, 1))

    assert [r.arm for r in rows if r.doc_idx == 0] == [RESET, *LEVELS]


def test_the_weights_are_restored_after_every_adaptation():
    adapter, compute = wired()

    dynamic_eval.measure(cases=[case(0)], compute=compute, adapter=adapter, setting=Setting("adam", 0.1, 1))

    kinds = [c[0] for c in adapter.calls]
    assert kinds == ["adapt", "restore"] * len(LEVELS)


def test_each_level_adapts_on_its_own_prime():
    adapter, compute = wired()

    dynamic_eval.measure(cases=[case(0)], compute=compute, adapter=adapter, setting=Setting("adam", 0.1, 1))

    assert [c[1] for c in adapter.calls if c[0] == "adapt"] == [case(0).primes[level] for level in LEVELS]


def test_the_reset_arm_is_scored_before_any_adaptation():
    adapter, compute = wired()

    rows = dynamic_eval.measure(cases=[case(0)], compute=compute, adapter=adapter, setting=Setting("adam", 0.1, 1))

    assert next(r for r in rows if r.arm == RESET).ppl == pytest.approx(math.exp(2.0))


def test_tuning_adapts_on_each_objective_prime():
    adapter, compute = wired()

    dynamic_eval.tune(cases=[case(0)], compute=compute, adapter=adapter, settings=[Setting("sgd", 0.1, 1)])

    assert [c[1] for c in adapter.calls if c[0] == "adapt"] == [case(0).primes[SELF], case(0).primes[RELATED]]


class PrimeAwareAdapter(FakeAdapter):
    """Mild settings help on the related prime, aggressive ones only on the target's own text."""

    def adapt(self, token_ids, *, optimizer, lr, steps):
        self.calls.append(("adapt", tuple(token_ids), optimizer, lr, steps))
        own = tuple(token_ids) == case(0).primes[SELF]
        self.bonus = lr * steps if own else (0.1 if lr <= 0.1 else -0.5)


def test_each_objective_gets_its_own_best_setting():
    adapter = PrimeAwareAdapter()
    compute = FakeCompute(adapter)
    settings = [Setting("adam", 0.1, 1), Setting("adam", 0.3, 3)]

    rows = dynamic_eval.tune(cases=[case(0), case(1)], compute=compute, adapter=adapter, settings=settings)

    assert dynamic_eval.best(rows, SELF) == Setting("adam", 0.3, 3)
    assert dynamic_eval.best(rows, RELATED) == Setting("adam", 0.1, 1)


def test_the_best_setting_has_the_largest_same_document_gain():
    adapter, compute = wired()
    settings = [Setting("sgd", 0.5, 3), Setting("adam", 0.1, 1), Setting("adam", 0.2, 3)]

    rows = dynamic_eval.tune(cases=[case(0), case(1)], compute=compute, adapter=adapter, settings=settings)

    assert dynamic_eval.best(rows) == Setting("adam", 0.2, 3)


def test_the_rows_feed_the_relatedness_estimates():
    adapter, compute = wired()
    rows = dynamic_eval.measure(cases=[case(i) for i in range(4)], compute=compute, adapter=adapter,
                                setting=Setting("adam", 0.1, 1))

    estimates = relatedness_eval.estimates(rows, rng=NumpyRng(0), n_boot=20)

    gain = next(e for e in estimates if e.measure == "gain_self" and e.source == ALL_SOURCES)
    assert gain.interval.point > 0


def test_the_grid_crosses_every_rate_with_every_step_count():
    assert len(dynamic_eval.grid({"adam": (1e-5, 3e-5)}, (3, 10))) == 4


def test_a_setting_round_trips_through_text():
    setting = Setting("adam", 3e-05, 10)

    assert Setting.parse(setting.text) == setting


def test_objective_settings_parse_per_objective():
    parsed = dynamic_eval.parse_objectives("self=adam:3e-05:10;related=adam:1e-05:30")

    assert parsed == {"self": Setting("adam", 3e-05, 10), "related": Setting("adam", 1e-05, 30)}
