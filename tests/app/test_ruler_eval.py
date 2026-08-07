from __future__ import annotations

import pytest

from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.fake_generation import FakeGeneration
from ttt.adapters.fake_tokenizer import BYTE_OFFSET, FakeTokenizer
from ttt.app import ruler_eval
from ttt.core.config.ruler import RulerConfig
from ttt.core.ruler_types import RulerExample
from ttt.core.types import COLD_CARRY, FRESH

LAYERS = (1, 3)


class _FakeTask:
    name = "fake"

    def score(self, prediction: str, example: RulerExample) -> float:
        return 1.0 if example.targets[0] in prediction else 0.0


def _example(prompt: str = "hello", answer_prefix: str = " Answer:") -> RulerExample:
    return RulerExample(
        task="fake",
        length_bucket=128,
        prompt=prompt,
        answer_prefix=answer_prefix,
        targets=("t",),
    )


def _wiring(script=(BYTE_OFFSET + ord("t"),)):
    fast_weights = FakeFastWeights(LAYERS)
    generation = FakeGeneration(script, vocab_size=261, fast_weights=fast_weights)
    return fast_weights, generation, FakeTokenizer()


def _evaluate(fast_weights, generation, tokenizer, cfg, examples):
    return ruler_eval.evaluate(
        examples=examples,
        tasks={"fake": _FakeTask()},
        cfg=cfg,
        generation=generation,
        fast_weights=fast_weights,
        tokenizer=tokenizer,
        announce=lambda _: None,
    )


def test_it_runs_every_regime_for_every_example():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(FRESH, COLD_CARRY), max_new_tokens=1)

    report = _evaluate(
        fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(), _example())}
    )

    assert len(report.results) == 4


def test_evolve_matches_the_regime():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(FRESH, COLD_CARRY), max_new_tokens=1)

    _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(),)})

    modes = [e for e in fast_weights.events if e.startswith("mode(")]
    assert "mode(evolve=False,stream=True,session=False)" in modes
    assert "mode(evolve=True,stream=True,session=False)" in modes


def test_it_resets_stream_before_every_example():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1)

    _evaluate(
        fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(), _example())}
    )

    assert fast_weights.events.count("reset_stream") == 2


def test_cold_carry_evolves_the_stream_state():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1)

    _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(),)})

    assert fast_weights.state_ratio(family="stream") > 0.0


def test_fresh_never_evolves_the_stream_state():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(FRESH,), max_new_tokens=1)

    _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(),)})

    assert fast_weights.state_ratio(family="stream") == 0.0


def test_it_restores_the_carry_it_found():
    fast_weights, generation, tokenizer = _wiring()
    fast_weights.set_mode(evolve=True, stream=False, session=True)
    fast_weights.stage(4)
    fast_weights.advance_carry()
    before = fast_weights.state_ratio(family="carry")
    cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1)

    _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(),)})

    assert fast_weights.state_ratio(family="carry") == pytest.approx(before)


def test_the_report_aggregates_mean_score_per_bucket_and_regime():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1)

    report = _evaluate(
        fast_weights, generation, tokenizer, cfg, {"fake_128": (_example(), _example())}
    )

    assert report.metrics == {f"ruler/fake_128_{COLD_CARRY}": 1.0}


class TestPromptStyle:
    def test_base_style_appends_the_answer_prefix_to_the_prompt(self):
        fast_weights, generation, tokenizer = _wiring()
        cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1, prompt_style="base")
        example = _example(prompt="hello", answer_prefix=" Answer:")

        _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (example,)})

        assert tokenizer.decode(generation.prompts[0]) == "hello Answer:"

    def test_instruct_style_wraps_the_prompt_in_the_chat_template(self):
        fast_weights, generation, tokenizer = _wiring()
        cfg = RulerConfig(
            regimes=(COLD_CARRY,), max_new_tokens=1, prompt_style="instruct"
        )
        example = _example(prompt="hello")

        _evaluate(fast_weights, generation, tokenizer, cfg, {"fake_128": (example,)})

        assert tokenizer.decode(generation.prompts[0]) == tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], add_generation_prompt=True
        )


def test_an_empty_example_map_reports_nothing():
    fast_weights, generation, tokenizer = _wiring()
    cfg = RulerConfig(regimes=(COLD_CARRY,), max_new_tokens=1)

    report = _evaluate(fast_weights, generation, tokenizer, cfg, {})

    assert (report.results, report.metrics) == ((), {})
