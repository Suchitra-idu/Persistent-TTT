from __future__ import annotations

import pytest

from ttt.adapters.fake_generation import FakeGeneration
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.app import generate as generate_mod
from ttt.app.generate import MAX_TOKENS, STOP_TOKEN

PROMPT = (1, 2, 3)
GREEDY = dict(temperature=0.0, top_p=1.0, top_k=0)


def run(script=(7, 8, 9), *, max_new_tokens=5, stop_ids=frozenset(), **kwargs):
    generation = FakeGeneration(script, **kwargs)
    completion = generate_mod.generate(
        generation=generation,
        prompt_ids=PROMPT,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        **GREEDY,
    )
    return completion, generation


class TestTokenLoop:
    def test_it_emits_the_tokens_the_model_peaked_on(self):
        completion, _ = run((7, 8, 9), max_new_tokens=3)

        assert completion.token_ids == (7, 8, 9)

    def test_it_prefills_the_prompt_once(self):
        _, generation = run(max_new_tokens=3)

        assert generation.prompts == [PROMPT]

    def test_it_feeds_each_emitted_token_back(self):
        _, generation = run((7, 8, 9), max_new_tokens=3)

        assert generation.stepped == [7, 8, 9]

    def test_it_stops_at_the_token_budget(self):
        completion, _ = run((7,), max_new_tokens=2)

        assert (len(completion.token_ids), completion.stop_reason) == (2, MAX_TOKENS)

    def test_a_zero_budget_emits_nothing(self):
        completion, _ = run(max_new_tokens=0)

        assert completion.token_ids == ()

    def test_a_zero_budget_still_prefills(self):
        _, generation = run(max_new_tokens=0)

        assert generation.prompts == [PROMPT]


class TestStopping:
    def test_a_stop_token_ends_the_completion(self):
        completion, _ = run((7, 8, 9), stop_ids={8})

        assert completion.stop_reason == STOP_TOKEN

    def test_a_stop_token_is_not_emitted(self):
        completion, _ = run((7, 8, 9), stop_ids={8})

        assert completion.token_ids == (7,)

    def test_it_reports_which_token_stopped_it(self):
        completion, _ = run((7, 8, 9), stop_ids={8})

        assert completion.stop_token_id == 8

    def test_a_stop_token_is_not_fed_back(self):
        _, generation = run((7, 8, 9), stop_ids={8})

        assert generation.stepped == [7]

    def test_reaching_the_budget_names_no_stop_token(self):
        completion, _ = run((7,), max_new_tokens=2)

        assert completion.stop_token_id is None


class TestSampling:
    def test_a_seeded_generator_reproduces_its_draw(self):
        import torch

        def sampled(seed):
            generator = torch.Generator()
            generator.manual_seed(seed)
            return generate_mod.generate(
                generation=FakeGeneration((7, 8, 9), vocab_size=32),
                prompt_ids=PROMPT,
                max_new_tokens=4,
                temperature=1.0,
                top_p=1.0,
                generator=generator,
            ).token_ids

        assert sampled(11) == sampled(11)

    def test_different_seeds_can_draw_differently(self):
        import torch

        def sampled(seed):
            generator = torch.Generator()
            generator.manual_seed(seed)
            return generate_mod.generate(
                generation=FakeGeneration(range(32), vocab_size=32),
                prompt_ids=PROMPT,
                max_new_tokens=8,
                temperature=50.0,
                top_p=1.0,
                generator=generator,
            ).token_ids

        assert sampled(1) != sampled(2)


def test_the_fast_weight_sees_every_generated_token():
    fast_weights = FakeFastWeights((1, 3), chunk_size=2)
    fast_weights.set_mode(evolve=True, stream=True, session=False)
    generation = FakeGeneration((7,), fast_weights=fast_weights)

    generate_mod.generate(
        generation=generation, prompt_ids=PROMPT, max_new_tokens=3, **GREEDY
    )

    assert fast_weights.state_ratio(family="stream") > 0.0


def test_stepping_before_a_prefill_is_the_adapter_s_error():
    with pytest.raises(RuntimeError, match="step before prefill"):
        FakeGeneration((7,)).step(1)
