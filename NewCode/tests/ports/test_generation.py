from __future__ import annotations

import pytest
import torch

from tests.ports import _builders
from ttt.adapters.fake_generation import FakeGeneration
from ttt.adapters.torch_generation import TorchGeneration
from ttt.ports.generation import Generation

PROMPT = _builders.token_ids(6)
SCRIPT = (2, 5, 7)


class GenerationConformance:
    @pytest.fixture
    def generation(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, generation):
        assert isinstance(generation, Generation)

    def test_prefill_returns_one_logits_row(self, generation):
        assert generation.prefill(PROMPT).shape[0] == 1

    def test_prefill_returns_a_finite_row(self, generation):
        assert torch.isfinite(generation.prefill(PROMPT)).all()

    def test_prefill_on_an_empty_prompt_raises(self, generation):
        with pytest.raises(ValueError, match="empty prompt"):
            generation.prefill([])

    def test_step_before_prefill_raises(self, generation):
        with pytest.raises(RuntimeError, match="step before prefill"):
            generation.step(1)

    def test_step_after_prefill_returns_the_same_shape(self, generation):
        prefilled = generation.prefill(PROMPT)

        assert generation.step(0).shape == prefilled.shape

    def test_step_after_a_cache_reset_raises(self, generation):
        generation.prefill(PROMPT)

        generation.reset_cache()

        with pytest.raises(RuntimeError, match="step before prefill"):
            generation.step(0)

    def test_a_reset_cache_can_be_prefilled_again(self, generation):
        generation.prefill(PROMPT)
        generation.reset_cache()

        assert generation.prefill(PROMPT).shape[0] == 1

    def test_resetting_an_unprimed_cache_is_safe(self, generation):
        generation.reset_cache()


class TestFakeGeneration(GenerationConformance):
    @pytest.fixture
    def generation(self):
        return FakeGeneration(SCRIPT)

    def test_the_script_comes_back_as_the_argmax(self, generation):
        emitted = [int(generation.prefill(PROMPT).argmax())]
        emitted += [int(generation.step(0).argmax()) for _ in range(2)]

        assert emitted == list(SCRIPT)

    def test_a_short_script_cycles(self, generation):
        generation.prefill(PROMPT)
        stream = [int(generation.step(0).argmax()) for _ in range(3)]

        assert stream == [5, 7, 2]

    def test_it_records_the_prompt(self, generation):
        generation.prefill(PROMPT)

        assert generation.prompts == [tuple(PROMPT)]

    def test_it_records_the_tokens_it_was_fed(self, generation):
        generation.prefill(PROMPT)
        generation.step(4)

        assert generation.stepped == [4]

    def test_it_counts_cache_resets(self, generation):
        generation.reset_cache()

        assert generation.cache_resets == 1

    def test_an_empty_script_is_rejected(self):
        with pytest.raises(ValueError, match="at least one scripted token"):
            FakeGeneration(())

    def test_a_token_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(ValueError, match="outside vocab"):
            FakeGeneration((99,), vocab_size=8)

    def test_it_feeds_the_stream_fast_weight(self):
        from ttt.adapters.fake_fast_weights import FakeFastWeights

        fast_weights = FakeFastWeights([1], chunk_size=4)
        fast_weights.set_mode(evolve=True, stream=True, session=False)
        generation = FakeGeneration(SCRIPT, fast_weights=fast_weights)

        generation.prefill(PROMPT)

        assert fast_weights.stream_progress() == (len(PROMPT) % 4, 4)


class TestTorchGeneration(GenerationConformance):
    @pytest.fixture
    def generation(self):
        model, _ = _builders.tiny_model()
        return TorchGeneration(model, device="cpu")

    def test_the_cache_grows_by_one_per_step(self, generation):
        generation.prefill(PROMPT)

        generation.step(0)

        assert generation._cache == len(PROMPT) + 1

    def test_a_reset_drops_the_cache(self, generation):
        generation.prefill(PROMPT)

        generation.reset_cache()

        assert generation._cache is None

    def test_it_produces_a_row_per_vocabulary_entry(self, generation):
        assert generation.prefill(PROMPT).shape[1] == _builders.VOCAB
