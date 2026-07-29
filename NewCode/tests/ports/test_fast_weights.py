from __future__ import annotations

import pytest
import torch

from tests.ports import _builders
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.torch_fast_weights import TorchFastWeights
from ttt.core.config.ttt import TTTConfig
from ttt.core.types import Carry
from ttt.ports.fast_weights import CARRY, STREAM, FastWeights

CHUNK = 4
ITEM_TOKENS = 3 * CHUNK


class FastWeightsConformance:
    @pytest.fixture
    def fast_weights(self):
        raise NotImplementedError

    def run_item(self, fast_weights, n_tokens: int = ITEM_TOKENS) -> None:
        raise NotImplementedError

    def staged_carry(self, fast_weights) -> Carry:
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()
        staged = fast_weights.snapshot()
        fast_weights.reset_carry()
        return staged

    def test_it_satisfies_the_port(self, fast_weights):
        assert isinstance(fast_weights, FastWeights)

    def test_the_layer_indices_are_the_base_model_ones(self, fast_weights):
        assert fast_weights.layer_indices == _builders.LAYER_INDICES

    def test_nothing_is_carried_before_the_first_item(self, fast_weights):
        assert fast_weights.snapshot().is_empty

    def test_a_session_item_stages_a_carry(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)

        self.run_item(fast_weights)
        fast_weights.advance_carry()

        assert fast_weights.snapshot().layer_indices == _builders.LAYER_INDICES

    def test_a_carry_is_not_promoted_until_advance(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)

        self.run_item(fast_weights)

        assert fast_weights.snapshot().is_empty

    def test_reset_carry_clears_it(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()

        fast_weights.reset_carry()

        assert fast_weights.snapshot().is_empty

    def test_a_carry_round_trips_through_install(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()
        saved = fast_weights.snapshot(to_cpu=True)

        fast_weights.reset_carry()
        fast_weights.install(saved)

        assert fast_weights.state_ratio(family=CARRY) > 0.0

    def test_a_snapshot_is_a_copy(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()
        saved = fast_weights.snapshot()

        self.run_item(fast_weights)
        fast_weights.advance_carry()

        assert not torch.equal(
            saved.get(_builders.LAYER_INDICES[0]),
            fast_weights.snapshot().get(_builders.LAYER_INDICES[0]),
        )

    def test_installing_an_unknown_layer_index_raises(self, fast_weights):
        with pytest.raises(KeyError, match="this model does not have"):
            fast_weights.install(Carry(deltas={999: torch.zeros(1, 1)}))

    def test_an_unknown_state_family_raises(self, fast_weights):
        with pytest.raises(ValueError, match="unknown state family"):
            fast_weights.state_ratio(family="nonsense")

    def test_the_state_ratio_is_zero_with_nothing_staged(self, fast_weights):
        assert fast_weights.state_ratio(family=CARRY) == 0.0

    def test_a_frozen_fast_weight_stages_nothing(self, fast_weights):
        fast_weights.set_mode(evolve=False, stream=False, session=True)

        self.run_item(fast_weights)
        fast_weights.advance_carry()

        assert fast_weights.snapshot().is_empty

    def test_streaming_commits_a_chunk_at_a_time(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)

        self.run_item(fast_weights, CHUNK - 1)
        partial = fast_weights.state_ratio(family=STREAM)
        self.run_item(fast_weights, 1)

        assert (partial, fast_weights.state_ratio(family=STREAM) > 0.0) == (0.0, True)

    def test_stream_progress_reports_the_partial_chunk(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)

        self.run_item(fast_weights, CHUNK + 1)

        assert fast_weights.stream_progress() == (1, CHUNK)

    def test_reset_stream_clears_the_stream_state(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)
        self.run_item(fast_weights, 2 * CHUNK)

        fast_weights.reset_stream()

        assert fast_weights.state_ratio(family=STREAM) == 0.0

    def test_the_two_families_are_independent(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()

        fast_weights.reset_stream()

        assert fast_weights.state_ratio(family=CARRY) > 0.0

    def test_a_seed_can_be_installed_into_the_stream_family(self, fast_weights):
        seed = self.staged_carry(fast_weights)

        fast_weights.install(seed, family=STREAM)

        assert fast_weights.state_ratio(family=STREAM) > 0.0

    def test_a_stream_install_leaves_the_carry_empty(self, fast_weights):
        seed = self.staged_carry(fast_weights)

        fast_weights.install(seed, family=STREAM)

        assert fast_weights.snapshot().is_empty

    def test_installing_into_an_unknown_family_raises(self, fast_weights):
        with pytest.raises(ValueError, match="unknown state family"):
            fast_weights.install(Carry.empty(), family="nonsense")

    def test_decaying_the_stream_scales_it(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)
        self.run_item(fast_weights, 2 * CHUNK)
        before = fast_weights.state_ratio(family=STREAM)

        fast_weights.decay_stream(factor=0.5)

        assert fast_weights.state_ratio(family=STREAM) == pytest.approx(0.5 * before)

    def test_decaying_the_stream_by_one_changes_nothing(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)
        self.run_item(fast_weights, 2 * CHUNK)
        before = fast_weights.state_ratio(family=STREAM)

        fast_weights.decay_stream(factor=1.0)

        assert fast_weights.state_ratio(family=STREAM) == pytest.approx(before)

    def test_decaying_the_stream_leaves_the_carry_alone(self, fast_weights):
        seed = self.staged_carry(fast_weights)
        fast_weights.install(seed)
        before = fast_weights.state_ratio(family=CARRY)

        fast_weights.decay_stream(factor=0.0)

        assert fast_weights.state_ratio(family=CARRY) == pytest.approx(before)

    def test_decaying_an_empty_stream_is_safe(self, fast_weights):
        fast_weights.decay_stream(factor=0.5)

        assert fast_weights.state_ratio(family=STREAM) == 0.0

    def test_reset_v_context_leaves_the_stream_delta_alone(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=True, session=False)
        self.run_item(fast_weights, 2 * CHUNK)
        before = fast_weights.state_ratio(family=STREAM)

        fast_weights.reset_v_context()

        assert fast_weights.state_ratio(family=STREAM) == before


class TestFakeFastWeights(FastWeightsConformance):
    @pytest.fixture
    def fast_weights(self):
        return FakeFastWeights(_builders.LAYER_INDICES, chunk_size=CHUNK)

    def run_item(self, fast_weights, n_tokens: int = ITEM_TOKENS) -> None:
        fast_weights.stage(n_tokens)

    def test_it_records_every_lifecycle_call(self, fast_weights):
        fast_weights.reset_carry()
        fast_weights.advance_carry()

        assert fast_weights.events[-2:] == ["reset_carry", "advance_carry"]

    def test_the_carry_decays_between_items(self):
        fast_weights = FakeFastWeights([1], chunk_size=CHUNK, decay=0.5)
        fast_weights.set_mode(evolve=True, stream=False, session=True)

        for _ in range(2):
            fast_weights.stage(1)
            fast_weights.advance_carry()

        assert float(fast_weights.snapshot().get(1)) == pytest.approx(1.5)


class TestTorchFastWeights(FastWeightsConformance):
    @pytest.fixture
    def fast_weights(self):
        model, cfg = _builders.tiny_model(chunk_size=CHUNK)
        return TorchFastWeights(model, cfg)

    def run_item(self, fast_weights, n_tokens: int = ITEM_TOKENS) -> None:
        ids = torch.tensor([_builders.token_ids(n_tokens, seed=n_tokens)])
        fast_weights.model(input_ids=ids)

    def test_a_model_with_no_ttt_layers_is_rejected(self):
        with pytest.raises(ValueError, match="no InPlaceTTTMLP layers"):
            TorchFastWeights(_builders.TinyCausalLM(), TTTConfig())

    def test_a_carry_shaped_for_another_model_is_rejected(self, fast_weights):
        wrong = Carry(deltas={_builders.LAYER_INDICES[0]: torch.zeros(1, 2, 2)})

        with pytest.raises(ValueError, match="incompatible with W0"):
            fast_weights.install(wrong)

    def test_the_gate_statistics_are_recorded(self, fast_weights):
        fast_weights.set_mode(evolve=True, stream=False, session=True)

        self.run_item(fast_weights)

        assert fast_weights.gate_stats() is not None
