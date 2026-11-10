from __future__ import annotations

import pytest
import torch

from tests.ports import _builders
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.torch_fast_weights import TorchFastWeights
from ttt.core.config.ttt import TTTConfig
from ttt.core.types import Carry
from ttt.ports.fast_weights import FastWeights

CHUNK = 4
ITEM_TOKENS = 3 * CHUNK


class FastWeightsConformance:
    @pytest.fixture
    def fast_weights(self):
        raise NotImplementedError

    def run_item(self, fast_weights, n_tokens: int = ITEM_TOKENS) -> None:
        raise NotImplementedError

    def staged_carry(self, fast_weights) -> Carry:
        fast_weights.set_mode(evolve=True, session=True)
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
        fast_weights.set_mode(evolve=True, session=True)

        self.run_item(fast_weights)
        fast_weights.advance_carry()

        assert fast_weights.snapshot().layer_indices == _builders.LAYER_INDICES

    def test_a_carry_is_not_promoted_until_advance(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)

        self.run_item(fast_weights)

        assert fast_weights.snapshot().is_empty

    def test_detaching_the_carry_keeps_its_value(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()
        before = fast_weights.snapshot()

        fast_weights.detach_carry()

        after = fast_weights.snapshot()
        assert all(torch.equal(before.get(i), after.get(i)) for i in before.layer_indices)


    def test_reset_carry_clears_it(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()

        fast_weights.reset_carry()

        assert fast_weights.snapshot().is_empty

    def test_a_carry_round_trips_through_install(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)
        self.run_item(fast_weights)
        fast_weights.advance_carry()
        saved = fast_weights.snapshot(to_cpu=True)

        fast_weights.reset_carry()
        fast_weights.install(saved)

        assert fast_weights.state_ratio() > 0.0

    def test_a_snapshot_is_a_copy(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)
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

    def test_the_state_ratio_is_zero_with_nothing_staged(self, fast_weights):
        assert fast_weights.state_ratio() == 0.0

    def test_a_frozen_fast_weight_stages_nothing(self, fast_weights):
        fast_weights.set_mode(evolve=False, session=True)

        self.run_item(fast_weights)
        fast_weights.advance_carry()

        assert fast_weights.snapshot().is_empty

class TestFakeFastWeights(FastWeightsConformance):
    @pytest.fixture
    def fast_weights(self):
        return FakeFastWeights(_builders.LAYER_INDICES, chunk_size=CHUNK)

    def run_item(self, fast_weights, n_tokens: int = ITEM_TOKENS) -> None:
        fast_weights.stage(n_tokens)


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

    def test_a_model_configured_with_no_ttt_layers_is_accepted(self):
        fast_weights = TorchFastWeights(
            _builders.TinyCausalLM(), TTTConfig(layer_indices=())
        )

        assert fast_weights.layer_indices == ()

    def test_a_lora_only_model_snapshots_an_empty_carry(self):
        fast_weights = TorchFastWeights(
            _builders.TinyCausalLM(), TTTConfig(layer_indices=())
        )
        fast_weights.reset_carry()
        fast_weights.advance_carry()

        assert fast_weights.snapshot().deltas == {}

    def test_a_carry_shaped_for_another_model_is_rejected(self, fast_weights):
        wrong = Carry(deltas={_builders.LAYER_INDICES[0]: torch.zeros(1, 2, 2)})

        with pytest.raises(ValueError, match="incompatible with W0"):
            fast_weights.install(wrong)

    def test_the_gate_mean_is_recorded(self, fast_weights):
        fast_weights.set_mode(evolve=True, session=True)

        self.run_item(fast_weights)

        assert fast_weights.gate_mean() is not None
