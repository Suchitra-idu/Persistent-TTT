from __future__ import annotations

import math

import pytest
import torch

from tests.ports import _builders
from ttt.adapters.fake_compute import FakeCompute
from ttt.adapters.torch_compute import TorchCompute
from ttt.core.config.train import TrainConfig
from ttt.ports.compute import GROUPS, Compute

IDS = _builders.token_ids(12)
OTHER_IDS = _builders.token_ids(12, seed=99)
RATES = {"lora": 1e-5, "wdown": 3e-5, "new": 2e-5}
CLIP = 10.0
FLOOR = 1e-6


class ComputeConformance:
    @pytest.fixture
    def compute(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, compute):
        assert isinstance(compute, Compute)

    def test_a_loss_is_a_finite_float(self, compute):
        assert math.isfinite(compute.loss(IDS))

    def test_backward_without_a_loss_raises(self, compute):
        with pytest.raises(RuntimeError, match="backward without a preceding loss"):
            compute.backward(scale=1.0)

    def test_a_loss_can_be_backwarded_once(self, compute):
        compute.loss(IDS)

        compute.backward(scale=0.5)

    def test_a_second_backward_on_the_same_loss_raises(self, compute):
        compute.loss(IDS)
        compute.backward(scale=1.0)

        with pytest.raises(RuntimeError):
            compute.backward(scale=1.0)

    def test_a_step_reports_a_norm_for_every_group(self, compute):
        compute.loss(IDS)
        compute.backward(scale=1.0)

        stats = compute.clip_and_step(max_grad_norm=CLIP, learning_rates=RATES)

        assert all(math.isfinite(getattr(stats, name)) for name in GROUPS)

    def test_a_step_reports_a_finite_total_norm(self, compute):
        compute.loss(IDS)
        compute.backward(scale=1.0)

        stats = compute.clip_and_step(max_grad_norm=CLIP, learning_rates=RATES)

        assert math.isfinite(stats.total_norm)

    def test_zero_grad_is_safe_before_any_backward(self, compute):
        compute.zero_grad()

    def test_eval_loss_is_a_finite_float(self, compute):
        assert math.isfinite(compute.eval_loss(IDS))

    def test_eval_loss_does_not_consume_a_pending_backward(self, compute):
        compute.loss(IDS)
        compute.eval_loss(IDS)

        compute.backward(scale=1.0)

    def test_parameter_counts_cover_every_group(self, compute):
        assert set(compute.parameter_counts()) == set(GROUPS)


class TestFakeCompute(ComputeConformance):
    @pytest.fixture
    def compute(self):
        return FakeCompute(losses=(2.0, 3.0))

    def test_scripted_losses_come_back_in_order(self, compute):
        assert [compute.loss(IDS), compute.loss(IDS)] == [2.0, 3.0]

    def test_a_short_script_cycles(self, compute):
        assert [compute.loss(IDS) for _ in range(3)] == [2.0, 3.0, 2.0]

    def test_it_records_what_it_was_asked_to_forward(self, compute):
        compute.loss(IDS)

        assert compute.forwards == [tuple(IDS)]

    def test_it_records_the_accumulation_scale(self, compute):
        compute.loss(IDS)
        compute.backward(scale=0.0625)

        assert compute.backwards == [0.0625]

    def test_it_records_the_learning_rates_it_stepped_with(self, compute):
        compute.clip_and_step(max_grad_norm=CLIP, learning_rates=RATES)

        assert compute.steps == [RATES]

    def test_a_nonfinite_loss_can_be_scripted(self):
        assert math.isnan(FakeCompute(losses=(float("nan"),)).loss(IDS))

    def test_it_stages_a_carry_when_wired_to_fast_weights(self):
        from ttt.adapters.fake_fast_weights import FakeFastWeights

        fast_weights = FakeFastWeights([1, 3])
        fast_weights.set_mode(evolve=True, stream=False, session=True)
        compute = FakeCompute(fast_weights=fast_weights)

        compute.loss(IDS)
        fast_weights.advance_carry()

        assert not fast_weights.snapshot().is_empty

    def test_an_empty_script_is_rejected(self):
        with pytest.raises(ValueError, match="at least one scripted loss"):
            FakeCompute(losses=())


class TestTorchCompute(ComputeConformance):
    @pytest.fixture
    def compute(self):
        model, cfg = _builders.tiny_model(chunk_size=4)
        return TorchCompute(
            model, ttt_cfg=cfg, train_cfg=TrainConfig(), device="cpu"
        )

    def test_the_loss_is_the_next_token_cross_entropy_of_those_tokens(self, compute):
        ids = torch.tensor([IDS])
        logits = compute.model(input_ids=ids).logits
        oracle = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])

        assert compute.loss(IDS) == pytest.approx(float(oracle.detach()), rel=1e-6)

    def test_different_tokens_give_a_different_loss(self, compute):
        assert compute.loss(IDS) != compute.loss(OTHER_IDS)

    def test_the_retained_graph_reaches_the_trainable_parameters(self, compute):
        w_target = compute.model.model.layers[1].mlp.w_target
        compute.loss(IDS)

        compute.backward(scale=1.0)

        assert w_target.grad is not None and w_target.grad.abs().sum() > 0.0

    def test_the_ttt_parameters_land_in_the_new_group(self, compute):
        assert compute.parameter_counts()["new"] > 0

    def test_the_ttt_down_projections_land_in_the_wdown_group(self, compute):
        assert compute.parameter_counts()["wdown"] > 0

    def test_the_total_norm_is_measured_before_clipping(self, compute):
        compute.loss(IDS)
        compute.backward(scale=1e6)

        stats = compute.clip_and_step(max_grad_norm=FLOOR, learning_rates=RATES)

        assert stats.total_norm > 1.0

    def test_the_group_norms_are_measured_before_clipping(self, compute):
        compute.loss(IDS)
        compute.backward(scale=1e6)

        stats = compute.clip_and_step(max_grad_norm=FLOOR, learning_rates=RATES)

        assert stats.new > FLOOR

    def test_a_step_moves_a_trainable_parameter(self, compute):
        before = compute.model.model.layers[1].mlp.w_target.detach().clone()
        compute.loss(IDS)
        compute.backward(scale=1.0)

        compute.clip_and_step(max_grad_norm=CLIP, learning_rates=RATES)

        assert not compute.model.model.layers[1].mlp.w_target.equal(before)

    def test_an_unclassified_trainable_parameter_is_rejected(self):
        model, cfg = _builders.tiny_model()
        model.lm_head.weight.requires_grad_(True)

        with pytest.raises(RuntimeError, match="unclassified trainable parameter"):
            TorchCompute(model, ttt_cfg=cfg, train_cfg=TrainConfig(), device="cpu")
