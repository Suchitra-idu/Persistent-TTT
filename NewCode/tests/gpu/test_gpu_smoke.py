"""Manual, gated, never in CI: `make test-gpu` on a real H100.

Everything else in the suite runs on `TinyCausalLM` or fakes. This is the one
tier that touches the actual base model, and it is the last thing to run before
believing a cutover.
"""

from __future__ import annotations

import math

import pytest

from ttt import cli
from ttt.adapters.fake_data_source import FakeDataSource
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.in_memory_tracker import InMemoryTracker
from ttt.adapters.model_builder import build_model
from ttt.adapters.torch_compute import TorchCompute
from ttt.adapters.torch_fast_weights import TorchFastWeights
from ttt.app import data_pipeline, train_loop
from ttt.experiments import sanity_check_v1
from ttt.ports.tracker import outside_budget

pytestmark = pytest.mark.gpu

STEPS = 20
DOCS = 8
IDENTITY_TEXT = "Test-time training updates a subset of weights during inference. " * 20


@pytest.fixture(scope="module")
def resolved():
    return cli.resolve(
        dataset="fixture",
        strategy="everlasting",
        num_epochs=1,
        grad_accum_steps=2,
        min_doc_tokens=64,
        max_seq_len=512,
        eval_every=0,
        save_every=1000,
        log_every=5,
        wandb_enabled=False,
    )


@pytest.fixture(scope="module")
def loaded(resolved):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU; run this tier by hand on a real device")
    model, ttt_cfg = build_model(
        resolved.base_model, ttt_cfg=resolved.ttt, train_cfg=resolved.train
    )
    from ttt.adapters.hf_tokenizer import HfTokenizer

    return model, ttt_cfg, HfTokenizer.from_pretrained(resolved.base_model)


class TestIdentity:
    def test_a_zeroed_fast_weight_contributes_nothing(self, loaded):
        model, ttt_cfg, tokenizer = loaded

        result = sanity_check_v1.run(
            model,
            tokenizer.encode(IDENTITY_TEXT),
            fast_weights=TorchFastWeights(model, ttt_cfg),
        )

        assert result.passed

    def test_the_check_runs_past_a_chunk_boundary(self, loaded):
        _, ttt_cfg, tokenizer = loaded

        assert len(tokenizer.encode(IDENTITY_TEXT)) > ttt_cfg.chunk_size


class TestTrainingRun:
    @pytest.fixture(scope="class")
    def run(self, resolved, loaded):
        model, ttt_cfg, tokenizer = loaded
        source = FakeDataSource(
            {"fixture": [
                {"text": "the quick brown fox " * 200,
                 "meta": {"redpajama_set_name": name}}
                for name in ("FixtureProse", "FixtureCode") * (DOCS // 2)
            ]}
        )
        docs = data_pipeline.documents(
            data_pipeline.load(
                source=source,
                spec=resolved.spec,
                cfg=resolved.train,
                rng=_rng(resolved.train.seed),
                tokenizer=tokenizer,
                weights=None,
            )
        )
        tracker = InMemoryTracker()
        result = train_loop.train(
            docs=docs[:DOCS],
            compute=TorchCompute(model, ttt_cfg=ttt_cfg, train_cfg=resolved.train),
            fast_weights=TorchFastWeights(model, ttt_cfg),
            tracker=tracker,
            strategy=resolved.strategy,
            cfg=resolved.train,
            rng=_rng(resolved.train.seed),
        )
        return result, tracker

    def test_it_takes_the_steps_it_planned(self, run):
        result, _ = run

        assert result.steps == result.total_steps > 0

    def test_no_loss_diverged(self, run):
        _, tracker = run

        assert all(math.isfinite(v) for v in tracker.values_of("micro/doc_loss"))

    def test_it_trained_a_carrier_per_source(self, run):
        result, _ = run

        assert set(result.carries) == {"FixtureProse", "FixtureCode"}

    def test_the_fast_weight_actually_moved(self, run):
        _, tracker = run

        assert max(tracker.values_of("micro/state_ratio_mean")) > 0.0

    def test_the_gradients_were_finite(self, run):
        _, tracker = run

        assert all(math.isfinite(v) for v in tracker.values_of("grad/wdown"))

    def test_nothing_outside_the_logging_budget_was_emitted(self, run):
        _, tracker = run

        assert not set().union(*(outside_budget(r) for r in tracker.records))


class TestCheckpointRoundTrip:
    def test_a_checkpoint_reloads_into_a_fresh_model(self, resolved, loaded):
        import torch

        from ttt.adapters import checkpoint_io

        model, ttt_cfg, _ = loaded
        storage = InMemoryStorage()
        checkpoint_io.save_ttt_params(
            model, storage, "ttt_params.pt", layer_indices=ttt_cfg.layer_indices
        )
        fresh, _ = build_model(
            resolved.base_model, ttt_cfg=resolved.ttt, train_cfg=resolved.train
        )

        checkpoint_io.load_ttt_params(fresh, storage, "ttt_params.pt")

        assert torch.equal(
            fresh.base_model.model.model.layers[ttt_cfg.layer_indices[0]].mlp.w_target,
            model.base_model.model.model.layers[ttt_cfg.layer_indices[0]].mlp.w_target,
        )


def _rng(seed: int):
    from ttt.adapters.numpy_rng import NumpyRng

    return NumpyRng(seed)
