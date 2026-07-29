"""Manual, gated, never in CI: `make test-gpu-modal` on a real H100.

Everything else in the suite runs on `TinyCausalLM` or fakes. This is the one
tier that touches the actual base model, and it is the last thing to run before
believing a cutover.

The device is not incidental. The base model loads in bfloat16, which needs
sm_80 or newer, so a pre-Ampere card measures nothing worth having even when it
fits. `run_on_modal.py` is the supported path.
"""

from __future__ import annotations

import math
import time

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

MIN_STEPS = 20
DOCS = 40
# The pipeline holds a slice out, so the corpus has to over-supply what the run
# trains on. `test_the_corpus_was_big_enough` is what catches this shrinking.
RAW_DOCS = 4 * DOCS
IDENTITY_TEXT = "Test-time training updates a subset of weights during inference. " * 20


@pytest.fixture(scope="module")
def resolved():
    # `warmup_min_steps` is 10 by default, which over a smoke-length run means
    # every sampled step is still ramping and W_target has not moved off zero.
    return cli.resolve(
        dataset="fixture",
        strategy="everlasting",
        num_epochs=1,
        grad_accum_steps=2,
        min_doc_tokens=64,
        max_seq_len=512,
        warmup_min_steps=2,
        eval_every=0,
        save_every=1000,
        log_every=5,
        wandb_enabled=False,
    )


def say(message: str) -> None:
    """Reaches the terminal only under `-s`, which is why `make test-gpu` sets it."""
    print(f"\n[gpu] {message}", flush=True)


@pytest.fixture(scope="module")
def loaded(resolved):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU; run this tier by hand on a real device")
    say(f"device: {torch.cuda.get_device_name(0)}")
    say(f"loading {resolved.base_model} — a cold cache downloads ~1.5GB here")
    started = time.monotonic()
    model, ttt_cfg = build_model(
        resolved.base_model, ttt_cfg=resolved.ttt, train_cfg=resolved.train
    )
    from ttt.adapters.hf_tokenizer import HfTokenizer

    say(f"model ready in {time.monotonic() - started:.1f}s")
    return model, ttt_cfg, HfTokenizer.from_pretrained(resolved.base_model)


def perturb(model) -> None:
    """W_target is zero-init, so an untouched one agrees perfectly while proving
    nothing. Left zeroed afterwards by the check itself, as it was built."""
    import torch

    from ttt.extensions.mechanism import iter_ttt_modules

    with torch.no_grad():
        for module in iter_ttt_modules(model):
            module.w_target.normal_(std=1e-2)


class TestIdentity:
    @pytest.fixture(scope="class")
    def checked(self, loaded):
        model, ttt_cfg, tokenizer = loaded
        perturb(model)
        return sanity_check_v1.run(
            model,
            tokenizer.encode(IDENTITY_TEXT),
            fast_weights=TorchFastWeights(model, ttt_cfg),
            announce=say,
        )

    def test_a_zeroed_fast_weight_contributes_nothing(self, checked):
        assert checked.passed

    def test_the_check_could_have_failed(self, checked):
        assert checked.diff_at_init > 0.0

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
                for name in ("FixtureProse", "FixtureCode") * (RAW_DOCS // 2)
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

    def test_the_corpus_was_big_enough(self, run):
        result, _ = run

        assert result.sessions >= DOCS

    def test_it_runs_long_enough_for_the_fast_weight_to_leave_zero(self, run):
        """W_target is zero-init, so a run shorter than warmup measures nothing —
        which is what a 2-step run silently did before."""
        result, _ = run

        assert result.total_steps >= MIN_STEPS

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
