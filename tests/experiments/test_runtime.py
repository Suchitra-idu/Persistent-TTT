from __future__ import annotations


import torch

from tests.experiments import _builders
from ttt import cli
from ttt.experiments import _runtime, sanity_check_v1


class _PeftShaped(torch.nn.Module):
    """Nests the model one level deeper, the way `get_peft_model` does, so that
    `model.model.layers` stops resolving."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = inner

    def forward(self, **kwargs):
        return self.base_model(**kwargs)


class TestEngine:
    def test_it_seeds_its_rng_from_the_resolved_config(self):
        engine = _builders.engine(_builders.resolved(seed=7))

        assert engine.rng().seed == 7


    def test_the_same_seed_draws_the_same_permutation(self):
        engine = _builders.engine()

        assert engine.rng().permutation(10) == engine.rng().permutation(10)


    def test_a_run_with_nowhere_to_write_still_completes(self):
        engine = _runtime.Engine(resolved=_builders.resolved(), tokenizer=None)

        engine.save(step=1, carries={}, n_updates={})


class TestSanityCheck:
    def test_a_zeroed_fast_weight_contributes_exactly_nothing(self):
        from tests.ports import _builders as port_builders
        from ttt.adapters.torch_fast_weights import TorchFastWeights

        model, cfg = port_builders.tiny_model(chunk_size=4)

        result = sanity_check_v1.run(
            model,
            port_builders.token_ids(20),
            fast_weights=TorchFastWeights(model, cfg),
            announce=lambda _: None,
        )

        assert result.passed

    def test_a_zeroed_fast_weight_is_bit_exact_not_merely_close(self):
        from tests.ports import _builders as port_builders
        from ttt.adapters.torch_fast_weights import TorchFastWeights

        model, cfg = port_builders.tiny_model(chunk_size=4)

        result = sanity_check_v1.run(
            model,
            port_builders.token_ids(20),
            fast_weights=TorchFastWeights(model, cfg),
            announce=lambda _: None,
        )

        assert result.diff_at_zero == 0.0

    def test_a_trained_fast_weight_does_contribute(self):
        from tests.ports import _builders as port_builders
        from ttt.adapters.torch_fast_weights import TorchFastWeights

        model, cfg = port_builders.tiny_model(chunk_size=4)

        result = sanity_check_v1.run(
            model,
            port_builders.token_ids(20),
            fast_weights=TorchFastWeights(model, cfg),
            announce=lambda _: None,
        )

        assert result.diff_at_init > result.diff_at_zero

    def test_it_runs_more_tokens_than_one_chunk(self):
        from tests.ports import _builders as port_builders
        from ttt.adapters.torch_fast_weights import TorchFastWeights

        model, cfg = port_builders.tiny_model(chunk_size=4)

        result = sanity_check_v1.run(
            model,
            port_builders.token_ids(20),
            fast_weights=TorchFastWeights(model, cfg),
            announce=lambda _: None,
        )

        assert result.n_tokens > cfg.chunk_size


    def test_it_finds_the_ttt_modules_through_a_peft_style_wrapper(self):
        from tests.ports import _builders as port_builders
        from ttt.adapters.torch_fast_weights import TorchFastWeights

        model, cfg = port_builders.tiny_model(chunk_size=4)
        wrapped = _PeftShaped(model)

        result = sanity_check_v1.run(
            wrapped,
            port_builders.token_ids(20),
            fast_weights=TorchFastWeights(wrapped, cfg),
            announce=lambda _: None,
        )

        assert result.passed


class TestCheckpointPaths:
    def test_what_a_run_saves_is_where_a_resume_reads(self):
        saved = cli.step_path("run-a", 200)
        resumed = cli.resume_path("step_200", "run-a")

        assert saved == resumed
