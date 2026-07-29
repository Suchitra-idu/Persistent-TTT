from __future__ import annotations

import json

import pytest
import torch

from tests.experiments import _builders
from ttt import cli
from ttt.adapters import checkpoint_io
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.experiments import _runtime, plot_pilot, sanity_check_v1


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

    def test_an_explicit_seed_overrides_that(self):
        assert _builders.engine().rng(99).seed == 99

    def test_the_same_seed_draws_the_same_permutation(self):
        engine = _builders.engine()

        assert engine.rng().permutation(10) == engine.rng().permutation(10)

    def test_a_run_with_no_checkpoint_carries_nothing(self):
        engine = _runtime.Engine(
            resolved=_builders.resolved(), tokenizer=None, storage=InMemoryStorage()
        )

        assert engine.carries() == ({}, {})

    def test_a_resumed_run_reads_the_carriers_back(self):
        storage = InMemoryStorage()
        resolution = _builders.resolved(resume_from="step_4", run_name="run-a")
        checkpoint_io.save_carries(
            {"FixtureProse": _builders.carry()},
            storage,
            f"run-a/step_4/{checkpoint_io.CARRIES}",
            n_updates={"FixtureProse": 12},
        )
        engine = _runtime.Engine(
            resolved=resolution, tokenizer=None, storage=storage
        )

        carries, meta = engine.carries()

        assert set(carries) == {"FixtureProse"} and meta["n_updates"] == (
            {"FixtureProse": 12}
        )

    def test_a_missing_checkpoint_is_not_an_error(self):
        resolution = _builders.resolved(resume_from="step_4", run_name="run-a")
        engine = _runtime.Engine(
            resolved=resolution, tokenizer=None, storage=InMemoryStorage()
        )

        assert engine.carries() == ({}, {})

    def test_a_run_with_nowhere_to_write_still_completes(self):
        engine = _runtime.Engine(resolved=_builders.resolved(), tokenizer=None)

        engine.save(step=1, carries={}, n_updates={})

    def test_the_boot_log_names_the_model_and_the_pool(self):
        spoken = []
        engine = _builders.engine()

        _runtime.announce_boot(engine, [], spoken.append)

        assert any("Qwen" in line for line in spoken)

    def test_the_boot_log_counts_the_trainable_parameters(self):
        spoken = []

        _runtime.announce_boot(_builders.engine(), [], spoken.append)

        assert any("trainable" in line for line in spoken)


class TestTrackerSelection:
    def test_wandb_off_still_reports_to_the_console(self):
        from ttt.adapters.console_tracker import ConsoleTracker

        chosen = _runtime.tracker(
            _builders.resolved(wandb_enabled=False), job_type="train"
        )

        assert isinstance(chosen, ConsoleTracker)

    def test_the_console_tracker_respects_the_log_cadence(self):
        chosen = _runtime.tracker(
            _builders.resolved(wandb_enabled=False, log_every=5), job_type="train"
        )

        assert chosen.every == 5


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

    def test_the_input_is_built_on_the_models_device(self):
        model = torch.nn.Linear(4, 4, device="meta")

        assert sanity_check_v1.input_ids([1, 2, 3], model).device.type == "meta"

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


class TestPlotPilot:
    def payload(self):
        from ttt.experiments import compounding_pilot_v1

        rows = compounding_pilot_v1.run(
            _builders.resolved(),
            engine=_builders.engine(),
            source=_builders.data_source(),
            announce=lambda _: None,
        )
        return json.loads(compounding_pilot_v1.payload(_builders.resolved(), rows))

    def test_it_groups_by_regime(self):
        assert set(plot_pilot.series(self.payload(), "nll")) == {"cold", "persist"}

    def test_a_series_is_positions_against_values(self):
        positions, values = plot_pilot.series(self.payload(), "nll")["cold"]

        assert len(positions) == len(values) > 0

    def test_positions_come_back_in_order(self):
        positions, _ = plot_pilot.series(self.payload(), "nll")["persist"]

        assert positions == sorted(positions)

    def test_both_metrics_are_plottable(self):
        assert plot_pilot.series(self.payload(), "ppl")["cold"][1]

    def test_an_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            plot_pilot.series(self.payload(), "loss")

    def test_the_title_names_the_source_when_there_is_one(self):
        from ttt.experiments import compounding_pilot_v1

        rows = compounding_pilot_v1.run(
            _builders.resolved(),
            engine=_builders.engine(),
            source=_builders.data_source(),
            pilot_source="FixtureCode",
            announce=lambda _: None,
        )
        payload = json.loads(compounding_pilot_v1.payload(_builders.resolved(), rows))

        assert "FixtureCode" in plot_pilot.title(payload)

    def test_the_title_counts_sources_when_there_are_several(self):
        assert "sources" in plot_pilot.title(self.payload())


class TestCheckpointPaths:
    def test_what_a_run_saves_is_where_a_resume_reads(self):
        saved = cli.step_path("run-a", 200)
        resumed = cli.resume_path("step_200", "run-a")

        assert saved == resumed
