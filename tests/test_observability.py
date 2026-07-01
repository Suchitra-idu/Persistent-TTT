"""Observability tests: telemetry never crashes training; metric collectors are correct."""

import torch

from observability import Telemetry, gpu_stats, param_health


def test_disabled_telemetry_is_total_noop():
    t = Telemetry(enabled=False, project="x", run_name="x",
                  job_type="train", config={})
    assert t.run is None
    t.log({"a": 1})
    t.alert("a", "b")
    t.finish()


def test_missing_api_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = Telemetry(enabled=True, project="x", run_name="x",
                  job_type="train", config={})
    assert t.run is None
    t.log({"a": 1})
    t.finish()


def test_gpu_stats_empty_without_cuda():
    if not torch.cuda.is_available():
        assert gpu_stats() == {}


def test_param_health_drift_and_norms(module_factory):
    m, mlp, tap = module_factory(randomize=True)
    wdown = [mlp.down_proj.weight]
    init = [p.detach().clone() for p in wdown]
    named = {"wdown": wdown, "lora": [torch.nn.Parameter(torch.ones(2))]}

    h = param_health(named, init, [m])
    assert h["health/wdown_drift_L0"] == 0.0
    assert h["health/w_target_L0"] > 0
    assert h["health/conv_L0"] > 0
    assert h["health/lora_norm"] > 0

    with torch.no_grad():
        wdown[0].mul_(2.0)                    # drift = ||W - W0|| / ||W0|| = 1
    h2 = param_health(named, init, [m])
    assert abs(h2["health/wdown_drift_L0"] - 1.0) < 1e-9
