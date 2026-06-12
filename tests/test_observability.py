"""
Observability tests. The contract under test is "telemetry can never
crash or stall training", plus correctness of the metric collectors
that drive go/no-go decisions (drift, state ratio).
"""

import torch

from conftest import C, D, scan
from observability import (
    Telemetry, gpu_stats, param_health, session_metrics, snapshot_wdown,
)


# ---------------------------------------------------------------- telemetry --
def test_disabled_telemetry_is_total_noop():
    t = Telemetry(enabled=False, project="x", run_name="x",
                  job_type="train", config={})
    assert t.run is None
    t.log({"a": 1})        # must not raise
    t.alert("a", "b")
    t.finish()


def test_missing_api_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = Telemetry(enabled=True, project="x", run_name="x",
                  job_type="train", config={})
    assert t.run is None
    t.log({"a": 1})
    t.finish()


# --------------------------------------------------------------- collectors --
def test_gpu_stats_empty_without_cuda():
    if not torch.cuda.is_available():
        assert gpu_stats() == {}


def test_session_metrics_aggregation():
    assert session_metrics({}) == {}
    out = session_metrics({0: 0.1, 1: 0.3})
    assert out["session/state_ratio_mean"] == 0.2
    assert out["session/state_ratio_max"] == 0.3
    assert out["session/state_ratio_L1"] == 0.3


def test_snapshot_wdown_isolates_from_mutation():
    p = torch.nn.Parameter(torch.ones(3, 3))
    snap = snapshot_wdown([p])
    with torch.no_grad():
        p.add_(5.0)
    assert torch.equal(snap[0], torch.ones(3, 3))


def test_param_health_drift_and_norms(module_factory):
    m, mlp, tap = module_factory(randomize=True)
    wdown = [mlp.down_proj.weight]
    init = snapshot_wdown(wdown)
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
