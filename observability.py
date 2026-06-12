"""
Observability for In-Place TTT training, built on Weights & Biases.

Design rules
  * Telemetry NEVER crashes or stalls a training run. Missing API key,
    network hiccups, wandb errors, all degrade to console-only.
  * Every known failure mode of this project has a metric that exposes
    it. The mapping:

    failure mode                          metric to watch
    ------------------------------------  --------------------------------
    X0 tap / target wiring broken         grad/new ~ 0 while grad/lora healthy
    TTT components not learning           health/w_target_L*, health/conv_L* flat
    unbounded fast weight growth          session/state_ratio_* climbing
    W_down drifting from pretrained       health/wdown_drift_L* large
    LoRA overpowering / dead              health/lora_norm trend
    loss spike / divergence               train/loss + anomaly/nonfinite_count
    eta mis-scaled                        session/state_ratio_* (too big/small)
    throughput regression                 perf/tokens_per_s, perf/sec_per_step
    OOM creep                             gpu/mem_alloc_gb, gpu/mem_reserved_gb
    clipping silently active              train/grad_clip_ratio < 1 persistently

Two x-axes: optimizer steps ("train/step") for aggregates, micro steps
("micro/step") for per-paper signals. wandb's run.define_metric wires
each namespace to its axis so the charts come out right by default.
"""

from __future__ import annotations

import os
import time


class Telemetry:
    """Thin, failure-proof wandb wrapper. All public methods are no-ops
    when disabled, so the training loop never branches on wandb state."""

    def __init__(self, enabled: bool, project: str, run_name: str,
                 job_type: str, config: dict):
        self.run = None
        if not enabled:
            print("telemetry disabled by config")
            return
        if "WANDB_API_KEY" not in os.environ:
            print("WANDB_API_KEY not set, telemetry degrades to console. "
                  "Create it with  modal secret create wandb "
                  "WANDB_API_KEY=...  and attach to the function.")
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project,
                name=f"{run_name}-{time.strftime('%m%d-%H%M')}",
                job_type=job_type,
                config=config,
            )
            # Aggregates plot against optimizer steps, per-paper signals
            # against micro steps; everything else follows train/step.
            self.run.define_metric("train/step")
            self.run.define_metric("micro/step")
            self.run.define_metric("micro/*", step_metric="micro/step")
            for ns in ("train/*", "grad/*", "session/*", "health/*",
                       "perf/*", "gpu/*", "anomaly/*"):
                self.run.define_metric(ns, step_metric="train/step")
        except Exception as e:               # never kill training for telemetry
            print(f"wandb init failed ({e}), continuing without telemetry")
            self.run = None

    def log(self, metrics: dict):
        if self.run is None:
            return
        try:
            self.run.log(metrics)
        except Exception as e:
            print(f"wandb log failed ({e}), continuing")

    def alert(self, title: str, text: str):
        """Push notification for anomalies (needs Scriptable Alerts
        enabled in W&B user settings; harmless otherwise)."""
        if self.run is None:
            return
        try:
            import wandb

            self.run.alert(title=title, text=text,
                           level=wandb.AlertLevel.WARN)
        except Exception:
            pass

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# ===========================================================================
# Metric collectors. Pure functions, each returns a flat dict ready for
# Telemetry.log, namespaced per the table in the module docstring.
# ===========================================================================
def gpu_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 2**30,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "gpu/mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def snapshot_wdown(wdown_params: list) -> list:
    """Clone the TTT-layer W_down initial values (bf16, ~600MB on GPU)
    so drift from the pretrained state can be tracked over training."""
    return [p.detach().clone() for p in wdown_params]


def param_health(named_groups: dict, wdown_init: list,
                 ttt_modules: list) -> dict:
    """Heavier health metrics, intended for every param_log_every steps.
    Flat learning curves on w_target/conv after warmup mean the new
    components are not training; large wdown_drift means the fast weight
    initial state is being pushed far from pretrained Qwen3-8B."""
    import torch

    out = {}
    for i, (p, p0) in enumerate(zip(named_groups["wdown"], wdown_init)):
        out[f"health/wdown_drift_L{i}"] = float(
            (p.detach() - p0).norm() / p0.norm()
        )
    for i, m in enumerate(ttt_modules):
        out[f"health/w_target_L{i}"] = float(m.w_target.detach().norm())
        out[f"health/conv_L{i}"] = float(m.target_conv.weight.detach().norm())
    out["health/lora_norm"] = float(torch.sqrt(sum(
        p.detach().float().pow(2).sum() for p in named_groups["lora"]
    )))
    return out


def session_metrics(state_norms: dict) -> dict:
    out = {f"session/state_ratio_L{i}": v for i, v in state_norms.items()}
    if state_norms:
        out["session/state_ratio_mean"] = (
            sum(state_norms.values()) / len(state_norms)
        )
        out["session/state_ratio_max"] = max(state_norms.values())
    return out
