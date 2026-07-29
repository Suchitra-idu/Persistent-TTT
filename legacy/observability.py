"""Observability for In-Place TTT training, built on Weights & Biases.

Telemetry never crashes or stalls a run; failures degrade to console-only.
Two x-axes: optimizer steps ("train/step") for aggregates, micro steps
("micro/step") for per-paper signals.
"""

from __future__ import annotations

import os
import time
import traceback


_GB = 2 ** 30


def _wandb_errors():
    """The exception classes we consider 'expected wandb failures' to swallow.
    Falls back to (Exception,) if wandb isn't importable yet."""
    try:
        import wandb
        return (wandb.Error,)
    except Exception:
        return (Exception,)


class Telemetry:
    """Thin, failure-proof wandb wrapper; all public methods are no-ops when disabled."""

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
            self.run.define_metric("train/step")
            self.run.define_metric("micro/step")
            self.run.define_metric("micro/*", step_metric="micro/step")
            for ns in ("train/*", "grad/*", "session/*", "health/*",
                       "perf/*", "gpu/*", "anomaly/*"):
                self.run.define_metric(ns, step_metric="train/step")
        except _wandb_errors() as e:
            # Never kill training for telemetry. Broader Exception hides
            # programmer bugs (AttributeError, TypeError), so we scope to
            # wandb's own errors.
            print(f"wandb init failed ({e}), continuing without telemetry")
            self.run = None

    def log(self, metrics: dict):
        if self.run is None:
            return
        try:
            self.run.log(metrics)
        except _wandb_errors() as e:
            print(f"wandb log failed ({e}), continuing")

    def alert(self, title: str, text: str):
        """Push notification (needs Scriptable Alerts enabled in W&B settings)."""
        if self.run is None:
            return
        try:
            import wandb

            self.run.alert(title=title, text=text,
                           level=wandb.AlertLevel.WARN)
        except _wandb_errors():
            traceback.print_exc()

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except _wandb_errors():
                traceback.print_exc()


def gpu_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / _GB,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / _GB,
        "gpu/mem_peak_gb": torch.cuda.max_memory_allocated() / _GB,
    }


def param_health(named_groups: dict, wdown_init: list,
                 ttt_modules: list) -> dict:
    """Heavier health metrics, intended for every param_log_every steps."""
    import torch

    out = {}
    for i, (p, p0) in enumerate(zip(named_groups["wdown"], wdown_init)):
        out[f"health/wdown_drift_L{i}"] = float(
            (p.detach() - p0).norm() / p0.norm()
        )
    for i, m in enumerate(ttt_modules):
        out[f"health/w_target_L{i}"] = float(m.w_target.detach().norm())
        out[f"health/conv_L{i}"] = float(m.target_conv.weight.detach().norm())
        # Sigmoid-gate diagnostic (only present when cfg.output_gate=True).
        # See gate_stats() docstring for interpretation.
        if getattr(m, "_gate_mean", None) is not None:
            out[f"health/gate_mean_L{i}"] = m._gate_mean
            out[f"health/gate_std_L{i}"] = m._gate_std
    out["health/lora_norm"] = float(torch.sqrt(sum(
        p.detach().float().pow(2).sum() for p in named_groups["lora"]
    )))
    return out
