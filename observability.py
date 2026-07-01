"""Observability for In-Place TTT training, built on Weights & Biases.

Telemetry never crashes or stalls a run; failures degrade to console-only.
Two x-axes: optimizer steps ("train/step") for aggregates, micro steps
("micro/step") for per-paper signals.
"""

from __future__ import annotations

import os
import time


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
        """Push notification (needs Scriptable Alerts enabled in W&B settings)."""
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


def gpu_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / 2**30,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "gpu/mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
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
    out["health/lora_norm"] = float(torch.sqrt(sum(
        p.detach().float().pow(2).sum() for p in named_groups["lora"]
    )))
    return out
