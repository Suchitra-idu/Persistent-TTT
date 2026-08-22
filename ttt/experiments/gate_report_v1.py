"""Per-layer output_gate.bias/weight straight off a loaded checkpoint —
independent of any eval's own gate_mean, so a suspiciously-flat reading can
be checked against the actual trained parameter rather than trusted.

    modal run ttt/experiments/gate_report_v1.py --flags "resume_from=step_600"
"""

from __future__ import annotations

from typing import Callable

import modal
import torch

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.core import report
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine
from ttt.extensions.mechanism import iter_ttt_modules

app = modal.App("ttt-gate-report-v1")
image = modal_runtime.build_image()


def run(engine: Engine, *, announce: Callable[[str], None] = print) -> report.Table:
    rows = tuple(
        (
            str(index),
            f"{float(module.output_gate.bias):+.3f}",
            f"{float(torch.sigmoid(module.output_gate.bias)):.3f}",
            f"{float(module.output_gate.weight.norm()):.4f}",
        )
        for index, module in enumerate(iter_ttt_modules(engine.model))
    )
    table = report.Table(
        headers=("layer", "bias", "sigmoid(bias)", "|weight|"),
        rows=rows,
        aligns=(report.LEFT,) + (report.RIGHT,) * 3,
    )
    announce(report.render(table))
    return table


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=15 * 60,
)
@modal_runtime.caching
def gate_report(**flags):
    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    run(engine)


@app.local_entrypoint()
def main(flags: str = ""):
    gate_report.remote(**cli.parse_flags(flags))
