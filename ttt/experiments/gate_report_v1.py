"""Per-layer output_gate.bias/weight straight off a loaded checkpoint —
independent of any eval's own gate_mean, so a suspiciously-flat reading can
be checked against the actual trained parameter rather than trusted. Logs
per-layer scalars and a bar chart to wandb.

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
    values = [
        (
            index,
            float(module.output_gate.bias),
            float(torch.sigmoid(module.output_gate.bias)),
            float(module.output_gate.weight.norm()),
        )
        for index, module in enumerate(iter_ttt_modules(engine.model))
    ]
    table = report.Table(
        headers=("layer", "bias", "sigmoid(bias)", "|weight|"),
        rows=tuple(
            (str(i), f"{bias:+.3f}", f"{sig:.3f}", f"{norm:.4f}")
            for i, bias, sig, norm in values
        ),
        aligns=(report.LEFT,) + (report.RIGHT,) * 3,
    )
    announce(report.render(table))
    for i, bias, sig, norm in values:
        engine.tracker.log(
            {
                f"gate/layer_{i}/bias": bias,
                f"gate/layer_{i}/sigmoid_bias": sig,
                f"gate/layer_{i}/weight_norm": norm,
            }
        )
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
def gate_report(invocation: str = "", **flags):
    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    engine.tracker = _runtime.tracker(resolved, job_type="eval", invocation=invocation)
    try:
        table = run(engine)
        _log_chart(engine.tracker, table)
    finally:
        engine.tracker.finish()


def _log_chart(tracker, table: report.Table) -> None:
    import os
    import tempfile

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = [int(row[0]) for row in table.rows]
    sigmoids = [float(row[2]) for row in table.rows]
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.5), 4))
    ax.bar(layers, sigmoids, color="#8A6BAF")
    ax.axhline(0.5, color="#9A9A9A", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TTT layer")
    ax.set_ylabel("sigmoid(bias)")
    ax.set_ylim(0, 1)
    ax.set_title("Output gate bias, per layer")
    fig.tight_layout()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gate_bias.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        tracker.log_image("gate/bias_by_layer", path)


@app.local_entrypoint()
def main(flags: str = ""):
    gate_report.remote(invocation=cli.invocation(), **cli.parse_flags(flags))
