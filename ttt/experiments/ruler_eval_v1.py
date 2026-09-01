"""RULER's GPU half: score example sets `ruler_prepare_v1` already built.
Logs the aggregate and per-task/length/regime scores to wandb.

`resume_from` travels inside `--flags`, not as its own CLI option — Modal
builds the CLI from `main`'s literal parameters (`cli.py`'s note), and this
entrypoint only declares `ruler_flags` and `flags`.

    modal run ttt/experiments/ruler_eval_v1.py \
        --flags "resume_from=step_600" \
        --ruler-flags "tasks=niah_single,context_lengths=4096"
"""

from __future__ import annotations

from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.core import report
from ttt.core.config.ruler import RulerConfig, parse_ruler_flags
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine
from ttt.ports.storage import Storage

app = modal.App("ttt-ruler-eval-v1")
image = modal_runtime.build_image()


def run(
    cfg: RulerConfig,
    *,
    engine: Engine,
    storage: Storage,
    announce: Callable[[str], None] = print,
):
    from ttt.app import ruler_eval
    from ttt.app import ruler_examples_io as io
    from ttt.extensions.ruler_tasks import RULER_TASKS

    examples = {
        io.bucket_key(task, length): io.read(
            storage, cfg, task, RULER_TASKS[task].version, length
        )
        for task in cfg.tasks
        for length in cfg.context_lengths
    }
    result = ruler_eval.evaluate(
        examples=examples,
        tasks=RULER_TASKS,
        cfg=cfg,
        generation=engine.generation,
        fast_weights=engine.fast_weights,
        tokenizer=engine.tokenizer,
        announce=announce,
    )
    announce(report.render(report.ruler_table(result.results)))
    engine.tracker.log(dict(result.metrics))
    for r in result.results:
        engine.tracker.log(
            {
                "context_length": float(r.length_bucket),
                f"ruler/{r.task}/{r.regime}": r.score,
            }
        )
    return result


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=60 * 60,
)
@modal_runtime.caching
def ruler_eval(ruler_flags: str = "", invocation: str = "", **flags):
    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    storage = modal_runtime.checkpoint_storage()
    engine = _runtime.build(
        resolved, storage=storage, root=modal_runtime.CKPT_MOUNT, trainable=False
    )
    engine.tracker = _runtime.tracker(resolved, job_type="eval", invocation=invocation)
    try:
        return run(parse_ruler_flags(ruler_flags), engine=engine, storage=storage).metrics
    finally:
        engine.tracker.finish()


@app.local_entrypoint()
def main(ruler_flags: str = "", flags: str = ""):
    ruler_eval.remote(
        ruler_flags=ruler_flags, invocation=cli.invocation(), **cli.parse_flags(flags)
    )
