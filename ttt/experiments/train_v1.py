"""Continual pretraining with session-persistent TTT. The training entrypoint.

    modal run --detach ttt/experiments/train_v1.py::train
"""

from __future__ import annotations

from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, eval_loop, train_loop
from ttt.core import report
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-train-v1")
image = modal_runtime.build_image()


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    announce: Callable[[str], None] = print,
) -> train_loop.TrainResult:
    """The composition. Every costly thing arrives through `engine`."""
    docs = data_pipeline.documents(
        data_pipeline.load(
            source=source,
            spec=resolved.spec,
            cfg=resolved.train,
            rng=engine.rng(),
            tokenizer=engine.tokenizer,
            limit_docs=resolved.limit_docs,
            weights=resolved.weights,
        )
    )
    _runtime.announce_boot(engine, docs, announce)

    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    for shortfall in holdout.shortfalls:
        announce(
            f"  holdout: {shortfall.source} has {shortfall.got} of "
            f"{shortfall.wanted} requested documents"
        )

    carries, _ = engine.carries()
    return train_loop.train(
        docs=docs,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        tracker=engine.tracker,
        strategy=resolved.strategy,
        cfg=resolved.train,
        rng=engine.rng(),
        carries=carries,
        checkpoint=engine.save,
        evaluate=_evaluator(resolved, engine, holdout.docs, announce),
        announce=announce,
    )


def _evaluator(resolved, engine, holdout_docs, announce):
    """None when there is nothing held out — an eval on an empty pool would
    report nan and look like a measurement."""
    if not holdout_docs:
        return None

    def evaluate(*, step: int, carries):
        measured = eval_loop.evaluate(
            docs=holdout_docs,
            compute=engine.compute,
            fast_weights=engine.fast_weights,
            n_slices=resolved.train.eval_n_slices,
            carries=carries,
            session_training=resolved.train.session_training,
        )
        announce(report.render(report.per_source_table(measured.summaries)))
        return measured.metrics

    return evaluate


@app.function(
    image=image,
    gpu=modal_runtime.GPU,
    volumes={
        modal_runtime.CKPT_MOUNT: modal_runtime.checkpoint_volume(),
        modal_runtime.HF_CACHE_MOUNT: modal_runtime.cache_volume(),
    },
    secrets=modal_runtime.secrets(),
    timeout=24 * 60 * 60,
)
@modal_runtime.caching
def train(**flags):
    from ttt.adapters.hf_data_source import HfDataSource

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=True,
    )
    engine.tracker = _runtime.tracker(resolved, job_type="train")
    result = run(resolved, engine=engine, source=HfDataSource())
    print(f"done: {result.steps} steps, {result.total_tokens:,d} tokens")
    return result.steps


@app.local_entrypoint()
def main(flags: str = ""):
    train.remote(**cli.parse_flags(flags))
