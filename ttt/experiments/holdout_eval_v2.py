"""Six-regime perplexity, plus the same gap broken out by distance into the
document. v1's aggregate can't tell a win concentrated in the first slice
from one that holds up in the last — exactly the failure mode plain
perplexity has for long-context claims (arXiv 2410.23771).

    modal run ttt/experiments/holdout_eval_v2.py --flags "resume_from=step_600"
"""

from __future__ import annotations

import contextlib
from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, eval_loop
from ttt.core import metrics, report
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-holdout-eval-v2")
image = modal_runtime.build_image()

# A standalone run wants the tighter estimate; a --flags override still wins.
MIN_DOCS_PER_SOURCE = 50


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    seeded: bool = True,
    force_source: str = "",
    announce: Callable[[str], None] = print,
) -> eval_loop.EvalReport:
    """`force_source` installs one source's carrier for every document — the
    swap test: a source-specific benefit should degrade under a mismatch."""
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    if not holdout.docs:
        announce("no held-out documents matched the eval filters")
        return eval_loop.EvalReport(rows=(), metrics={}, summaries=())

    carries, meta = engine.carries()
    measured = eval_loop.evaluate(
        docs=holdout.docs,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=resolved.train.eval_n_slices,
        carries=_seeds(carries, holdout.docs, seeded, force_source),
    )
    _announce(measured, holdout, meta, announce)
    return measured


def _seeds(carries, docs, seeded: bool, force_source: str):
    if not seeded:
        return None
    if not force_source:
        return carries
    swapped = carries.get(force_source)
    if swapped is None:
        raise KeyError(
            f"no trained carrier for --force-source {force_source!r}; "
            f"the checkpoint has {sorted(carries)}"
        )
    return {doc.source: swapped for doc in docs}


def _announce(measured, holdout, meta, announce) -> None:
    announce(f"{len(holdout.docs)} held-out documents")
    for shortfall in holdout.shortfalls:
        announce(f"  {shortfall.source}: {shortfall.got} of {shortfall.wanted}")
    if meta.get("n_updates"):
        announce(f"  carrier updates: {dict(sorted(meta['n_updates'].items()))}")
    announce(report.render(report.per_source_table(measured.summaries)))
    announce(
        "  ".join(f"{key} {value:.4g}" for key, value in sorted(measured.metrics.items()))
    )
    if measured.slices:
        by_slice = metrics.summarise_by_slice_index(measured.slices)
        announce(report.render(report.slice_gap_table(by_slice)))


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
def holdout_eval(seeded: bool = True, force_source: str = "", **flags):
    from ttt.adapters.hf_data_source import HfDataSource

    resolved = cli.from_flags(
        **{"eval_n_docs_per_source": MIN_DOCS_PER_SOURCE, **cli.env_defaults(), **flags}
    )
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    engine.compute = _EvalCompute(engine.model)
    return run(
        resolved,
        engine=engine,
        source=HfDataSource(),
        seeded=seeded,
        force_source=force_source,
    ).metrics


class _EvalCompute:
    """`eval_loss` only: eval never steps, and building an optimizer over a
    frozen model would fail on the empty parameter groups."""

    def __init__(self, model) -> None:
        import torch

        self._torch = torch
        self.model = model

    def eval_loss(self, token_ids, *, lora: bool = True) -> float:
        ids = self._torch.tensor([list(token_ids)], device="cuda", dtype=self._torch.long)
        ctx = contextlib.nullcontext() if lora else self.model.disable_adapter()
        with self._torch.no_grad(), ctx:
            return float(self.model(input_ids=ids, labels=ids).loss)


@app.local_entrypoint()
def main(seeded: bool = True, force_source: str = "", flags: str = ""):
    holdout_eval.remote(
        seeded=seeded, force_source=force_source, **cli.parse_flags(flags)
    )
