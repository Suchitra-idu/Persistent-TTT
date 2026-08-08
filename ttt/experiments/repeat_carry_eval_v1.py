"""X held-out documents per source, each replayed for `n_repeats` sessions
with a fresh carry that is never reset between replays — does perplexity on
a document already seen improve the more times its own carry has seen it.

X is `--flags eval_n_docs_per_source=N`, an existing knob; SlimPajama is the
default dataset already.

    modal run ttt/experiments/repeat_carry_eval_v1.py --flags "resume_from=step_600"
"""

from __future__ import annotations

from typing import Callable

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, session_eval
from ttt.core import metrics, report
from ttt.core.types import RepeatRow
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-repeat-carry-eval-v1")
image = modal_runtime.build_image()


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    n_repeats: int = 10,
    announce: Callable[[str], None] = print,
) -> tuple[metrics.RepeatSummary, ...]:
    """Each document starts from a fresh carry, isolated from every other
    document — so a trend across repeats is that document's own, not
    something a shared per-source carrier already knew."""
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    if not holdout.docs:
        announce("no held-out documents matched the eval filters")
        return ()

    rows = tuple(
        RepeatRow(
            doc_idx=item.doc_idx,
            source=item.source,
            repeat=item.position,
            n_tokens=item.n_tokens,
            ppl=item.ppl,
        )
        for doc in holdout.docs
        for item in _replay(engine, doc, n_repeats)
    )
    summaries = metrics.summarise_by_repeat(rows)
    announce(f"{len(holdout.docs)} documents x {n_repeats} repeats")
    announce(report.render(report.repeat_table(summaries)))
    return summaries


def _replay(engine: Engine, doc, n_repeats: int):
    return session_eval.session_perplexity(
        docs=(doc,) * n_repeats,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=1,
        reset_between_items=False,
        reset_between_docs=False,
    )


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
def repeat_carry_eval(n_repeats: int = 10, **flags):
    from ttt.adapters.hf_data_source import HfDataSource
    from ttt.experiments.holdout_eval_v1 import _EvalCompute

    resolved = cli.from_flags(**{**cli.env_defaults(), **flags})
    engine = _runtime.build(
        resolved,
        storage=modal_runtime.checkpoint_storage(),
        root=modal_runtime.CKPT_MOUNT,
        trainable=False,
    )
    engine.compute = _EvalCompute(engine.model)
    run(resolved, engine=engine, source=HfDataSource(), n_repeats=n_repeats)


@app.local_entrypoint()
def main(n_repeats: int = 10, flags: str = ""):
    repeat_carry_eval.remote(n_repeats=n_repeats, **cli.parse_flags(flags))
