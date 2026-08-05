"""One held-out document, sliced into n parts, run as a single session.

Was `single_paper_eval` — SlimPajama has documents, not papers (D10).

    modal run ttt/experiments/single_doc_eval_v1.py --resume-from step_600
"""

from __future__ import annotations

from typing import Callable, Sequence

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, session_eval
from ttt.core import report
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-single-doc-eval-v1")
image = modal_runtime.build_image()


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    n_slices: int = 8,
    announce: Callable[[str], None] = print,
) -> tuple[session_eval.ItemRow, ...]:
    """Per-slice perplexity against position, with and without the carry —
    the smallest experiment that can show adaptation inside one document."""
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

    doc = holdout.docs[0]
    carry = _measure(engine, doc, n_slices, reset_between_items=False)
    carry_off = _measure(engine, doc, n_slices, reset_between_items=True)
    announce(f"document {doc.index} [{doc.source}], {doc.n_tokens:,d} tokens")
    announce(report.render(slice_table(carry, carry_off)))
    return carry


def _measure(engine, doc, n_slices, *, reset_between_items):
    return session_eval.session_perplexity(
        docs=[doc],
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=n_slices,
        reset_between_items=reset_between_items,
    )


def slice_table(
    carry: Sequence[session_eval.ItemRow], carry_off: Sequence[session_eval.ItemRow]
) -> report.Table:
    rows = tuple(
        (
            str(on.position),
            f"{on.n_tokens:,d}",
            report.ppl(on.ppl),
            report.ppl(off.ppl),
            report.delta(off.ppl - on.ppl),
            report.ratio(on.state_ratio),
        )
        for on, off in zip(carry, carry_off, strict=True)
    )
    return report.Table(
        headers=("slice", "n_tok", "carry", "carry-off", "Δbetween", "state/W0"),
        rows=rows,
        aligns=(report.LEFT,) + (report.RIGHT,) * 5,
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
def single_doc_eval(n_slices: int = 8, **flags):
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
    run(resolved, engine=engine, source=HfDataSource(), n_slices=n_slices)


@app.local_entrypoint()
def main(n_slices: int = 8, flags: str = ""):
    single_doc_eval.remote(n_slices=n_slices, **cli.parse_flags(flags))
