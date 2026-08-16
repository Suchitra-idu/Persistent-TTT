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
) -> tuple[tuple[str, tuple[session_eval.ItemRow, ...]], ...]:
    """Per-slice perplexity against position, with and without the carry —
    the smallest experiment that can show adaptation inside one document. One
    document per source, not just the first document in the whole pool —
    a trend on one source is not a trend."""
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

    results = []
    for doc in _one_per_source(holdout.docs):
        carry = _measure(engine, [doc], n_slices, reset_between_items=False)
        carry_off = _measure(engine, [doc], n_slices, reset_between_items=True)
        announce(f"document {doc.index} [{doc.source}], {doc.n_tokens:,d} tokens")
        announce(report.render(slice_table(carry, carry_off)))
        results.append((doc.source, carry))
    return tuple(results)


def run_chained(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    n_slices: int = 8,
    n_docs: int = 5,
    announce: Callable[[str], None] = print,
) -> tuple[tuple[str, tuple[session_eval.ItemRow, ...]], ...]:
    """Up to `n_docs` held-out documents *per source*, chained into one
    carry-persistent session per source — not mixed across sources. In
    production carry_scope=SOURCE keeps each source's carrier separate
    (train_loop._seed); a session that jumps between sources would measure
    a carry no real run ever produces. Each doc sliced into `n_slices` and
    run back to back — doc 1's slices, then doc 2's, ... — with the carry
    kept across every slice *and* every document boundary within that
    source. `run`'s one-doc-per-source comparison is the single-document
    case; this is the same question over a longer, still single-source,
    session."""
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    by_source = _group_by_source(holdout.docs)
    if not by_source:
        announce("no held-out documents matched the eval filters")
        return ()

    results = []
    for src in sorted(by_source):
        chosen = by_source[src][:n_docs]
        carry = _measure(engine, chosen, n_slices, reset_between_items=False)
        carry_off = _measure(engine, chosen, n_slices, reset_between_items=True)
        announce(
            f"[{src}] {len(chosen)} documents chained, {n_slices} slices "
            "each, carry kept across every slice and document boundary"
        )
        announce(report.render(chain_table(carry, carry_off)))
        results.append((src, carry))
    return tuple(results)


def _group_by_source(docs: Sequence) -> dict:
    by_source: dict = {}
    for doc in docs:
        by_source.setdefault(doc.source, []).append(doc)
    return by_source


def _one_per_source(docs: Sequence) -> tuple:
    """The first held-out document for each source, in sorted source order."""
    by_source = _group_by_source(docs)
    return tuple(by_source[source][0] for source in sorted(by_source))


def _measure(engine, docs: Sequence, n_slices, *, reset_between_items):
    return session_eval.session_perplexity(
        docs=docs,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=n_slices,
        reset_between_items=reset_between_items,
        reset_between_docs=False,
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


def chain_table(
    carry: Sequence[session_eval.ItemRow], carry_off: Sequence[session_eval.ItemRow]
) -> report.Table:
    """`slice_table` plus which document each row belongs to, since a chain's
    rows span more than one."""
    rows = tuple(
        (
            str(on.position),
            str(on.doc_idx),
            on.source,
            f"{on.n_tokens:,d}",
            report.ppl(on.ppl),
            report.ppl(off.ppl),
            report.delta(off.ppl - on.ppl),
            report.ratio(on.state_ratio),
        )
        for on, off in zip(carry, carry_off, strict=True)
    )
    return report.Table(
        headers=("pos", "doc", "source", "n_tok", "carry", "carry-off", "Δbetween", "state/W0"),
        rows=rows,
        aligns=(report.LEFT, report.LEFT, report.LEFT) + (report.RIGHT,) * 5,
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
def single_doc_eval(n_slices: int = 8, n_docs: int = 1, **flags):
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
    source = HfDataSource()
    if n_docs > 1:
        run_chained(resolved, engine=engine, source=source, n_slices=n_slices, n_docs=n_docs)
    else:
        run(resolved, engine=engine, source=source, n_slices=n_slices)


@app.local_entrypoint()
def main(n_slices: int = 8, n_docs: int = 1, flags: str = ""):
    """n_docs=1 (default): one document per source, measured independently.
    n_docs>1: that many documents *of each source*, chained into one
    per-source session instead — carry kept across every document boundary
    within a source, never across sources — see `run_chained`."""
    single_doc_eval.remote(n_slices=n_slices, n_docs=n_docs, **cli.parse_flags(flags))
