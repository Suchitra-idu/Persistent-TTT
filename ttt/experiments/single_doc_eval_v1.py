"""One held-out document, sliced into n parts, run as a single session.

Was `single_paper_eval` — SlimPajama has documents, not papers (D10).

    modal run ttt/experiments/single_doc_eval_v1.py --n-docs 5 \\
        --flags "resume_from=step_600"

`--n-docs` > 1 also writes a PNG of the chained run to `graphs/` (needs the
`plot` extra: `pip install -e ".[plot]"`).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, session_eval
from ttt.core import report
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

BySource = tuple[
    tuple[str, tuple[session_eval.ItemRow, ...], tuple[session_eval.ItemRow, ...]], ...
]

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
) -> BySource:
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
    resolved = _ensure_docs_available(resolved, n_docs)
    holdout = data_pipeline.holdout(
        source=source,
        spec=resolved.spec,
        cfg=resolved.train,
        rng=engine.rng(resolved.train.eval_holdout_seed),
        tokenizer=engine.tokenizer,
    )
    for shortfall in holdout.shortfalls:
        announce(
            f"[{shortfall.source}] only {shortfall.got} of {shortfall.wanted} "
            "documents held out — chaining fewer than asked for"
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
        results.append((src, carry, carry_off))
    return tuple(results)


def _ensure_docs_available(resolved: cli.Resolved, n_docs: int) -> cli.Resolved:
    """`data_pipeline.holdout` caps documents per source at
    train.eval_n_docs_per_source before run_chained ever sees the pool —
    asking to chain more than that silently truncates instead of erroring,
    so raise the cap to match what was asked for."""
    if resolved.train.eval_n_docs_per_source >= n_docs:
        return resolved
    return dataclasses.replace(
        resolved,
        train=dataclasses.replace(resolved.train, eval_n_docs_per_source=n_docs),
    )


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
            f"{on.gate_mean:.3f}",
        )
        for on, off in zip(carry, carry_off, strict=True)
    )
    return report.Table(
        headers=("slice", "n_tok", "carry", "carry-off", "Δbetween", "state/W0", "gate"),
        rows=rows,
        aligns=(report.LEFT,) + (report.RIGHT,) * 6,
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
            f"{on.gate_mean:.3f}",
        )
        for on, off in zip(carry, carry_off, strict=True)
    )
    return report.Table(
        headers=(
            "pos", "doc", "source", "n_tok", "carry", "carry-off",
            "Δbetween", "state/W0", "gate",
        ),
        rows=rows,
        aligns=(report.LEFT, report.LEFT, report.LEFT) + (report.RIGHT,) * 6,
    )


_POSITIVE = "#3B82A0"   # carry helps
_NEGATIVE = "#D9695F"   # carry hurts
_LINE = "#4A5A68"        # state/W0
_GATE = "#8A6BAF"        # gate_mean
_BOUNDARY = "#B8B8B8"    # document-boundary guide
_GRID = "#E4E4E4"


def plot_chained(by_source: BySource, *, resume_from: str, out_dir: str = "graphs") -> str:
    """A small-multiples PNG of run_chained's output: Δbetween, state/W0 and
    the output gate against position, one row per source, with document
    boundaries marked. gate_mean shares this plot rather than state/W0's
    axis (D: never dual-axis) so a damped-but-informative carry is visible
    as two separate, comparable curves. Filename carries the resumed step
    and a timestamp, so successive runs are never overwritten. Needs the
    `plot` extra."""
    import os
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    label = resume_from.replace("/", "_") if resume_from else "no-resume"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"single_doc_eval_{label}_{stamp}.png")

    fig, axes = plt.subplots(
        len(by_source), 3, figsize=(19, 2.1 * len(by_source)), squeeze=False, sharex="col"
    )
    fig.suptitle(
        "Carry effect across a per-source chained session",
        fontsize=13, fontweight="bold", x=0.02, ha="left",
    )

    deltas_by_source = {
        src: [off.ppl - on.ppl for on, off in zip(carry, carry_off, strict=True)]
        for src, carry, carry_off in by_source
    }
    delta_max = max(abs(d) for deltas in deltas_by_source.values() for d in deltas) or 1.0
    ratio_max = max(on.state_ratio for _, carry, _ in by_source for on in carry) or 1.0

    for row_idx, (src, carry, carry_off) in enumerate(by_source):
        pos = [on.position for on in carry]
        deltas = deltas_by_source[src]
        ratios = [on.state_ratio for on in carry]
        gates = [on.gate_mean for on in carry]
        boundaries = [
            i for i in range(1, len(carry)) if carry[i].doc_idx != carry[i - 1].doc_idx
        ]

        ax_delta, ax_ratio, ax_gate = axes[row_idx]
        ax_delta.bar(
            pos, deltas, width=0.75, zorder=3,
            color=[_POSITIVE if d >= 0 else _NEGATIVE for d in deltas],
        )
        ax_delta.axhline(0, color="#9A9A9A", linewidth=0.8, zorder=2)
        ax_delta.set_ylim(-delta_max * 1.1, delta_max * 1.1)

        ax_ratio.plot(pos, ratios, color=_LINE, linewidth=1.8, zorder=3)
        ax_ratio.fill_between(pos, ratios, color=_LINE, alpha=0.08, zorder=1)
        ax_ratio.set_ylim(0, ratio_max * 1.1)

        ax_gate.plot(pos, gates, color=_GATE, linewidth=1.8, zorder=3)
        ax_gate.fill_between(pos, gates, color=_GATE, alpha=0.08, zorder=1)
        ax_gate.set_ylim(0, 1.05)

        for ax in (ax_delta, ax_ratio, ax_gate):
            for b in boundaries:
                ax.axvline(b - 0.5, color=_BOUNDARY, linewidth=1, linestyle="--", zorder=2)
            ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#B0B0B0")
            ax.tick_params(colors="#6B6B6B", labelsize=8)

        ax_delta.set_ylabel(
            src, fontsize=9, fontweight="bold", color="#333333",
            rotation=0, ha="right", va="center", labelpad=10,
        )
        if row_idx == 0:
            ax_delta.set_title("Δbetween  (carry-off ppl − carry ppl; + = carry helps)", fontsize=9.5)
            ax_ratio.set_title("state/W0", fontsize=9.5)
            ax_gate.set_title("gate  (fraction of TTT term reaching the output)", fontsize=9.5)
        if row_idx == len(by_source) - 1:
            ax_delta.set_xlabel("position", fontsize=9)
            ax_ratio.set_xlabel("position", fontsize=9)
            ax_gate.set_xlabel("position", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


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
        return run_chained(
            resolved, engine=engine, source=source, n_slices=n_slices, n_docs=n_docs
        )
    run(resolved, engine=engine, source=source, n_slices=n_slices)
    return ()


@app.local_entrypoint()
def main(n_slices: int = 8, n_docs: int = 1, flags: str = ""):
    """n_docs=1 (default): one document per source, measured independently.
    n_docs>1: that many documents *of each source*, chained into one
    per-source session instead — carry kept across every document boundary
    within a source, never across sources — see `run_chained`. Also writes
    a PNG of the chained result to `graphs/`."""
    parsed = cli.parse_flags(flags)
    by_source = single_doc_eval.remote(n_slices=n_slices, n_docs=n_docs, **parsed)
    if not by_source:
        return
    resolved = cli.from_flags(**{**cli.env_defaults(), **parsed})
    try:
        path = plot_chained(by_source, resume_from=resolved.resume_from)
    except ImportError:
        print("skipped the graph: pip install -e '.[plot]' for matplotlib")
        return
    print(f"wrote {path}")
