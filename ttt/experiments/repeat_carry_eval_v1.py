"""X held-out documents per source, each replayed for `n_repeats` sessions
with a fresh carry that is never reset between replays — does perplexity on
a document already seen improve the more times its own carry has seen it,
against the same fresh/lora/cold-carry-off reference `eval_loop` uses
elsewhere.

X is `--flags eval_n_docs_per_source=N`, an existing knob; SlimPajama is the
default dataset already.

    modal run ttt/experiments/repeat_carry_eval_v1.py --flags "resume_from=step_600"

Also writes a PNG of Δbetween/state/gate against repeat count to `graphs/`
(needs the `plot` extra: `pip install -e ".[plot]"`).
"""

from __future__ import annotations

from typing import Callable, Sequence

import modal

from ttt import cli
from ttt.adapters import modal_runtime
from ttt.app import data_pipeline, eval_loop, session_eval
from ttt.core import metrics, report
from ttt.core.metrics import ALL_SOURCES, RepeatSummary
from ttt.core.types import COLD_CARRY, EvalRow, RepeatRow
from ttt.experiments import _runtime
from ttt.experiments._runtime import Engine

app = modal.App("ttt-repeat-carry-eval-v1")
image = modal_runtime.build_image()


def run(
    resolved: cli.Resolved,
    *,
    engine: Engine,
    source,
    n_repeats: int = 20,
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

    reference = eval_loop.evaluate(
        docs=holdout.docs,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=1,
        session_training=resolved.train.session_training,
    )
    rows = _reference_rows(reference.rows, n_repeats) + tuple(
        RepeatRow(
            doc_idx=item.doc_idx,
            source=item.source,
            regime=COLD_CARRY,
            repeat=item.position,
            n_tokens=item.n_tokens,
            ppl=item.ppl,
            state_ratio=item.state_ratio,
            gate_mean=item.gate_mean,
        )
        for doc in holdout.docs
        for item in _replay(engine, doc, n_repeats)
    )
    summaries = metrics.summarise_by_repeat(rows)
    announce(f"{len(holdout.docs)} documents x {n_repeats} repeats")
    announce(report.render(report.repeat_table(summaries)))
    return summaries


def _reference_rows(
    eval_rows: Sequence[EvalRow], n_repeats: int
) -> tuple[RepeatRow, ...]:
    """fresh / lora_only / cold_carry_off never evolve across a replay of the
    same document — measured once each (`eval_loop`, whole doc, one slice)
    and repeated so every index has a full row to diff against. `cold_carry`
    is dropped here: its real, evolving trend comes from `_replay` below."""
    return tuple(
        RepeatRow(
            doc_idx=row.doc_idx,
            source=row.source,
            regime=row.regime,
            repeat=repeat,
            n_tokens=row.n_tokens,
            ppl=row.ppl,
            state_ratio=row.state_ratio_final,
        )
        for row in eval_rows
        if row.regime != COLD_CARRY
        for repeat in range(n_repeats)
    )


def _replay(engine: Engine, doc, n_repeats: int):
    return session_eval.session_perplexity(
        docs=(doc,) * n_repeats,
        compute=engine.compute,
        fast_weights=engine.fast_weights,
        n_slices=1,
        reset_between_items=False,
        reset_between_docs=False,
    )


_POSITIVE = "#3B82A0"   # carry helps
_NEGATIVE = "#D9695F"   # carry hurts
_LINE = "#4A5A68"        # state/W0
_GATE = "#8A6BAF"        # gate_mean
_GRID = "#E4E4E4"


def plot_repeats(
    summaries: Sequence[RepeatSummary], *, resume_from: str, out_dir: str = "graphs"
) -> str:
    """A small-multiples PNG of `run`'s output: Δbetween, state/W0 and the
    output gate against repeat count, one row per source plus an
    ALL_SOURCES rollup — same three-column layout as
    single_doc_eval_v1.plot_chained, no document boundaries since every
    repeat is the same document(s) again. Filename carries the resumed step
    and a timestamp. Needs the `plot` extra."""
    import os
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    label = resume_from.replace("/", "_") if resume_from else "no-resume"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"repeat_carry_eval_{label}_{stamp}.png")

    by_source: dict[str, list[RepeatSummary]] = {}
    for s in summaries:
        by_source.setdefault(s.source, []).append(s)
    sources = sorted(by_source, key=lambda src: (src == ALL_SOURCES, src))
    for src in sources:
        by_source[src].sort(key=lambda s: s.repeat)

    fig, axes = plt.subplots(
        len(sources), 3, figsize=(19, 2.1 * len(sources)), squeeze=False, sharex="col"
    )
    fig.suptitle(
        "Carry effect across repeated replays of the same document(s)",
        fontsize=13, fontweight="bold", x=0.02, ha="left",
    )

    deltas_by_source = {src: [s.gaps.between for s in rows] for src, rows in by_source.items()}
    delta_max = max(abs(d) for deltas in deltas_by_source.values() for d in deltas) or 1.0
    ratio_max = max(s.state_ratio for rows in by_source.values() for s in rows) or 1.0

    for row_idx, src in enumerate(sources):
        rows = by_source[src]
        repeats = [s.repeat for s in rows]
        deltas = deltas_by_source[src]
        ratios = [s.state_ratio for s in rows]
        gates = [s.gate_mean for s in rows]

        ax_delta, ax_ratio, ax_gate = axes[row_idx]
        ax_delta.bar(
            repeats, deltas, width=0.75, zorder=3,
            color=[_POSITIVE if d >= 0 else _NEGATIVE for d in deltas],
        )
        ax_delta.axhline(0, color="#9A9A9A", linewidth=0.8, zorder=2)
        ax_delta.set_ylim(-delta_max * 1.1, delta_max * 1.1)

        ax_ratio.plot(repeats, ratios, color=_LINE, linewidth=1.8, zorder=3)
        ax_ratio.fill_between(repeats, ratios, color=_LINE, alpha=0.08, zorder=1)
        ax_ratio.set_ylim(0, ratio_max * 1.1)

        ax_gate.plot(repeats, gates, color=_GATE, linewidth=1.8, zorder=3)
        ax_gate.fill_between(repeats, gates, color=_GATE, alpha=0.08, zorder=1)
        ax_gate.set_ylim(0, 1.05)

        for ax in (ax_delta, ax_ratio, ax_gate):
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
        if row_idx == len(sources) - 1:
            ax_delta.set_xlabel("repeat", fontsize=9)
            ax_ratio.set_xlabel("repeat", fontsize=9)
            ax_gate.set_xlabel("repeat", fontsize=9)

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
def repeat_carry_eval(n_repeats: int = 20, **flags):
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
    return run(resolved, engine=engine, source=HfDataSource(), n_repeats=n_repeats)


@app.local_entrypoint()
def main(n_repeats: int = 20, flags: str = ""):
    parsed = cli.parse_flags(flags)
    summaries = repeat_carry_eval.remote(n_repeats=n_repeats, **parsed)
    if not summaries:
        return
    resolved = cli.from_flags(**{**cli.env_defaults(), **parsed})
    try:
        path = plot_repeats(summaries, resume_from=resolved.resume_from)
    except ImportError:
        print("skipped the graph: pip install -e '.[plot]' for matplotlib")
        return
    print(f"wrote {path}")
