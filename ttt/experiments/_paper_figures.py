"""Cross-checkpoint figures for the paper, in `_evidence_figures`' style.

A run keeps one colour and one marker in every figure, so identity never rests
on colour alone.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ttt.app.evidence import RepeatPoint
from ttt.core.stats import Estimate
from ttt.experiments._evidence_figures import INK, MUTED, SERIES, _chrome, _pyplot, _save

MARKERS = ("o", "s")
BAND_ALPHA = 0.18


def _style(index: int) -> dict:
    return dict(color=SERIES[index], marker=MARKERS[index], markersize=6, linewidth=2)


def _title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")


def _end_label(ax, x: float, y: float, text: str, *, above: bool = False, nudge: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(6, nudge + (4 if above else 0)), textcoords="offset points",
                va="bottom" if above else "center", fontsize=8, color=INK)


def _end_labels(ax, ends: Sequence[tuple[float, float, str]]) -> None:
    """Labels at line ends, pushed apart vertically when they would collide."""
    low, high = ax.get_ylim()
    ordered = sorted(ends, key=lambda e: e[1])
    for i, (x, y, text) in enumerate(ordered):
        crowded = len(ordered) > 1 and any(
            j != i and abs(y - other) < 0.08 * (high - low) for j, (_, other, _) in enumerate(ordered)
        )
        nudge = (5 if i == len(ordered) - 1 else -5) if crowded else 0
        _end_label(ax, x, y, text, nudge=nudge)


def gains_vs_step(
    series: Mapping[str, Mapping[int, Mapping[str, Estimate]]],
    panels: Sequence[tuple[str, str]],
    path: str,
) -> str:
    """One panel per (measure, title), one line per run with its 95% band; each
    panel keeps its own y-scale."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), squeeze=False)
    for ax, (measure, title) in zip(axes[0], panels):
        ends = []
        for index, (run, by_step) in enumerate(series.items()):
            steps = sorted(s for s in by_step if measure in by_step[s])
            gaps = [by_step[s][measure].interval for s in steps]
            ax.fill_between(steps, [g.low for g in gaps], [g.high for g in gaps],
                            color=SERIES[index], alpha=BAND_ALPHA, linewidth=0, zorder=2)
            ax.plot(steps, [g.point for g in gaps], label=run, zorder=3, **_style(index))
            ends.append((steps[-1], gaps[-1].point, run))
        ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
        _end_labels(ax, ends)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel("Δppl  (+ = lower perplexity)")
        ax.margins(x=0.25)
        _chrome(ax)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    _title(fig, "Where the gain comes from, by training step (95% CI, 106 held-out documents)")
    fig.tight_layout()
    return _save(fig, path)


def by_position(
    series: Mapping[str, Sequence[tuple[int, Estimate]]],
    measures: Sequence[tuple[str, str]],
    path: str,
) -> str:
    """Pooled gap per slice index, one panel per measure."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(measures), figsize=(4.6 * len(measures), 3.4), squeeze=False)
    for ax, (measure, title) in zip(axes[0], measures):
        for index, (run, points) in enumerate(series.items()):
            cells = sorted((i, e.interval) for i, e in points if e.measure == measure)
            xs = [i for i, _ in cells]
            ax.fill_between(xs, [g.low for _, g in cells], [g.high for _, g in cells],
                            color=SERIES[index], alpha=BAND_ALPHA, linewidth=0, zorder=2)
            ax.plot(xs, [g.point for _, g in cells], label=run, zorder=3, **_style(index))
        ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("slice index within the document")
        ax.set_ylabel("Δppl  (+ = lower perplexity)")
        _chrome(ax)
    axes[0][0].legend(frameon=False, fontsize=8)
    _title(fig, "TTT gain by position in the document")
    fig.tight_layout()
    return _save(fig, path)


def repeat_mechanism(
    curves: Mapping[str, Sequence[RepeatPoint]], clip_threshold: float, path: str
) -> str:
    """Left: carried state against the clip threshold (log scale). Right: what
    re-reading the same document does to perplexity."""
    plt = _pyplot()
    fig, (state_ax, gap_ax) = plt.subplots(1, 2, figsize=(10, 3.6))
    for index, (run, points) in enumerate(curves.items()):
        points = sorted(points, key=lambda p: p.repeat)
        xs = [p.repeat for p in points]
        state_ax.fill_between(xs, [p.state_ratio.low for p in points],
                              [p.state_ratio.high for p in points],
                              color=SERIES[index], alpha=BAND_ALPHA, linewidth=0, zorder=2)
        state_ax.plot(xs, [p.state_ratio.median for p in points], label=run, zorder=3, **_style(index))
        gap_ax.fill_between(xs, [p.gap.low for p in points], [p.gap.high for p in points],
                            color=SERIES[index], alpha=BAND_ALPHA, linewidth=0, zorder=2)
        gap_ax.plot(xs, [p.gap.point for p in points], label=run, zorder=3, **_style(index))
    state_ax.set_yscale("log")
    state_ax.axhline(clip_threshold, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    _end_label(state_ax, 1, clip_threshold, f"clip binds above {clip_threshold:.2f}", above=True)
    state_ax.set_title("Carried state  ‖ηS‖ / ‖W₀‖  (median, IQR)", loc="left")
    state_ax.set_ylabel("state ratio (log)")
    gap_ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    gap_ax.set_title("Perplexity gain vs. first reading (95% CI)", loc="left")
    gap_ax.set_ylabel("Δppl  (+ = lower perplexity)")
    for ax in (state_ax, gap_ax):
        ax.set_xlabel("times the same document was re-read")
        _chrome(ax)
    gap_ax.legend(frameon=False, fontsize=8)
    _title(fig, "The carry keeps growing; its effect does not")
    fig.tight_layout()
    return _save(fig, path)
