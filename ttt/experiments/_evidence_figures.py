"""The paper's figures, from `ttt.app.evidence` outputs. PNG, light surface.

Colours are the validated reference palette (dataviz skill). Every figure has
a table beside it in the bundle, which is the relief for the low-contrast
slots.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from ttt.app.evidence import Correlate, RepeatPoint
from ttt.core.metrics import ALL_SOURCES
from ttt.core.stats import Estimate

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")
HELPS = "#2a78d6"
HURTS = "#e34948"
MIDPOINT = "#f0efec"
DPI = 200


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 9,
        }
    )
    return plt


def _chrome(ax, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def _ordered_sources(sources) -> list[str]:
    return sorted(set(sources), key=lambda s: (s == ALL_SOURCES, s))


def _save(fig, path: str) -> str:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def forest(estimates: Sequence[Estimate], path: str, *, title: str) -> str:
    """One panel per measure, one row per source; a filled dot means the 95%
    interval excludes zero. Panels keep their own x-scale."""
    from matplotlib.ticker import MaxNLocator

    plt = _pyplot()
    by_measure: dict[str, list[Estimate]] = defaultdict(list)
    for e in estimates:
        by_measure[e.measure].append(e)
    measures = list(by_measure)
    sources = _ordered_sources(e.source for e in estimates)
    fig, axes = plt.subplots(
        1, len(measures), figsize=(3.2 * len(measures) + 1.2, 0.42 * len(sources) + 1.4),
        squeeze=False, sharey=True,
    )
    for ax, measure in zip(axes[0], measures):
        rows = {e.source: e for e in by_measure[measure]}
        for y, source in enumerate(reversed(sources)):
            e = rows.get(source)
            if e is None or not math.isfinite(e.interval.point):
                continue
            filled = e.interval.excludes_zero
            ax.hlines(y, e.interval.low, e.interval.high, color=SERIES[0], linewidth=2, zorder=3)
            ax.plot(
                e.interval.point, y, "o", markersize=7, zorder=4,
                markerfacecolor=SERIES[0] if filled else SURFACE,
                markeredgecolor=SERIES[0], markeredgewidth=2,
            )
        ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(measure, loc="left")
        ax.set_xlabel("Δppl  (+ = treated arm lower)")
        _chrome(ax, grid_axis="x")
    axes[0][0].set_yticks(range(len(sources)), list(reversed(sources)))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def repeat_curves(points: Sequence[RepeatPoint], path: str) -> str:
    """Δppl at repeat r against the cold pass, 95% band, one panel per source."""
    plt = _pyplot()
    by_source = _by_source(points)
    sources = _ordered_sources(by_source)
    fig, axes = _grid(plt, len(sources))
    for ax, source in zip(axes, sources):
        series = by_source[source]
        xs = [p.repeat for p in series]
        ax.fill_between(
            xs, [p.gap.low for p in series], [p.gap.high for p in series],
            color=SERIES[0], alpha=0.18, linewidth=0, zorder=2,
        )
        ax.plot(xs, [p.gap.point for p in series], color=SERIES[0], linewidth=2, marker="o", markersize=4, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
        ax.set_title(f"{source}  (n={series[0].n_docs})", loc="left")
        ax.set_xlabel("repeat")
        ax.set_ylabel("Δppl vs repeat 0")
        _chrome(ax)
    _hide_unused(axes, len(sources))
    fig.suptitle(
        "Gain from re-reading the same document with its own carry",
        x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def stability(points: Sequence[RepeatPoint], path: str) -> str:
    """state/W0 and gate, median with IQR across documents. Separate columns,
    never a shared axis: the two are on unrelated scales."""
    plt = _pyplot()
    by_source = _by_source(points)
    sources = _ordered_sources(by_source)
    fig, axes = plt.subplots(
        len(sources), 2, figsize=(10, 1.9 * len(sources) + 0.8), squeeze=False, sharex=True
    )
    for row, source in enumerate(sources):
        series = by_source[source]
        xs = [p.repeat for p in series]
        for col, (attr, label, colour) in enumerate(
            (("state_ratio", "state/W0", SERIES[0]), ("gate_mean", "gate", SERIES[1]))
        ):
            ax = axes[row][col]
            bands = [getattr(p, attr) for p in series]
            ax.fill_between(
                xs, [b.low for b in bands], [b.high for b in bands],
                color=colour, alpha=0.18, linewidth=0, zorder=2,
            )
            ax.plot(xs, [b.median for b in bands], color=colour, linewidth=2, zorder=3)
            if attr == "state_ratio":
                ax.set_ylim(bottom=0)
            ax.set_title(f"{source} — {label}", loc="left")
            _chrome(ax)
        axes[row][0].set_ylabel("median, IQR")
    for ax in axes[-1]:
        ax.set_xlabel("repeat")
    fig.suptitle(
        "Carry magnitude and output gate under repeated exposure",
        x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, path)


def swap_heatmap(estimates: Sequence[Estimate], path: str) -> str:
    """Rows prime source, columns target source; diverging around zero, `*`
    where the interval excludes zero."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt = _pyplot()
    cells = {
        (e.measure.removeprefix("prime:"), e.source): e
        for e in estimates
        if e.measure.startswith("prime:")
    }
    primes = sorted({p for p, _ in cells})
    targets = sorted({t for _, t in cells})
    values = [
        [cells[(p, t)].interval.point if (p, t) in cells else math.nan for t in targets]
        for p in primes
    ]
    finite = [abs(v) for row in values for v in row if math.isfinite(v)]
    bound = max(finite) if finite and max(finite) > 0 else 1.0
    cmap = LinearSegmentedColormap.from_list("polarity", [HURTS, MIDPOINT, HELPS])
    fig, ax = plt.subplots(figsize=(1.1 * len(targets) + 2.6, 0.8 * len(primes) + 1.6))
    image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(0.0, -bound, bound), aspect="auto")
    for i, p in enumerate(primes):
        for j, t in enumerate(targets):
            e = cells.get((p, t))
            if e is None:
                continue
            mark = "*" if e.interval.excludes_zero else ""
            ax.text(j, i, f"{e.interval.point:+.2f}{mark}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(targets)), targets, rotation=30, ha="right")
    ax.set_yticks(range(len(primes)), primes)
    ax.set_xlabel("target source (document read)")
    ax.set_ylabel("prime source (carry built on)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="Δppl vs unprimed  (+ = carry helps)", shrink=0.8)
    ax.set_title("Carry built on one source, read on another", loc="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def domain_scatter(
    fresh: Mapping[str, float],
    estimates: Sequence[Estimate],
    correlates: Sequence[Correlate],
    path: str,
) -> str:
    """One panel per measure: per-source relative gap with its interval against
    that source's base-model perplexity. Sources are direct-labelled."""
    plt = _pyplot()
    rho = {c.measure: c for c in correlates}
    measures = [c.measure for c in correlates]
    fig, axes = plt.subplots(
        1, len(measures), figsize=(3.6 * len(measures) + 0.6, 3.4), squeeze=False
    )
    for ax, measure in zip(axes[0], measures):
        rows = [e for e in estimates if e.measure == measure and e.source in fresh]
        for e in rows:
            x = fresh[e.source]
            scale = 100.0 / e.reference
            ax.vlines(x, e.interval.low * scale, e.interval.high * scale, color=SERIES[0], linewidth=2, zorder=3)
            ax.plot(x, e.relative * 100.0, "o", color=SERIES[0], markersize=7, zorder=4)
            ax.annotate(
                e.source, (x, e.relative * 100.0), xytext=(5, 3),
                textcoords="offset points", fontsize=7.5, color=INK_SECONDARY,
            )
        ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
        c = rho[measure]
        ax.set_title(f"{measure}   ρ = {c.rho:+.2f}  (n={c.n_sources})", loc="left")
        ax.set_xlabel("fresh ppl (base model)")
        ax.set_ylabel("Δppl, % of reference")
        _chrome(ax)
    fig.tight_layout()
    return _save(fig, path)


def _by_source(points: Sequence[RepeatPoint]) -> dict[str, list[RepeatPoint]]:
    grouped: dict[str, list[RepeatPoint]] = defaultdict(list)
    for p in points:
        grouped[p.source].append(p)
    for series in grouped.values():
        series.sort(key=lambda p: p.repeat)
    return grouped


def _grid(plt, n: int, columns: int = 3):
    rows = max(1, math.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.8 * rows + 0.6), squeeze=False)
    return fig, [ax for row in axes for ax in row]


def _hide_unused(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)
