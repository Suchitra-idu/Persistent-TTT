"""Every figure in the reports, in one print style, from the result bundles.

    python -m ttt.experiments.report_figures_v1 [--out results/paper/report_figures]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import evidence, paper_analysis
from ttt.core.metrics import ALL_SOURCES
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR
from ttt.experiments.thesis_figures_v1 import _rows

OUT = RESULTS_DIR / "paper" / "report_figures"
N_BOOT = 1000
CONTROL = "lora-only-matched"
GATED_200 = "hebbian-carry-b"
GATED = "hebbian-carry-600"
GATED_43 = "hebbian-carry-600-s43"
GATED_FULL = "gated-carry-1050"
FULL_STEP = 1049
ACROSS = "mixed-carry-600"
ACROSS_43 = "mixed-carry-600-s43"
TIER1_LOGS = Path(__file__).resolve().parents[2] / "plan" / "tier1" / "logs"

WIDTH = 6.3
INK = "#1a1a1a"
INK_2 = "#4d4c48"
GRID = "#e6e5df"
ZERO = "#8a8984"
BLUE = "#2a78d6"
NAVY = "#123f78"
ORANGE = "#eb6834"
GRAY = "#8f8d87"
BAND = 0.16


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })
    return plt


def _axes(ax, *, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _zero(ax, *, vertical: bool = False) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=ZERO, linewidth=0.9, zorder=1)


def _save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "png" / f"{name}.png", dpi=110, bbox_inches="tight", pad_inches=0.03)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _pooled(run: str, step: int, *, matched: bool = True) -> dict:
    base_step = step if step in (200, 400, 600) else 600
    base = _rows(CONTROL, base_step, "regimes")
    rows = paper_analysis.rebased(_rows(run, step, "regimes"), base)
    estimates = list(evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT))
    try:
        estimates += evidence.boundary_estimates(_rows(run, step, "boundary"), rng=NumpyRng(0), n_boot=N_BOOT)
    except (FileNotFoundError, LookupError):
        pass
    if matched:
        estimates += paper_analysis.matched_estimates(base, rows, rng=NumpyRng(0), n_boot=N_BOOT)
    return {e.measure: e for e in estimates if e.source == ALL_SOURCES}


GATED_RUNS = (("gated carry, seed 42", BLUE, "o", "-", {200: GATED_200, 400: GATED, 600: GATED, FULL_STEP: GATED_FULL}),
              ("gated carry, seed 43", BLUE, "D", "--", {200: GATED_43, 400: GATED_43, 600: GATED_43}))


def _training_curve() -> list[tuple[int, float]]:
    step, points = 0, []
    for line in (TIER1_LOGS / "gated-carry-1050.train.log").read_text(errors="ignore").splitlines():
        if m := re.search(r"step (\d+)/\d+ loss", line):
            step = int(m.group(1))
        elif "[eval]" in line and (m := re.search(r"gap_between ([-\d.e+]+)", line)):
            points.append((step, float(m.group(1))))
    return points


def gated_curve(out: Path) -> Path:
    plt = _plt()
    fig, (within_ax, across_ax) = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    curve = _training_curve()
    within_ax.plot([s for s, _ in curve], [g for _, g in curve], linestyle="none", marker="o", markersize=3.5,
                   markerfacecolor="white", markeredgecolor=GRAY, markeredgewidth=0.9,
                   label="in-training evaluation, 30 documents", zorder=2)
    for label, colour, marker, style, runs in GATED_RUNS:
        steps = sorted(runs)
        pooled = {s: _pooled(runs[s], s, matched=False) for s in steps}
        for ax, measure in ((within_ax, "between_slices"), (across_ax, "cross_doc")):
            have = [s for s in steps if measure in pooled[s]]
            intervals = [pooled[s][measure].interval for s in have]
            ax.errorbar(have, [i.point for i in intervals],
                        yerr=[[i.point - i.low for i in intervals], [i.high - i.point for i in intervals]],
                        color=colour, marker=marker, linestyle=style, capsize=2.5, linewidth=1.3,
                        markerfacecolor=colour if style == "-" else "white", label=label, zorder=3)
    within_ax.set_title("(a) Between slices of the same document")
    across_ax.set_title("(b) Into another document")
    for ax in (within_ax, across_ax):
        _zero(ax)
        ax.set_xticks((200, 400, 600, 800, 1049))
        ax.set_xlabel("Training step")
        _axes(ax)
    within_ax.set_ylabel("Gain in perplexity")
    handles, labels = within_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=2)
    return _save(fig, out, "F15_carry_design_curve")


def _dots(ax, levels, cases, values) -> None:
    width = 0.72 / len(cases)
    for k, (label, colour, marker, filled) in enumerate(cases):
        offset = (k - (len(cases) - 1) / 2) * width
        points = [values[label][m] for m, _ in levels]
        ax.errorbar([i + offset for i in range(len(levels))], [p for p, _, _ in points],
                    yerr=[[p - lo for p, lo, _ in points], [hi - p for p, _, hi in points]],
                    fmt=marker, color=colour, capsize=2, linewidth=1.1, markersize=5,
                    markerfacecolor=colour if filled else "white", markeredgewidth=1.1, label=label, zorder=3)
    for i in range(1, len(levels)):
        ax.axvline(i - 0.5, color=GRID, linewidth=0.6, zorder=0)
    _zero(ax)
    ax.set_xticks(range(len(levels)), [name for _, name in levels])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.tick_params(axis="x", length=0)
    _axes(ax)


CONF_LEVELS = (("gain_self", "same\ndocument"), ("gain_related", "related\narticle"), ("gain_generic", "generic\ncarry"),
               ("gain_random", "random\narticle"), ("gain_chain_10", "chain of\n10 articles"),
               ("gain_other_source", "other\nsource"))


FRESH_BUNDLES = {
    (GATED, 600): "20261102-112255-fresh",
    (GATED_43, 600): "20261102-114532-fresh",
    (GATED_FULL, FULL_STEP): "20261103-001617-fresh",
    (ACROSS, 600): "20261102-115716-fresh",
    (ACROSS_43, 600): "20261102-225743-fresh",
}


def _fresh(run: str, step: int) -> dict:
    """Pinned: a step also holds token-gain and short-text bundles of the same cases."""
    bundle = RESULTS_DIR / run / f"step_{step}" / FRESH_BUNDLES[(run, step)]
    out = {}
    for line in (bundle / "tables" / "relatedness.csv").read_text().splitlines()[1:]:
        measure, source, _, point, low, high, *_ = line.split(",")
        if source == ALL_SOURCES:
            out[measure] = (float(point), float(low), float(high))
    return out


def conf_relatedness(out: Path) -> Path:
    plt = _plt()
    within = (("seed 42", GATED, 600, BLUE, "o", True), ("seed 43", GATED_43, 600, BLUE, "D", False),
              ("seed 42, step 1049", GATED_FULL, FULL_STEP, NAVY, "^", True))
    across = (("seed 42", ACROSS, 600, ORANGE, "o", True), ("seed 43", ACROSS_43, 600, ORANGE, "D", False))
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
    for ax, cases, title in ((axes[0], within, "(a) Gated carry trained within documents"),
                             (axes[1], across, "(b) Gated carry trained across documents")):
        values = {label: _fresh(run, step) for label, run, step, *_ in cases}
        _dots(ax, CONF_LEVELS, [(l, c, m, f) for l, _, _, c, m, f in cases], values)
        ax.set_title(title)
        ax.set_ylabel("Gain in perplexity")
        ax.legend(loc="lower left", ncol=3)
    axes[1].set_xlabel("Text carried into the target")
    fig.tight_layout(h_pad=1.2)
    return _save(fig, out, "F18_fresh_relatedness")


TOKEN_BUNDLE = "20261103-164544-tokengain-fresh"
TOKEN_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
TOKEN_ARMS = (("self", "same document", NAVY, "o"), ("related", "related article", BLUE, "s"),
              ("random", "random article", GRAY, "D"), ("other_source", "other source", ORANGE, "v"))


def _token_gains(run: str):
    from ttt.app import token_gain_analysis

    rows = json.loads((RESULTS_DIR / run / "step_600" / TOKEN_BUNDLE / "raw" / "token_gain.json").read_text())
    return token_gain_analysis.gains(rows)


def _token_bands(g, cluster, n_boot: int = N_BOOT):
    import numpy as np

    from ttt.app.token_gain_analysis import profile

    edges = TOKEN_EDGES
    k = int(cluster.max()) + 1
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([np.flatnonzero(cluster == c) for c in rng.integers(0, k, k)])
        draws.append(profile(g[pick], edges))
    lo, hi = np.percentile(np.stack(draws), [2.5, 97.5], axis=0)
    return profile(g, edges), lo, hi


def token_position(out: Path) -> Path:
    import numpy as np

    from ttt.app.token_gain_analysis import clusters

    plt = _plt()
    titles = json.loads((RESULTS_DIR / "paper" / "inputs" / "fresh_case_titles.json").read_text())
    fig, a = plt.subplots(figsize=(WIDTH * 0.62, 2.7))
    xs = np.sqrt(np.asarray(TOKEN_EDGES[:-1]) * np.asarray(TOKEN_EDGES[1:]))
    order, g = _token_gains(GATED)
    cluster = np.asarray(clusters([{"prime_titles": titles[d]} for d in order]))
    for arm, label, colour, marker in TOKEN_ARMS:
        point, lo, hi = _token_bands(g[arm], cluster)
        a.fill_between(xs, lo, hi, color=colour, alpha=BAND, linewidth=0, zorder=2)
        a.plot(xs, point, color=colour, marker=marker, markersize=4, label=f"carry: {label}", zorder=3)
    a.set_xscale("log")
    _axes(a)
    _zero(a)
    a.set_xlabel("Position in the target (tokens)")
    a.set_ylabel("Gain per token over reset (nats)")
    a.legend(loc="upper right", fontsize=7.5, handlelength=1.6)
    return _save(fig, out, "F19_token_position")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    (args.out / "png").mkdir(parents=True, exist_ok=True)
    for make in (gated_curve, conf_relatedness, token_position):
        print(make(args.out))


if __name__ == "__main__":
    main()
