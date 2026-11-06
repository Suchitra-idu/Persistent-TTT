"""Every figure in the reports, in one print style, from the result bundles.

    python -m ttt.experiments.report_figures_v1 [--out results/paper/report_figures]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import evidence, paper_analysis, relatedness_eval
from ttt.core.metrics import ALL_SOURCES, geometric_mean_ppl
from ttt.core.types import COLD_CARRY, FRESH, LORA_ONLY
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR, is_development
from ttt.experiments.thesis_figures_v1 import _rows

OUT = RESULTS_DIR / "paper" / "report_figures"
N_BOOT = 1000
CONTROL = "lora-only-matched"
HEBBIAN = "ttt-v1.1"
DELTA = "ttt-v1.1-clean-deltachunk"
GATED_200 = "hebbian-carry-b"
GATED = "hebbian-carry-600"
GATED_43 = "hebbian-carry-600-s43"
GATED_FULL = "gated-carry-1050"
FULL_STEP = 1049
ACROSS = "mixed-carry-600"
ACROSS_43 = "mixed-carry-600-s43"
TIER1_LOGS = Path(__file__).resolve().parents[2] / "plan" / "tier1" / "logs"
NOCLIP_LOGS = RESULTS_DIR / "paper" / "inputs" / "noclip_training"

WIDTH = 6.3
INK = "#1a1a1a"
INK_2 = "#4d4c48"
GRID = "#e6e5df"
ZERO = "#8a8984"
BLUE = "#2a78d6"
NAVY = "#123f78"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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


def _band(ax, xs, intervals, colour, **line) -> None:
    ax.fill_between(xs, [i.low for i in intervals], [i.high for i in intervals],
                    color=colour, alpha=BAND, linewidth=0, zorder=2)
    ax.plot(xs, [i.point for i in intervals], color=colour, zorder=3, **line)


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


def ungated_gains(out: Path) -> Path:
    plt = _plt()
    panels = (("vs_matched_lora", "(a) Lead over the LoRA-only control"),
              ("within", "(b) Fast weights within a slice"),
              ("cross_doc", "(c) Carried into another document"))
    runs = (("Hebbian", HEBBIAN, BLUE, "o"), ("Delta", DELTA, ORANGE, "s"))
    steps = (200, 400, 600)
    data = {run: {s: _pooled(run, s) for s in steps} for _, run, _, _ in runs}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    for ax, (measure, title) in zip(axes, panels):
        for label, run, colour, marker in runs:
            _band(ax, steps, [data[run][s][measure].interval for s in steps], colour, marker=marker, label=label)
        _zero(ax)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(steps)
        ax.set_xlabel("Training step")
        _axes(ax)
    axes[0].set_ylabel("Gain in perplexity")
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.2)
    return _save(fig, out, "F1_gains_vs_step")


def ungated_by_slice(out: Path) -> Path:
    plt = _plt()
    runs = (("Hebbian", HEBBIAN, BLUE, "o"), ("Delta", DELTA, ORANGE, "s"))
    panels = (("within", "(a) Fast weights within a slice"), ("between_slices", "(b) State carried from earlier slices"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for label, run, colour, marker in runs:
        points = paper_analysis.position_estimates(_rows(run, 600, "regime_slices"), rng=NumpyRng(0), n_boot=N_BOOT)
        for ax, (measure, _) in zip(axes, panels):
            cells = sorted((i, e.interval) for i, e in points if e.measure == measure)
            _band(ax, [i + 1 for i, _ in cells], [g for _, g in cells], colour, marker=marker, label=label)
    for ax, (_, title) in zip(axes, panels):
        _zero(ax)
        ax.set_title(title)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Slice of the document")
        _axes(ax)
    axes[0].set_ylabel("Gain in perplexity")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=2)
    return _save(fig, out, "F2_by_position")


def repeat_mechanism(out: Path, threshold: float) -> Path:
    plt = _plt()
    runs = (("Hebbian", HEBBIAN, BLUE, "o"), ("Delta", DELTA, ORANGE, "s"))
    fig, (state_ax, gap_ax) = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for label, run, colour, marker in runs:
        rows = _rows(run, 600, "repeat")
        points = sorted((p for p in evidence.repeat_curve(rows, rng=NumpyRng(0), n_boot=N_BOOT)
                         if p.source == ALL_SOURCES), key=lambda p: p.repeat)
        xs = [p.repeat + 1 for p in points]
        state_ax.fill_between(xs, [p.state_ratio.low for p in points], [p.state_ratio.high for p in points],
                              color=colour, alpha=BAND, linewidth=0)
        state_ax.plot(xs, [p.state_ratio.median for p in points], color=colour, marker=marker, label=label)
        _band(gap_ax, xs, [p.gap for p in points], colour, marker=marker, label=label)
    state_ax.set_yscale("log")
    state_ax.axhline(threshold, color=INK, linewidth=0.9, linestyle="--")
    state_ax.annotate(f"clip threshold {threshold:.2f}", (xs[0], threshold), xytext=(2, 4),
                      textcoords="offset points", fontsize=8, color=INK)
    state_ax.set_ylabel(r"State ratio $\|\eta S\|_F / \|W_{\mathrm{down}}\|_F$")
    state_ax.set_title("(a) Size of the carried state")
    gap_ax.set_title("(b) Gain against the first reading")
    gap_ax.set_ylabel("Gain in perplexity")
    _zero(gap_ax)
    for ax in (state_ax, gap_ax):
        ax.set_xlabel("Reading of the same document")
        ax.set_xticks(xs)
        _axes(ax)
    gap_ax.set_ylim(-0.01, 0.01)
    state_ax.legend(loc="center right")
    fig.tight_layout(w_pad=2)
    return _save(fig, out, "F3_repeat_mechanism")


def absolute_ppl(out: Path) -> Path:
    plt = _plt()
    steps = (200, 400, 600)
    series = (("LoRA-only", CONTROL, LORA_ONLY, AQUA, "^"), ("Hebbian", HEBBIAN, COLD_CARRY, BLUE, "o"),
              ("Delta", DELTA, COLD_CARRY, ORANGE, "s"))
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for label, run, regime, colour, marker in series:
        values = [geometric_mean_ppl([r.ppl for r in _rows(run, s, "regimes") if r.regime == regime]) for s in steps]
        ax.plot(steps, values, color=colour, marker=marker, label=label)
        ax.annotate(f"{label} {values[-1]:.2f}", (steps[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xticks(steps)
    ax.set_xlim(170, 760)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity (geometric mean)")
    _axes(ax)
    fig.tight_layout()
    return _save(fig, out, "F5_absolute_ppl")


def chain_states(out: Path) -> Path:
    plt = _plt()
    arms = (("persist", "carried across documents", BLUE, "o"), ("doc_reset", "reset at each document", ORANGE, "s"),
            ("slice_reset", "reset at each slice", AQUA, "^"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for ax, (label, run) in zip(axes, (("(a) Hebbian", HEBBIAN), ("(b) Delta", DELTA))):
        rows = _rows(run, 600, "boundary")
        positions = sorted({r.chain_position for r in rows})
        for arm, name, colour, marker in arms:
            medians = [statistics.median(r.state_ratio for r in rows if r.arm == arm and r.chain_position == p)
                       for p in positions]
            ax.plot([p + 1 for p in positions], medians, color=colour, marker=marker, label=name)
        ax.set_yscale("log")
        ax.set_xticks([p + 1 for p in positions])
        ax.set_xlabel("Document in the chain")
        ax.set_title(label)
        _axes(ax)
    axes[0].set_ylabel("State ratio, median")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=2)
    return _save(fig, out, "F8_chain_states")


def state_distribution(out: Path, threshold: float) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for label, run, colour in (("Hebbian", HEBBIAN, BLUE), ("Delta", DELTA, ORANGE)):
        ratios = [r.state_ratio for name in ("boundary", "repeat", "swap") for r in _rows(run, 600, name)]
        ratios += [r.state_ratio_final for r in _rows(run, 600, "regimes") if r.regime == COLD_CARRY]
        values = [v for v in ratios if v > 0]
        bins = [10 ** (x / 10) for x in range(-5, 61)]
        ax.hist(values, bins=bins, color=colour, alpha=0.6, label=f"{label} ({len(values):,} measurements)")
    ax.set_xscale("log")
    ax.axvline(threshold, color=INK, linewidth=0.9, linestyle="--")
    ax.annotate(f"clip threshold {threshold:.2f}", (threshold, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK, va="top")
    ax.set_xlabel(r"State ratio $\|\eta S\|_F / \|W_{\mathrm{down}}\|_F$ (log scale)")
    ax.set_ylabel("Measurements")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8))
    _axes(ax)
    fig.tight_layout()
    return _save(fig, out, "F10_state_distribution")


LOG_LINE = re.compile(r"step (\d+)/\d+ loss ([\d.]+) \|g\| lora ([\d.]+) wdown ([\d.]+) new ([\d.]+)")


def noclip_training(out: Path) -> Path:
    plt = _plt()
    runs = (("Hebbian, no clip", "hebbian-noclip", BLUE, "o"), ("Delta, no clip", "delta-noclip-probe", ORANGE, "s"))
    fig, (loss_ax, grad_ax) = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for label, name, colour, marker in runs:
        points = [tuple(float(g) for g in m.groups()) for line in (NOCLIP_LOGS / f"{name}.log").read_text().splitlines()
                  if (m := LOG_LINE.search(line))]
        steps = [p[0] for p in points]
        loss_ax.plot(steps, [p[1] for p in points], color=colour, marker=marker, markersize=3.5, label=label)
        grad_ax.plot(steps, [p[3] for p in points], color=colour, marker=marker, markersize=3.5, label=label)
    loss_ax.set_ylabel("Training loss")
    loss_ax.set_title("(a) Training loss")
    grad_ax.set_yscale("log")
    grad_ax.axhline(10, color=INK, linewidth=0.9, linestyle="--")
    grad_ax.annotate("gradient clip", (grad_ax.get_xlim()[1], 10), xytext=(-4, 4), textcoords="offset points",
                     ha="right", fontsize=8, color=INK)
    grad_ax.set_ylabel(r"Gradient norm of $W_{\mathrm{down}}$")
    grad_ax.set_title("(b) Gradient norm")
    for ax in (loss_ax, grad_ax):
        ax.set_xlabel("Training step")
        _axes(ax)
    loss_ax.legend(loc="upper right")
    fig.tight_layout(w_pad=2)
    return _save(fig, out, "F11_noclip_training")


def gates(out: Path) -> Path:
    plt = _plt()
    cases = (("Hebbian, with clip, step 600", HEBBIAN, 600, BLUE), ("Delta, with clip, step 600", DELTA, 600, ORANGE),
             ("Delta, with clip, step 100", DELTA, 100, ORANGE), ("Delta, no clip, step 100", "delta-noclip", 100, GRAY))
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    for y, (name, run, step, colour) in enumerate(cases):
        value = statistics.median(r.gate_mean for r in _rows(run, step, "boundary"))
        ax.barh(y, value, color=colour, height=0.55)
        ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(cases)), [c[0] for c in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Output gate, median")
    _axes(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, out, "F12_gates")


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


def gated_by_slice(out: Path) -> Path:
    plt = _plt()
    cases = (("ungated carry, step 600", HEBBIAN, 600, GRAY, "s"), ("gated carry, step 600", GATED, 600, BLUE, "o"),
             ("gated carry, step 1049", GATED_FULL, FULL_STEP, NAVY, "D"))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for label, run, step, colour, marker in cases:
        points = paper_analysis.position_estimates(_rows(run, step, "regime_slices"), rng=NumpyRng(0), n_boot=N_BOOT)
        cells = sorted((i, e.interval) for i, e in points if e.measure == "between_slices" and i > 0)
        _band(ax, [i + 1 for i, _ in cells], [g for _, g in cells], colour, marker=marker, label=label)
    _zero(ax)
    ax.set_xticks(range(2, 9))
    ax.set_xlabel("Slice of the document")
    ax.set_ylabel("Gain in perplexity")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    _axes(ax)
    fig.tight_layout()
    return _save(fig, out, "F14_carry_design_by_slice")


DEV_LEVELS = (("gain_self", "same\ndocument"), ("gain_related", "related\narticle"),
              ("gain_random", "random\narticle"), ("gain_other_source", "other\nsource"))
DEV_CASES = (("ungated carry", HEBBIAN, 600, GRAY, "s", True),
             ("gated carry, seed 42", GATED, 600, BLUE, "o", True),
             ("gated carry, seed 43", GATED_43, 600, BLUE, "D", False),
             ("gated carry, step 1049", GATED_FULL, FULL_STEP, NAVY, "^", True))


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


def dev_relatedness(out: Path) -> Path:
    plt = _plt()
    values = {}
    for label, run, step, *_ in DEV_CASES:
        estimates = relatedness_eval.estimates(_rows(run, step, "relatedness"), rng=NumpyRng(0), n_boot=N_BOOT)
        values[label] = {e.measure: (e.interval.point, e.interval.low, e.interval.high)
                         for e in estimates if e.source == ALL_SOURCES}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    _dots(ax, DEV_LEVELS, [(l, c, m, f) for l, _, _, c, m, f in DEV_CASES], values)
    ax.set_xlabel("Text carried into the target")
    ax.set_ylabel("Gain in perplexity")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    return _save(fig, out, "F16_relatedness")


CONF_LEVELS = (("gain_self", "same\ndocument"), ("gain_related", "related\narticle"), ("gain_generic", "generic\ncarry"),
               ("gain_random", "random\narticle"), ("gain_chain_10", "chain of\n10 articles"),
               ("gain_other_source", "other\nsource"))


def _fresh(run: str, step: int) -> dict:
    bundle = sorted(p for p in (RESULTS_DIR / run / f"step_{step}").glob("*") if not is_development(p))[-1]
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


ABLATION = (("ungated carry, gate bias $-2$", {42: HEBBIAN}),
            ("closed gate only (C)", {42: "ablate-c-only", 43: "ablate-c-only-s43"}),
            ("separate path, not trained through (A, C)", {42: "ablate-a-c", 43: "ablate-a-c-s43"}),
            ("trained through, no separate path (B, C)", {42: "ablate-b-c", 43: "ablate-b-c-s43"}),
            ("gated carry (A, B, C)", {42: GATED_200, 43: GATED_43}))


def ablation(out: Path) -> Path:
    plt = _plt()
    pooled = {(label, seed): _pooled(run, 200) for label, runs in ABLATION for seed, run in runs.items()}
    panels = (("between_slices", "(a) Between slices"), ("vs_matched_lora", "(b) Lead over the LoRA-only control"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for ax, (measure, title) in zip(axes, panels):
        for seed, colour, marker, offset, filled in ((42, BLUE, "o", -0.14, True), (43, BLUE, "D", 0.14, False)):
            ys, intervals = [], []
            for y, (label, runs) in enumerate(ABLATION):
                if seed in runs:
                    ys.append(y + offset)
                    intervals.append(pooled[(label, seed)][measure].interval)
            ax.errorbar([i.point for i in intervals], ys,
                        xerr=[[i.point - i.low for i in intervals], [i.high - i.point for i in intervals]],
                        fmt=marker, color=colour, capsize=2, linewidth=1.1, markersize=5,
                        markerfacecolor=colour if filled else "white", markeredgewidth=1.1, label=f"seed {seed}")
        _zero(ax, vertical=True)
        ax.set_title(title)
        ax.set_xlabel("Gain in perplexity")
        _axes(ax, grid="x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ABLATION)), [label for label, _ in ABLATION])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.08))
    fig.tight_layout(w_pad=1.5)
    return _save(fig, out, "F17_ablation")


def _box(ax, xy, w, h, text, *, fill="white", edge=INK_2, size=8.5, bold=False):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal")


def _arrow(ax, a, b, *, colour=INK_2, style="-|>", dashed=False, rad=0.0, lw=1.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=colour, linewidth=lw,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def _canvas(plt, w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


BLUE_FILL = "#dbe8f8"
ORANGE_FILL = "#fbe1d6"


def _line(ax, points, colour=INK_2, lw=1.0):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=colour, linewidth=lw, solid_capstyle="round")


def _op(ax, xy, text, *, colour=INK_2, r=0.17, size=8.5):
    import matplotlib.pyplot as plt

    ax.add_patch(plt.Circle(xy, r, facecolor="white", edgecolor=colour, linewidth=0.9, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=size, color=INK, zorder=4)


def _note(ax, xy, text, colour=INK_2, **kw):
    ax.text(*xy, text, fontsize=7.5, color=colour, **{"ha": "center", "va": "center", **kw})


def diagram_layer(out: Path) -> Path:
    plt = _plt()
    fig, ax = _canvas(plt, WIDTH, 3.25)
    top, mid, low = 2.75, 1.85, 0.95
    _box(ax, (0.45, mid), 0.7, 0.42, "$z$", size=10)
    _note(ax, (0.45, mid - 0.36), "MLP activation")
    bus = 1.0
    _line(ax, [(0.8, mid), (bus, mid)])
    _line(ax, [(bus, top), (bus, low)])
    _box(ax, (1.85, top), 1.2, 0.42, r"$W_{\mathrm{down}}$")
    _note(ax, (1.85, top + 0.33), "pretrained")
    _box(ax, (1.85, mid), 1.2, 0.42, r"$S_{\mathrm{in}}$", fill=BLUE_FILL, edge=BLUE)
    _note(ax, (1.85, mid + 0.33), "this slice, from 0", BLUE)
    _box(ax, (1.85, low), 1.2, 0.42, r"$S_{\mathrm{carried}}$", fill=ORANGE_FILL, edge=ORANGE)
    for y in (top, mid, low):
        _arrow(ax, (bus, y), (1.25, y))
    _note(ax, (1.85, low + 0.33), "earlier slices", ORANGE)
    _op(ax, (3.2, mid), "$g$", colour=BLUE)
    _note(ax, (3.2, mid + 0.33), "gate", BLUE)
    _arrow(ax, (2.45, mid), (3.03, mid))
    _box(ax, (2.95, low), 0.7, 0.36, "clip", edge=ORANGE, size=8)
    _note(ax, (2.95, low - 0.33), "keeps direction", ORANGE)
    _arrow(ax, (2.45, low), (2.6, low))
    _op(ax, (3.75, low), "$g_c$", colour=ORANGE)
    _note(ax, (3.75, low - 0.33), "own gate", ORANGE)
    _arrow(ax, (3.3, low), (3.58, low))
    _op(ax, (4.4, low), r"$\alpha$", colour=ORANGE)
    _note(ax, (4.4, low - 0.33), "starts at 0", ORANGE)
    _arrow(ax, (3.92, low), (4.23, low))
    plus = (5.15, mid)
    _op(ax, plus, "+", size=11)
    _line(ax, [(2.45, top), (plus[0], top)])
    _arrow(ax, (plus[0], top), (plus[0], mid + 0.17))
    _arrow(ax, (3.37, mid), (plus[0] - 0.17, mid))
    _line(ax, [(4.57, low), (plus[0], low)])
    _arrow(ax, (plus[0], low), (plus[0], mid - 0.17))
    _box(ax, (5.9, mid), 0.7, 0.42, "output", size=8.5)
    _arrow(ax, (plus[0] + 0.17, mid), (5.55, mid))
    _box(ax, (1.85, 0.18), 2.3, 0.3, r"end of each slice:  $S \leftarrow 0.9\,S + \Delta S_{\mathrm{item}}$",
         fill="white", edge=ORANGE, size=7.5)
    _arrow(ax, (1.85, 0.33), (1.85, low - 0.21), colour=ORANGE, dashed=True)
    return _save(fig, out, "D1_gated_carry_layer")


def diagram_protocol(out: Path) -> Path:
    plt = _plt()
    fig, ax = _canvas(plt, WIDTH, 3.35)
    ax.text(0.02, 3.18, "(a) Training: one document per session", fontsize=9, fontweight="bold", color=INK)
    xs = (0.75, 2.05, 3.35, 4.65)
    y = 2.55
    for i, x in enumerate(xs):
        _box(ax, (x, y), 0.85, 0.4, f"slice {i + 1}", fill=BLUE_FILL, edge=BLUE)
    for a, b in zip(xs, xs[1:]):
        _arrow(ax, (a + 0.45, y), (b - 0.45, y), colour=ORANGE, lw=1.4)
    _note(ax, ((xs[0] + xs[1]) / 2, y + 0.13), "carry", ORANGE)
    for a, b in ((xs[1], xs[0]), (xs[3], xs[2])):
        _arrow(ax, (a - 0.1, y - 0.21), (b + 0.1, y - 0.21), dashed=True, rad=-0.45)
    _note(ax, ((xs[0] + xs[1]) / 2, y - 0.62), "gradient")
    _note(ax, ((xs[2] + xs[3]) / 2, y - 0.62), "gradient")
    ax.text(5.2, y, "reset before the\nnext document", va="center", fontsize=7.5, color=INK_2)
    ax.text(0.02, 1.55, "(b) Testing across documents", fontsize=9, fontweight="bold", color=INK)
    r1, r2 = 1.12, 0.22
    _box(ax, (0.95, r1), 1.6, 0.42, "prime, 1024 tokens", fill=ORANGE_FILL, edge=ORANGE, size=8)
    _box(ax, (0.95, r2), 1.6, 0.36, "no prime", size=8)
    _arrow(ax, (1.75, r1), (2.55, r1), colour=ORANGE, lw=1.4)
    _note(ax, (2.15, r1 - 0.43), "folded into the carry,\nthen removed from attention", ORANGE)
    _arrow(ax, (1.75, r2), (2.55, r2))
    _note(ax, (2.15, r2 + 0.13), "carry reset")
    _box(ax, (3.35, r1), 1.6, 0.42, "target, scored", fill=BLUE_FILL, edge=BLUE, size=8)
    _box(ax, (3.35, r2), 1.6, 0.36, "same target, scored", size=8)
    _line(ax, [(4.2, r1), (4.35, r1), (4.35, r2), (4.2, r2)])
    ax.text(4.47, (r1 + r2) / 2, "gain = difference\nin perplexity", va="center", fontsize=7.5, color=INK)
    return _save(fig, out, "D2_protocol")


def diagram_reset(out: Path) -> Path:
    plt = _plt()
    fig, ax = _canvas(plt, WIDTH, 1.55)
    xs = (1.95, 3.55, 5.15)
    ax.text(0.02, 1.15, "reset\n(In-Place TTT)", va="center", fontsize=8.5, color=INK)
    ax.text(0.02, 0.35, "carried\n(this report)", va="center", fontsize=8.5, color=INK)
    for i, x in enumerate(xs):
        _box(ax, (x, 1.15), 0.95, 0.38, f"document {i + 1}", size=8)
        ax.text(x, 0.78, r"$S$ starts at 0", ha="center", fontsize=7.5, color=INK_2)
        _box(ax, (x, 0.35), 0.95, 0.38, f"document {i + 1}", fill=BLUE_FILL, edge=BLUE, size=8)
    for a, b in zip(xs, xs[1:]):
        _arrow(ax, (a + 0.5, 0.35), (b - 0.5, 0.35), colour=ORANGE, lw=1.4)
        ax.text((a + b) / 2, 0.47, r"$S_{\mathrm{carried}}$", ha="center", fontsize=8, color=ORANGE)
    return _save(fig, out, "D3_reset_vs_carry")


TOKEN_RUNS = ((GATED, "seed 42", "-"), (GATED_43, "seed 43", "--"))
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
    threshold = json.loads((RESULTS_DIR / "paper" / "manifest.json").read_text())["clip_threshold"]
    for make in (diagram_layer, diagram_protocol, diagram_reset, ungated_gains, ungated_by_slice, absolute_ppl,
                 chain_states, noclip_training, gates, gated_curve, gated_by_slice, dev_relatedness,
                 conf_relatedness, ablation, token_position):
        print(make(args.out))
    print(repeat_mechanism(args.out, threshold))
    print(state_distribution(args.out, threshold))


if __name__ == "__main__":
    main()
