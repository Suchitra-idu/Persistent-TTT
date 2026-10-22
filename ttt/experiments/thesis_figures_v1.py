"""Extra thesis figures from the same paper_eval bundles. Local, no GPU.

    python -m ttt.experiments.thesis_figures_v1

Writes figures/F5..F12 under results/paper/ and the per-source numbers behind
them as tables/T7_sources_by_step and tables/T8_domain_correlates.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import evidence, paper_analysis
from ttt.core.metrics import ALL_SOURCES, geometric_mean_ppl
from ttt.core.stats import ranks
from ttt.core.types import COLD_CARRY, FRESH, LORA_ONLY
from ttt.experiments import paper_eval_v1
from ttt.experiments._evidence_figures import (
    GRID, INK, MUTED, SERIES, SURFACE, _chrome, _ordered_sources, _pyplot, _save,
)
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR, newest_bundle

OUT = RESULTS_DIR / "paper"
CONTROL = "lora-only-matched"
RUNS = {"hebbian": "ttt-v1.1", "delta_chunk": "ttt-v1.1-clean-deltachunk"}
STEPS = (200, 400, 600)
CONTROL_COLOUR = "#1baf7a"
MARKERS = {"LoRA-only": "^", "hebbian": "o", "delta_chunk": "s"}
COLOURS = {"LoRA-only": CONTROL_COLOUR, "hebbian": SERIES[0], "delta_chunk": SERIES[1]}
HEAT = ("#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
NOCLIP_LOGS = OUT / "inputs" / "noclip_training"
LOG_LINE = re.compile(r"step (\d+)/\d+ loss ([\d.]+) \|g\| lora ([\d.]+) wdown ([\d.]+) new ([\d.]+)")
N_BOOT = paper_eval_v1.DEFAULT_N_BOOT


def _rows(run: str, step: int, name: str):
    return newest_bundle(RESULTS_DIR, run, step).rows(name)


def _short(source: str) -> str:
    return source.replace("RedPajama", "")


def _title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")


def absolute_ppl(path: Path) -> str:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    base = geometric_mean_ppl([r.ppl for r in _rows(CONTROL, STEPS[-1], "regimes") if r.regime == FRESH])
    series = {"LoRA-only": (CONTROL, LORA_ONLY)} | {label: (run, COLD_CARRY) for label, run in RUNS.items()}
    for label, (run, regime) in series.items():
        values = [geometric_mean_ppl([r.ppl for r in _rows(run, s, "regimes") if r.regime == regime]) for s in STEPS]
        ax.plot(STEPS, values, color=COLOURS[label], marker=MARKERS[label], markersize=6, linewidth=2, label=label)
        ax.annotate(f"{values[-1]:.3f}", (STEPS[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("training step")
    ax.set_ylabel("geometric-mean perplexity")
    ax.set_title(f"Base model {base:.3f}, off the scale above", loc="left", fontsize=9, color=MUTED)
    ax.margins(x=0.18)
    ax.legend(frameon=False, fontsize=8)
    _chrome(ax)
    _title(fig, "Held-out perplexity of each model (106 documents, lower is better)")
    fig.tight_layout()
    return _save(fig, str(path))


def source_estimates():
    """{(label, measure, step): {source: Estimate}} for the per-source heatmaps."""
    out: dict = defaultdict(dict)
    for step in STEPS:
        base_rows = _rows(CONTROL, step, "regimes")
        for label, run in RUNS.items():
            rows = paper_analysis.rebased(_rows(run, step, "regimes"), base_rows)
            estimates = evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT)
            estimates += paper_analysis.matched_estimates(base_rows, rows, rng=NumpyRng(0), n_boot=N_BOOT)
            for e in estimates:
                out[(label, e.measure, step)][e.source] = e
    return out


def source_heatmaps(estimates, path: Path) -> str:
    """Relative gain in percent of the reference perplexity, so sources compare."""
    plt = _pyplot()
    panels = [(label, measure) for measure in ("within", paper_analysis.VS_MATCHED) for label in RUNS]
    sources = [s for s in _ordered_sources(estimates[("hebbian", "within", STEPS[0])]) if s != ALL_SOURCES]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6), squeeze=False)
    for ax, (label, measure) in zip(axes[0], panels):
        grid = [[100 * estimates[(label, measure, s)][src].relative for s in STEPS] for src in sources]
        top = max(max(row) for row in grid)
        ax.imshow(grid, cmap=_heat_map(), vmin=0, vmax=top, aspect="auto")
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                star = "*" if estimates[(label, measure, STEPS[j])][sources[i]].interval.excludes_zero else ""
                ax.text(j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=7,
                        color=SURFACE if value > 0.6 * top else INK)
        ax.set_xticks(range(len(STEPS)), [str(s) for s in STEPS])
        ax.set_yticks(range(len(sources)), [_short(s) for s in sources] if ax is axes[0][0] else [])
        ax.set_xlabel("training step")
        name = "within document" if measure == "within" else "vs. matched LoRA"
        ax.set_title(f"{label}, {name}", loc="left", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    _title(fig, "Relative gain per source in percent (* = 95% CI excludes zero)")
    fig.tight_layout()
    return _save(fig, str(path))


def _heat_map():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blue", HEAT)


def domain_correlates(estimates, path: Path) -> tuple[str, list[tuple]]:
    """Per-source relative within gain against the true base perplexity."""
    plt = _pyplot()
    base_rows = _rows(CONTROL, STEPS[-1], "regimes")
    by_source = defaultdict(list)
    for r in base_rows:
        if r.regime == FRESH:
            by_source[r.source].append(r.ppl)
    base = {s: geometric_mean_ppl(v) for s, v in by_source.items()}
    sources = sorted(base)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    table = []
    for label in RUNS:
        gains = [100 * estimates[(label, "within", STEPS[-1])][s].relative for s in sources]
        rho = _spearman([base[s] for s in sources], gains)
        table.append((label, len(sources), rho))
        ax.scatter([base[s] for s in sources], gains, color=COLOURS[label], marker=MARKERS[label],
                   s=40, label=f"{label}, Spearman ρ = {rho:+.2f}", zorder=3)
        for s, g in zip(sources, gains):
            ax.annotate(_short(s), (base[s], g), xytext=(5, 3), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis, (5, 10, 20, 40))
    ax.set_xlabel("base-model perplexity of the source (log)")
    ax.set_ylabel("within-document gain, % of reference")
    ax.legend(frameon=False, fontsize=8)
    _chrome(ax)
    _title(fig, "Does TTT help harder domains more? Step 600, six sources")
    fig.tight_layout()
    return _save(fig, str(path)), table


def _plain_log_ticks(axis, ticks) -> None:
    """Plain numbers on a log axis instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if ticks is not None:
        axis.set_ticks(ticks)
    else:
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_formatter(NullFormatter())


def _spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def chain_states(path: Path) -> str:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(RUNS), figsize=(10, 3.6), squeeze=False)
    arms = (("persist", "carried across documents"), ("doc_reset", "reset at each document"),
            ("slice_reset", "reset at each slice"))
    for ax, (label, run) in zip(axes[0], RUNS.items()):
        rows = _rows(run, STEPS[-1], "boundary")
        for index, (arm, name) in enumerate(arms):
            positions = sorted({r.chain_position for r in rows})
            medians = [statistics.median(r.state_ratio for r in rows if r.arm == arm and r.chain_position == p)
                       for p in positions]
            colour = (SERIES[1], SERIES[0], CONTROL_COLOUR)[index]
            ax.plot([p + 1 for p in positions], medians, color=colour, marker="os^"[index],
                    markersize=6, linewidth=2, label=name)
        ax.set_yscale("log")
        _plain_log_ticks(ax.yaxis, None)
        ax.set_xticks([p + 1 for p in positions])
        ax.set_xlabel("document position in the chain")
        ax.set_ylabel("state ratio, median (log)")
        ax.set_title(label, loc="left")
        _chrome(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center")
    _title(fig, "Carrying state makes it larger, not more useful (step 600, 6 chains of 5 documents, one per source)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, str(path))


def state_distribution(threshold: float, path: Path) -> str:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, run in RUNS.items():
        ratios = [r.state_ratio for name in ("boundary", "repeat", "swap") for r in _rows(run, STEPS[-1], name)]
        ratios += [r.state_ratio_final for r in _rows(run, STEPS[-1], "regimes") if r.regime == COLD_CARRY]
        logs = [math.log10(v) for v in ratios if v > 0]
        ax.hist(logs, bins=40, color=COLOURS[label], alpha=0.55, label=f"{label}, {len(logs)} measurements")
    ax.axvline(math.log10(threshold), color=INK, linewidth=1.5, linestyle="--")
    ax.annotate(f"clip binds above {threshold:.2f}", (math.log10(threshold), 0), xytext=(6, 60),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("state ratio ‖ηS‖ / ‖W₀‖, log10")
    ax.set_ylabel("measurements")
    ax.legend(frameon=False, fontsize=8)
    _chrome(ax)
    _title(fig, "Every measured state is far above the clip threshold (step 600)")
    fig.tight_layout()
    return _save(fig, str(path))


def noclip_training(path: Path) -> str:
    plt = _pyplot()
    runs = {
        "hebbian, no clip": ("hebbian-noclip", SERIES[0], "o"),
        "delta_chunk, no clip": ("delta-noclip-probe", SERIES[1], "s"),
    }
    fig, (loss_ax, grad_ax) = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, (name, colour, marker) in runs.items():
        points = [tuple(float(g) for g in LOG_LINE.search(line).groups())
                  for line in (NOCLIP_LOGS / f"{name}.log").read_text().splitlines() if LOG_LINE.search(line)]
        steps = [p[0] for p in points]
        loss_ax.plot(steps, [p[1] for p in points], color=colour, marker=marker, markersize=4, linewidth=2, label=label)
        grad_ax.plot(steps, [p[3] for p in points], color=colour, marker=marker, markersize=4, linewidth=2, label=label)
    loss_ax.set_ylabel("training loss, window mean")
    grad_ax.set_yscale("log")
    grad_ax.axhline(10, color=INK, linewidth=1.2, linestyle="--")
    grad_ax.annotate("gradient clip 10", (0, 10), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    grad_ax.set_ylabel("W₀ gradient norm (log)")
    for ax in (loss_ax, grad_ax):
        ax.set_xlabel("training step")
        _chrome(ax)
    loss_ax.legend(frameon=False, fontsize=8)
    _title(fig, "Training without the read-time clip, same data order and settings")
    fig.tight_layout()
    return _save(fig, str(path))


def gates(path: Path) -> str:
    plt = _pyplot()
    cases = (
        ("hebbian, clip, step 600", "ttt-v1.1", 600, SERIES[0]),
        ("delta_chunk, clip, step 600", "ttt-v1.1-clean-deltachunk", 600, SERIES[1]),
        ("delta_chunk, clip, step 100", "ttt-v1.1-clean-deltachunk", 100, SERIES[1]),
        ("delta_chunk, no clip, step 100", "delta-noclip", 100, MUTED),
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names, values = [], []
    for name, run, step, colour in cases:
        median = statistics.median(r.gate_mean for r in _rows(run, step, "boundary"))
        names.append(name)
        values.append(median)
        ax.barh(len(names) - 1, median, color=colour, height=0.6)
        ax.annotate(f"{median:.3f}", (median, len(names) - 1), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("output gate, median over boundary-suite measurements")
    _chrome(ax, grid_axis="x")
    _title(fig, "The optimiser closes the gate when the fast weights are unusable")
    fig.tight_layout()
    return _save(fig, str(path))


def swap_matrices(figures: Path) -> list[str]:
    from ttt.experiments import _evidence_figures

    made = []
    for label, run in RUNS.items():
        estimates = evidence.swap_estimates(_rows(run, STEPS[-1], "swap"), rng=NumpyRng(0), n_boot=N_BOOT)
        made.append(_evidence_figures.swap_heatmap(estimates, str(figures / f"F9_swap_{label}.pdf")))
    return made


def swap_summary() -> list[str]:
    """Primed against unprimed, split into same-source and other-source pairs."""
    lines = ["run,pairs,n_pairs,mean_delta_ppl,significantly_worse,significantly_better,min_delta_ppl"]
    for label, run in RUNS.items():
        primed = [e for e in evidence.swap_estimates(_rows(run, STEPS[-1], "swap"), rng=NumpyRng(0), n_boot=N_BOOT)
                  if e.measure.startswith("prime:")]
        for name, keep in (("same_source", True), ("other_source", False)):
            group = [e for e in primed if (e.measure.removeprefix("prime:") == e.source) == keep]
            points = [e.interval.point for e in group]
            worse = sum(e.interval.high < 0 for e in group)
            better = sum(e.interval.low > 0 for e in group)
            lines.append(f"{label},{name},{len(group)},{statistics.fmean(points):.6g},{worse},{better},{min(points):.6g}")
    return lines


def repeat_bounds() -> list[str]:
    """Every later reading against the first, pooled over sources."""
    lines = ["run,repeat,delta_ppl,ci_low,ci_high,excludes_zero"]
    for label, run in RUNS.items():
        for p in evidence.repeat_curve(_rows(run, STEPS[-1], "repeat"), rng=NumpyRng(0), n_boot=N_BOOT):
            if p.source == ALL_SOURCES:
                lines.append(f"{label},{p.repeat},{p.gap.point:.6g},{p.gap.low:.6g},{p.gap.high:.6g},"
                             f"{int(p.gap.low > 0 or p.gap.high < 0)}")
    return lines


def write_tables(estimates, correlates) -> None:
    tables = OUT / "tables"
    lines = ["run,measure,step,source,delta_ppl,ci_low,ci_high,relative,excludes_zero"]
    for (label, measure, step), by_source in sorted(estimates.items()):
        if measure not in ("within", paper_analysis.VS_MATCHED, "lora"):
            continue
        for source, e in sorted(by_source.items()):
            lines.append(f"{label},{measure},{step},{source},{e.interval.point:.6g},{e.interval.low:.6g},"
                         f"{e.interval.high:.6g},{e.relative:.6g},{int(e.interval.excludes_zero)}")
    (tables / "T7_sources_by_step.csv").write_text("\n".join(lines) + "\n")
    rows = ["run,n_sources,spearman_rho_within_vs_base_ppl"] + [f"{l},{n},{rho:.4f}" for l, n, rho in correlates]
    (tables / "T8_domain_correlates.csv").write_text("\n".join(rows) + "\n")
    (tables / "T9_swap_summary.csv").write_text("\n".join(swap_summary()) + "\n")
    (tables / "T10_repeat_curve.csv").write_text("\n".join(repeat_bounds()) + "\n")


CARRY_DESIGN = "hebbian-carry-b"
CARRY_DESIGN_CASES = (
    ("plain Hebbian, step 200", "ttt-v1.1", 200),
    ("carry design, step 100", CARRY_DESIGN, 100),
    ("carry design, step 200", CARRY_DESIGN, 200),
)
CARRY_DESIGN_MEASURES = (
    ("between_slices", "Carried from earlier slices\nof the same document"),
    ("cross_slice", "Carried across slices,\nboundary suite"),
    ("cross_doc", "Carried into a\ndifferent document"),
    ("within", "Fast weights within\na slice"),
)


def carry_design_estimates() -> dict[tuple[str, str], object]:
    """Pooled estimates for each case; the pre-fix TTT bundle is rebased onto the true base."""
    base = _rows(CONTROL, 200, "regimes")
    out = {}
    for label, run, step in CARRY_DESIGN_CASES:
        rows = paper_analysis.rebased(_rows(run, step, "regimes"), base)
        estimates = (
            evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT)
            + evidence.boundary_estimates(_rows(run, step, "boundary"), rng=NumpyRng(0), n_boot=N_BOOT)
        )
        if step == 200:
            estimates += paper_analysis.matched_estimates(base, rows, rng=NumpyRng(0), n_boot=N_BOOT)
        for e in estimates:
            if e.source == ALL_SOURCES:
                out[(label, e.measure)] = e
    return out


def carry_design_figure(estimates, path: Path) -> str:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(CARRY_DESIGN_MEASURES), figsize=(12, 3.4), squeeze=False)
    colours = (SERIES[0], SERIES[1], SERIES[1])
    for ax, (measure, title) in zip(axes[0], CARRY_DESIGN_MEASURES):
        for y, (label, _, _) in enumerate(CARRY_DESIGN_CASES):
            e = estimates[(label, measure)]
            i = e.interval
            filled = i.excludes_zero
            ax.hlines(y, i.low, i.high, color=colours[y], linewidth=2)
            ax.plot(i.point, y, "osD"[y], markersize=7, markeredgewidth=2, markeredgecolor=colours[y],
                    markerfacecolor=colours[y] if filled else SURFACE)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(len(CARRY_DESIGN_CASES)),
                      [c[0] for c in CARRY_DESIGN_CASES] if ax is axes[0][0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("Δppl  (+ = lower perplexity)")
        _chrome(ax, grid_axis="x")
    _title(fig, "The carry design against plain Hebbian (95% CI; filled = interval excludes zero)")
    fig.tight_layout()
    return _save(fig, str(path))


def carry_design_by_position(path: Path) -> str:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cases = (("plain Hebbian, step 200", "ttt-v1.1", 200, SERIES[0], "o"),
             ("carry design, step 200", CARRY_DESIGN, 200, SERIES[1], "D"))
    for label, run, step, colour, marker in cases:
        points = sorted((i, e.interval) for i, e in paper_analysis.position_estimates(
            _rows(run, step, "regime_slices"), rng=NumpyRng(0), n_boot=N_BOOT) if e.measure == "between_slices")
        xs = [i + 1 for i, _ in points]
        ax.fill_between(xs, [g.low for _, g in points], [g.high for _, g in points], color=colour, alpha=0.18, linewidth=0)
        ax.plot(xs, [g.point for _, g in points], color=colour, marker=marker, markersize=6, linewidth=2, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("slice of the document")
    ax.set_ylabel("Δppl, carried minus reset")
    ax.legend(frameon=False, fontsize=8)
    _chrome(ax)
    _title(fig, "Carry from earlier slices of the same document, by slice")
    fig.tight_layout()
    return _save(fig, str(path))


def carry_design_table(estimates) -> list[str]:
    lines = ["case,measure,n_docs,delta_ppl,ci_low,ci_high,excludes_zero,win_rate"]
    for (label, measure), e in sorted(estimates.items()):
        i = e.interval
        lines.append(f"{label},{measure},{e.n_docs},{i.point:.6g},{i.low:.6g},{i.high:.6g},"
                     f"{int(i.excludes_zero)},{e.win_rate:.4f}")
    return lines


def main() -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    threshold = json.loads((OUT / "manifest.json").read_text())["clip_threshold"]
    estimates = source_estimates()
    made = [
        absolute_ppl(figures / "F5_absolute_ppl.pdf"),
        source_heatmaps(estimates, figures / "F6_source_heatmaps.pdf"),
    ]
    correlate_figure, correlates = domain_correlates(estimates, figures / "F7_domain_correlates.pdf")
    made += [
        correlate_figure,
        chain_states(figures / "F8_chain_states.pdf"),
        *swap_matrices(figures),
        state_distribution(threshold, figures / "F10_state_distribution.pdf"),
        noclip_training(figures / "F11_noclip_training.pdf"),
        gates(figures / "F12_gates.pdf"),
    ]
    design = carry_design_estimates()
    made += [
        carry_design_figure(design, figures / "F13_carry_design.pdf"),
        carry_design_by_position(figures / "F14_carry_design_by_slice.pdf"),
    ]
    (OUT / "tables" / "T11_carry_design.csv").write_text("\n".join(carry_design_table(design)) + "\n")
    write_tables(estimates, correlates)
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
