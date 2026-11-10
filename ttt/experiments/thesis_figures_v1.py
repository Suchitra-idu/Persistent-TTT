"""Cross-run tables T7, T8 and T11 to T16 from the paper_eval bundles. Local, no GPU.

    python -m ttt.experiments.thesis_figures_v1
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import context_eval, evidence, paper_analysis, relatedness_eval
from ttt.core.metrics import ALL_SOURCES, geometric_mean_ppl
from ttt.core import stats
from ttt.core.stats import ranks
from ttt.core.types import FRESH
from ttt.experiments import paper_eval_v1
from ttt.experiments._evidence_figures import MUTED, SERIES
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR, Bundle, is_development, newest_bundle, pinned

OUT = RESULTS_DIR / "paper"
CONTROL = "lora-only-matched"
RUNS = {"Hebbian": "ttt-v1.1", "Delta": "ttt-v1.1-clean-deltachunk"}
STEPS = (200, 400, 600)
CONTROL_COLOUR = "#1baf7a"
N_BOOT = paper_eval_v1.DEFAULT_N_BOOT


def _rows(run: str, step: int, name: str):
    """From the pinned bundle, else the newest at that step that measured
    `name`; a step can hold several bundles, each with different suites."""
    pin = pinned(RESULTS_DIR, run, step, name)
    if pin is not None:
        return Bundle(run=run, step=step, path=pin).rows(name)
    for stamp in sorted((RESULTS_DIR / run / f"step_{step}").glob("*/raw"), reverse=True):
        if (stamp / f"{name}.json").is_file() and is_development(stamp.parent):
            return Bundle(run=run, step=step, path=stamp.parent).rows(name)
    return newest_bundle(RESULTS_DIR, run, step).rows(name)


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


def domain_correlates(estimates) -> list[tuple]:
    """Per-source relative within gain against the true base perplexity."""
    base_rows = _rows(CONTROL, STEPS[-1], "regimes")
    by_source = defaultdict(list)
    for r in base_rows:
        if r.regime == FRESH:
            by_source[r.source].append(r.ppl)
    base = {s: geometric_mean_ppl(v) for s, v in by_source.items()}
    sources = sorted(base)
    table = []
    for label in RUNS:
        gains = [100 * estimates[(label, "within", STEPS[-1])][s].relative for s in sources]
        table.append((label, len(sources), _spearman([base[s] for s in sources], gains)))
    return table


def _spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


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


CARRY_DESIGN = "hebbian-carry-b"
CARRY_DESIGN_LONG = "hebbian-carry-600"
CARRY_DESIGN_CASES = (
    ("ungated carry, step 200", "ttt-v1.1", 200),
    ("ungated carry, step 600", "ttt-v1.1", 600),
    ("gated carry, step 100", CARRY_DESIGN, 100),
    ("gated carry, step 200", CARRY_DESIGN, 200),
    ("gated carry, step 400", CARRY_DESIGN_LONG, 400),
    ("gated carry, step 600", CARRY_DESIGN_LONG, 600),
)


def carry_design_estimates() -> dict[tuple[str, str], object]:
    """Pooled estimates for each case; the pre-fix TTT bundle is rebased onto the true base."""
    out = {}
    for label, run, step in CARRY_DESIGN_CASES:
        base = _rows(CONTROL, step if step in STEPS else 200, "regimes")
        rows = paper_analysis.rebased(_rows(run, step, "regimes"), base)
        estimates = (
            evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT)
            + evidence.boundary_estimates(_rows(run, step, "boundary"), rng=NumpyRng(0), n_boot=N_BOOT)
        )
        if step in STEPS:
            estimates += paper_analysis.matched_estimates(base, rows, rng=NumpyRng(0), n_boot=N_BOOT)
        for e in estimates:
            if e.source == ALL_SOURCES:
                out[(label, e.measure)] = e
    return out


# (label, run, raw rows, colour, marker, line style); the two carry seeds share a colour.
RELATEDNESS_CASES = (
    ("ungated carry", "ttt-v1.1", "relatedness", MUTED, "o", "-"),
    ("gated carry, seed 42", CARRY_DESIGN_LONG, "relatedness", SERIES[1], "s", "-"),
    ("gated carry, seed 43", "hebbian-carry-600-s43", "relatedness", SERIES[1], "D", "--"),
    ("dynamic evaluation, tuned for related", CONTROL, "dyneval_related", SERIES[0], "^", "-"),
    ("dynamic evaluation, tuned for same document", CONTROL, "dyneval_self", CONTROL_COLOUR, "v", "--"),
)


def relatedness_estimates() -> dict[tuple[str, str], object]:
    out = {}
    for label, run, raw, *_ in RELATEDNESS_CASES:
        for e in relatedness_eval.estimates(_rows(run, 600, raw), rng=NumpyRng(0), n_boot=N_BOOT):
            if e.source == ALL_SOURCES:
                out[(label, e.measure)] = e
    return out


def carry_vs_dyneval() -> list[str]:
    """Per arm, the carry's log-perplexity gain minus dynamic evaluation's, paired
    by case, in percent; dynamic evaluation tuned for the related arm."""
    dyn = _rows(CONTROL, 600, "dyneval_related")
    lines = ["carry,arm,n_docs,difference_percent,ci_low,ci_high,carry_better_share"]
    for label, run in (("seed 42", CARRY_DESIGN_LONG), ("seed 43", "hebbian-carry-600-s43")):
        carry = _rows(run, 600, "relatedness")
        for arm in relatedness_eval.LEVELS:
            diffs = _gain_differences(carry, dyn, arm)
            rs = evidence.resamples(len(diffs), N_BOOT, NumpyRng(0))
            interval = stats.bootstrap(lambda idx: 100 * statistics.fmean(diffs[i] for i in idx), len(diffs), rs)
            better = sum(d > 0 for d in diffs) / len(diffs)
            lines.append(f"{label},{arm},{len(diffs)},{interval.point:.4f},{interval.low:.4f},"
                         f"{interval.high:.4f},{better:.4f}")
    return lines


def _gain_differences(carry, dyn, arm: str) -> list[float]:
    def by_case(rows):
        return {(r.doc_idx, r.arm): r.ppl for r in rows}

    c, d = by_case(carry), by_case(dyn)
    docs = sorted(k for k, a in c if a == arm)
    return [
        (math.log(c[(k, "reset")]) - math.log(c[(k, arm)]))
        - (math.log(d[(k, "reset")]) - math.log(d[(k, arm)]))
        for k in docs
    ]


def repeat_summary() -> list[str]:
    lines = ["model,measure,n_docs,delta_ppl,ci_low,ci_high,excludes_zero,win_rate"]
    for label, run in (("ungated carry", "ttt-v1.1"), ("gated carry seed 42", CARRY_DESIGN_LONG),
                       ("gated carry seed 43", "hebbian-carry-600-s43")):
        rows = _rows(run, 600, "repeat")
        estimates = evidence.repeat_trend(rows, rng=NumpyRng(0), n_boot=N_BOOT)
        first = [p for p in evidence.repeat_curve(rows, rng=NumpyRng(0), n_boot=N_BOOT)
                 if p.source == ALL_SOURCES and p.repeat == 1]
        for e in estimates:
            if e.source == ALL_SOURCES:
                i = e.interval
                lines.append(f"{label},{e.measure},{e.n_docs},{i.point:.6g},{i.low:.6g},{i.high:.6g},"
                             f"{int(i.excludes_zero)},{e.win_rate:.4f}")
        for p in first:
            lines.append(f"{label},repeat_1_vs_0,{p.n_docs},{p.gap.point:.6g},{p.gap.low:.6g},{p.gap.high:.6g},"
                         f"{int(p.gap.low > 0 or p.gap.high < 0)},")
    return lines


def relatedness_table(estimates) -> list[str]:
    lines = ["model,measure,n_docs,delta_ppl,ci_low,ci_high,excludes_zero,win_rate"]
    for (label, measure), e in sorted(estimates.items()):
        i = e.interval
        lines.append(f"\"{label}\",{measure},{e.n_docs},{i.point:.6g},{i.low:.6g},{i.high:.6g},"
                     f"{int(i.excludes_zero)},{e.win_rate:.4f}")
    return lines


ABLATION_CASES = (
    ("closed gate only, no carry path (C)", "ablate-c-only"),
    ("separate path, not trained through (A+C)", "ablate-a-c"),
    ("trained through, no separate path (B+C)", "ablate-b-c"),
    ("gated carry (A+B+C)", CARRY_DESIGN),
)


def ablation_estimates() -> dict[tuple[str, str], object]:
    base = _rows(CONTROL, 200, "regimes")
    out = {}
    for label, run in ABLATION_CASES:
        rows = paper_analysis.rebased(_rows(run, 200, "regimes"), base)
        estimates = (
            evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT)
            + evidence.boundary_estimates(_rows(run, 200, "boundary"), rng=NumpyRng(0), n_boot=N_BOOT)
            + paper_analysis.matched_estimates(base, rows, rng=NumpyRng(0), n_boot=N_BOOT)
        )
        for e in estimates:
            if e.source == ALL_SOURCES:
                out[(label, e.measure)] = e
    return out


SEEDS = (("seed 42", CARRY_DESIGN_LONG), ("seed 43", "hebbian-carry-600-s43"))
SEED_MEASURES = ("between_slices", "cross_doc", "within", "vs_matched_lora", "carry_gain", "context_gain",
                 "gain_self", "gain_related", "gain_random", "gain_other_source", "related_vs_random")


def seed_estimates() -> dict[tuple[str, str], object]:
    out = {}
    for step in (400, 600):
        base = _rows(CONTROL, step, "regimes")
        for label, run in SEEDS:
            rows = paper_analysis.rebased(_rows(run, step, "regimes"), base)
            estimates = (
                evidence.regime_estimates(rows, rng=NumpyRng(0), n_boot=N_BOOT)
                + evidence.boundary_estimates(_rows(run, step, "boundary"), rng=NumpyRng(0), n_boot=N_BOOT)
                + paper_analysis.matched_estimates(base, rows, rng=NumpyRng(0), n_boot=N_BOOT)
            )
            if step == 600:
                estimates += context_eval.estimates(_rows(run, step, "context"), rng=NumpyRng(0), n_boot=N_BOOT)
                estimates += relatedness_eval.estimates(_rows(run, step, "relatedness"), rng=NumpyRng(0), n_boot=N_BOOT)
            for e in estimates:
                if e.source == ALL_SOURCES and e.measure in SEED_MEASURES:
                    out[(f"{label}, step {step}", e.measure)] = e
    return out


def carry_design_table(estimates) -> list[str]:
    lines = ["case,measure,n_docs,delta_ppl,ci_low,ci_high,excludes_zero,win_rate"]
    for (label, measure), e in sorted(estimates.items()):
        i = e.interval
        lines.append(f"\"{label}\",{measure},{e.n_docs},{i.point:.6g},{i.low:.6g},{i.high:.6g},"
                     f"{int(i.excludes_zero)},{e.win_rate:.4f}")
    return lines


def main() -> None:
    estimates = source_estimates()
    write_tables(estimates, domain_correlates(estimates))
    tables = OUT / "tables"
    (tables / "T11_carry_design.csv").write_text("\n".join(carry_design_table(carry_design_estimates())) + "\n")
    related = relatedness_estimates()
    (tables / "T12_relatedness.csv").write_text("\n".join(relatedness_table(related)) + "\n")
    (tables / "T13_ablation.csv").write_text("\n".join(carry_design_table(ablation_estimates())) + "\n")
    (tables / "T14_seeds.csv").write_text("\n".join(carry_design_table(seed_estimates())) + "\n")
    (tables / "T15_carry_vs_dyneval.csv").write_text("\n".join(carry_vs_dyneval()) + "\n")
    (tables / "T16_repeat.csv").write_text("\n".join(repeat_summary()) + "\n")
    for name in ("T7_sources_by_step", "T8_domain_correlates", "T11_carry_design", "T12_relatedness",
                 "T13_ablation", "T14_seeds", "T15_carry_vs_dyneval", "T16_repeat"):
        print(tables / f"{name}.csv")


if __name__ == "__main__":
    main()
