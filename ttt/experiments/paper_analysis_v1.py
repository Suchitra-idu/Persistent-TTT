"""The paper's cross-checkpoint tables and figures from paper_eval bundles. Local, no GPU.

    python -m ttt.experiments.paper_analysis_v1 --control lora-only-matched \\
        --run hebbian=ttt-v1.1 --run delta_chunk=ttt-v1.1-clean-deltachunk \\
        --steps 200,400,600 --w0-norms results/paper/inputs/w0_norms.json

Each (run, step) reads its newest bundle under results/<run>/step_<n>/. Rewrites
tables/, figures/, manifest.json and SUMMARY.md under --out and nothing else.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ttt.adapters.local_storage import LocalStorage
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import (
    boundary_eval,
    context_eval,
    evidence,
    paper_analysis,
    relatedness_eval,
    repeat_eval,
    swap_eval,
)
from ttt.app.paper_analysis import VS_MATCHED, Saturation, StatePoint
from ttt.core import report
from ttt.core.metrics import ALL_SOURCES
from ttt.core.stats import Estimate
from ttt.core.types import COLD_CARRY, EvalRow, SliceRow
from ttt.experiments import paper_eval_v1

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "paper"
# evidence calls base -> LoRA path "lora"; on a TTT checkpoint that path also
# carries the trained W_down, so the paper names it for what it measures.
TRAINING = "training"
MAIN_MEASURES = (TRAINING, "within", "between_slices", "total", VS_MATCHED, "cross_doc", "cross_slice")
PER_SOURCE_MEASURES = (VS_MATCHED, "within", "cross_doc")
READABLE = {"within": "within document", VS_MATCHED: "vs. matched LoRA", "cross_doc": "cross-document"}
ROW_TYPES = {
    "regimes": EvalRow,
    "regime_slices": SliceRow,
    "boundary": boundary_eval.ArmRow,
    "repeat": repeat_eval.RepeatItem,
    "swap": swap_eval.SwapRow,
    "context": context_eval.ContextRow,
    "relatedness": relatedness_eval.RelatedRow,
}


@dataclass(frozen=True)
class Bundle:
    run: str
    step: int
    path: Path

    def rows(self, name: str) -> tuple:
        raw = self.path / "raw" / f"{name}.json"
        if not raw.is_file():
            return ()
        return evidence.rows_from(ROW_TYPES[name], json.loads(raw.read_text()))

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.path / "manifest.json").read_text())


@dataclass
class Paper:
    main: list[tuple[str, int, Estimate]] = field(default_factory=list)
    per_source: list[tuple[str, Estimate]] = field(default_factory=list)
    persistence: list[tuple[str, Estimate]] = field(default_factory=list)
    positions: dict[str, tuple[tuple[int, Estimate], ...]] = field(default_factory=dict)
    repeat: dict[str, tuple[evidence.RepeatPoint, ...]] = field(default_factory=dict)
    states: list[tuple[str, StatePoint]] = field(default_factory=list)
    saturation: list[tuple[str, int, Saturation]] = field(default_factory=list)
    bundles: list[Bundle] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    clip_threshold: float = float("nan")


def newest_bundle(results: Path, run: str, step: int) -> Bundle:
    folder = results / run / f"step_{step}"
    stamps = sorted(folder.glob("*/manifest.json"))
    if not stamps:
        raise FileNotFoundError(f"no paper_eval bundle under {folder}")
    return Bundle(run=run, step=step, path=stamps[-1].parent)


def analyse(
    *,
    control: str,
    runs: Mapping[str, str],
    steps: Sequence[int],
    mechanism_step: int,
    w0_norm: float,
    results: Path = RESULTS_DIR,
    n_boot: int = paper_eval_v1.DEFAULT_N_BOOT,
) -> Paper:
    def rng():
        return NumpyRng(paper_eval_v1.BOOT_SEED)

    paper = Paper(labels=list(runs))
    for step in steps:
        base = newest_bundle(results, control, step)
        paper.bundles.append(base)
        base_rows = base.rows("regimes")
        paper.main += [(control, step, e) for e in _regimes(base_rows, rng(), n_boot)]
        for label, run in runs.items():
            bundle = newest_bundle(results, run, step)
            paper.bundles.append(bundle)
            rows = paper_analysis.rebased(bundle.rows("regimes"), base_rows)
            estimates = (
                _regimes(rows, rng(), n_boot)
                + paper_analysis.matched_estimates(base_rows, rows, rng=rng(), n_boot=n_boot)
                + evidence.boundary_estimates(bundle.rows("boundary"), rng=rng(), n_boot=n_boot)
            )
            paper.main += [(label, step, e) for e in estimates]
            clip = _clip_tau(bundle)
            if clip is not None:
                paper.clip_threshold = clip / w0_norm
                paper.saturation.append(
                    (label, step, paper_analysis.clip_saturation(
                        _state_ratios(bundle), clip_tau=clip, w0_norm=w0_norm
                    ))
                )
            if step == mechanism_step:
                _mechanism(paper, label, bundle, estimates, rng, n_boot)
    return paper


def _regimes(rows: Sequence[EvalRow], rng, n_boot: int) -> tuple[Estimate, ...]:
    return tuple(
        dataclasses.replace(e, measure=TRAINING) if e.measure == "lora" else e
        for e in evidence.regime_estimates(rows, rng=rng, n_boot=n_boot)
    )


def _mechanism(paper: Paper, label: str, bundle: Bundle, estimates, rng, n_boot: int) -> None:
    paper.per_source += [(label, e) for e in estimates if e.measure in PER_SOURCE_MEASURES]
    repeat_rows = bundle.rows("repeat")
    persistence = (
        tuple(e for e in estimates if e.measure in (evidence.CROSS_DOC, evidence.CROSS_SLICE))
        + evidence.repeat_trend(repeat_rows, rng=rng(), n_boot=n_boot)
        + tuple(
            e for e in evidence.swap_estimates(bundle.rows("swap"), rng=rng(), n_boot=n_boot)
            if e.measure == evidence.SAME_VS_OTHER
        )
    )
    paper.persistence += [(label, e) for e in persistence if e.source == ALL_SOURCES]
    paper.positions[label] = paper_analysis.position_estimates(
        bundle.rows("regime_slices"), rng=rng(), n_boot=n_boot
    )
    paper.repeat[label] = tuple(
        p for p in evidence.repeat_curve(repeat_rows, rng=rng(), n_boot=n_boot)
        if p.source == ALL_SOURCES
    )
    paper.states += [
        (label, p)
        for p in paper_analysis.state_points(
            bundle.rows("boundary"), group=lambda r: r.arm, position=lambda r: r.chain_position
        )
    ]


def _clip_tau(bundle: Bundle) -> float | None:
    describe = bundle.manifest().get("describe", {})
    if describe.get("clip_enabled") != "True":
        return None
    return float(describe["clip_tau"])


def _state_ratios(bundle: Bundle) -> list[float]:
    ratios = [r.state_ratio for name in ("boundary", "repeat", "swap") for r in bundle.rows(name)]
    return ratios + [r.state_ratio_final for r in bundle.rows("regimes") if r.regime == COLD_CARRY]


def write(paper: Paper, out: Path, *, inputs: Mapping[str, Any], announce: Callable[[str], None] = print) -> Path:
    storage = LocalStorage(out)
    _estimate_table(storage, "tables/T1_main_gaps", ("run", "step", "measure"), [
        ((label, str(step), e.measure), e)
        for measure in MAIN_MEASURES
        for label, step, e in paper.main
        if e.source == ALL_SOURCES and e.measure == measure
    ])
    _estimate_table(storage, "tables/T2_per_source", ("run", "measure", "source"), [
        ((label, e.measure, e.source), e)
        for measure in PER_SOURCE_MEASURES
        for label, e in paper.per_source
        if e.measure == measure
    ])
    _estimate_table(storage, "tables/T3_persistence", ("run", "measure"), [
        ((label, e.measure), e) for label, e in paper.persistence
    ])
    _estimate_table(storage, "tables/T4_by_position", ("run", "measure", "slice"), [
        ((label, e.measure, str(index)), e)
        for label, points in paper.positions.items()
        for index, e in points
    ])
    _state_table(storage, "tables/T5_state_and_gate", paper.states)
    _saturation_table(storage, "tables/T6_clip_saturation", paper.saturation)
    figures = _figures(paper, out, announce)
    evidence.save_json(storage, "manifest.json", {
        **inputs,
        "bundles": [
            {"run": b.run, "step": b.step, "path": str(b.path), "git": b.manifest().get("git")}
            for b in paper.bundles
        ],
        "clip_threshold": paper.clip_threshold,
        "n_boot": inputs.get("n_boot"),
        "boot_seed": paper_eval_v1.BOOT_SEED,
        "git": paper_eval_v1.git_state(),
    })
    storage.write_bytes("SUMMARY.md", _summary(paper, figures).encode())
    announce(f"paper analysis -> {out}")
    return out


ESTIMATE_HEADERS = ("n", "Δppl", "95% CI", "", "win")
ESTIMATE_CSV = ("n_docs", "delta_ppl", "ci_low", "ci_high", "excludes_zero", "win_rate")


def _estimate_table(storage, stem: str, keys: tuple[str, ...], rows) -> None:
    md_rows = tuple(
        (*cells, str(e.n_docs), report.delta(e.interval.point),
         f"[{report.delta(e.interval.low)}, {report.delta(e.interval.high)}]",
         "*" if e.interval.excludes_zero else "", f"{e.win_rate:.2f}")
        for cells, e in rows
    )
    aligns = (report.LEFT,) * len(keys) + (report.RIGHT, report.RIGHT, report.RIGHT, report.LEFT, report.RIGHT)
    table = report.Table(headers=keys + ESTIMATE_HEADERS, rows=md_rows, aligns=aligns)
    storage.write_bytes(f"{stem}.md", (report.render(table) + "\n").encode())
    csv = [",".join(keys + ESTIMATE_CSV)] + [
        ",".join((*cells, str(e.n_docs), f"{e.interval.point:.6g}", f"{e.interval.low:.6g}",
                  f"{e.interval.high:.6g}", str(int(e.interval.excludes_zero)), f"{e.win_rate:.4f}"))
        for cells, e in rows
    ]
    storage.write_bytes(f"{stem}.csv", ("\n".join(csv) + "\n").encode())


def _state_table(storage, stem: str, states: Sequence[tuple[str, StatePoint]]) -> None:
    headers = ("run", "arm", "chain_position", "n", "state_ratio_median", "state_ratio_q25",
               "state_ratio_q75", "gate_median")
    rows = [
        (label, p.group, str(p.position), str(p.n_rows), f"{p.state_ratio.median:.1f}",
         f"{p.state_ratio.low:.1f}", f"{p.state_ratio.high:.1f}", f"{p.gate_mean.median:.3f}")
        for label, p in states
    ]
    _plain_table(storage, stem, headers, rows)


def _saturation_table(storage, stem: str, saturation) -> None:
    headers = ("run", "step", "rows", "clip_threshold", "fraction_clipped", "min_times_threshold")
    rows = [
        (label, str(step), str(s.n_rows), f"{s.threshold:.3f}", f"{s.fraction_clipped:.3f}",
         f"{s.min_factor:.0f}")
        for label, step, s in saturation
    ]
    _plain_table(storage, stem, headers, rows)


def _plain_table(storage, stem: str, headers: tuple[str, ...], rows) -> None:
    aligns = (report.LEFT,) * 2 + (report.RIGHT,) * (len(headers) - 2)
    table = report.Table(headers=headers, rows=tuple(rows), aligns=aligns)
    storage.write_bytes(f"{stem}.md", (report.render(table) + "\n").encode())
    storage.write_bytes(f"{stem}.csv", ("\n".join(",".join(r) for r in (headers, *rows)) + "\n").encode())


def _figures(paper: Paper, out: Path, announce) -> list[str]:
    try:
        from ttt.experiments import _paper_figures as figs
    except ImportError:
        announce("skipped figures: matplotlib not importable")
        return []
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    series: dict[str, dict[int, dict[str, Estimate]]] = {}
    for label, step, e in paper.main:
        if e.source == ALL_SOURCES and label in paper.labels:
            series.setdefault(label, {}).setdefault(step, {})[e.measure] = e
    if series:
        made.append(figs.gains_vs_step(series, (
            (VS_MATCHED, "TTT run vs. matched LoRA-only"),
            ("within", "Fast-weight updates within a document"),
            ("cross_doc", "Carrying state across documents"),
        ), str(folder / "F1_gains_vs_step.pdf")))
    if paper.positions:
        made.append(figs.by_position(paper.positions, (
            ("within", "Within a slice"), ("between_slices", "Carried from earlier slices"),
        ), str(folder / "F2_by_position.pdf")))
    if paper.repeat:
        made.append(figs.repeat_mechanism(paper.repeat, paper.clip_threshold,
                                          str(folder / "F3_repeat_mechanism.pdf")))
    if paper.per_source:
        from ttt.experiments import _evidence_figures

        for label in paper.labels:
            made.append(_evidence_figures.forest(
                [dataclasses.replace(e, measure=READABLE.get(e.measure, e.measure))
                 for run, e in paper.per_source if run == label],
                str(folder / f"F4_per_source_{label}.pdf"),
                title=f"Per-source gains, {label}, step {max(p.step for p in paper.bundles)}",
                xlabel="Δppl  (+ = lower perplexity)",
            ))
    return [str(Path(p).relative_to(out)) for p in made]


def _summary(paper: Paper, figures: Sequence[str]) -> str:
    lines = ["# Paper analysis", "", "Regenerated by `ttt.experiments.paper_analysis_v1`; "
             "bundles and git state are in manifest.json.", "", "## Tables", ""]
    lines += [f"- [{name}](tables/{name}.md)" for name in (
        "T1_main_gaps", "T2_per_source", "T3_persistence", "T4_by_position",
        "T5_state_and_gate", "T6_clip_saturation",
    )]
    lines += ["", "## Figures", ""] + [f"- [{f}]({f})" for f in figures]
    if paper.saturation:
        lines += ["", "## Clip saturation", ""] + [
            f"- {label} step {step}: {100 * s.fraction_clipped:.1f}% of {s.n_rows} rows above "
            f"the clip threshold {s.threshold:.3f}, the smallest by {s.min_factor:.0f}x"
            for label, step, s in paper.saturation
        ]
    return "\n".join(lines) + "\n"


def _labelled_runs(pairs: Sequence[str]) -> dict[str, str]:
    runs = {}
    for pair in pairs:
        label, sep, run = pair.partition("=")
        if not sep or not label or not run:
            raise SystemExit(f"--run takes label=run, got {pair!r}")
        runs[label] = run
    return runs


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(prog="paper_analysis_v1")
    parser.add_argument("--control", required=True)
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--steps", required=True)
    parser.add_argument("--mechanism-step", type=int, default=0)
    parser.add_argument("--w0-norms", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=paper_eval_v1.DEFAULT_N_BOOT)
    args = parser.parse_args(argv)

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    runs = _labelled_runs(args.runs)
    norms = json.loads(args.w0_norms.read_text())["norms"]
    w0_norm = sum(norms.values()) / len(norms)
    paper = analyse(
        control=args.control, runs=runs, steps=steps,
        mechanism_step=args.mechanism_step or max(steps),
        w0_norm=w0_norm, results=args.results, n_boot=args.n_boot,
    )
    inputs = {
        "control": args.control, "runs": runs, "steps": steps,
        "mechanism_step": args.mechanism_step or max(steps),
        "w0_norms": str(args.w0_norms), "w0_norm_mean": w0_norm, "n_boot": args.n_boot,
    }
    return write(paper, args.out, inputs=inputs)


if __name__ == "__main__":
    main()
