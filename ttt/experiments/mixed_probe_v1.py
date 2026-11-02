"""The mixed-session probe's contrasts from two `paper_eval_v1` bundles. Local, no GPU.

    python -m ttt.experiments.mixed_probe_v1 \\
        results/hebbian-carry-600/step_600/<stamp> results/mixed-carry-600/step_600/<stamp>

First bundle the baseline (carry s42), second the candidate (mixed s42). Each
contrast is the candidate's gain over its own reset minus the baseline's, on
the same cases (plan/PREREGISTRATION.md §7).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ttt.adapters.local_storage import LocalStorage
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import evidence, relatedness_eval
from ttt.core import report
from ttt.core.stats import Estimate
from ttt.core.types import COLD_CARRY, COLD_CARRY_OFF, EvalRow
from ttt.experiments import paper_eval_v1

PREREGISTERED_N_BOOT = 10_000
CHAIN_ARM = f"{relatedness_eval.CHAIN}10"
ARMS = (relatedness_eval.OTHER_SOURCE, CHAIN_ARM, relatedness_eval.RELATED)
BETWEEN_SLICES = "between_slices_gain_change"
IGNORED_WITHIN = 0.01


def contrasts(baseline: Path, candidate: Path, *, n_boot: int) -> tuple[Estimate, ...]:
    rng = NumpyRng(paper_eval_v1.BOOT_SEED)
    base_rel, cand_rel = (_rows(f, "relatedness", relatedness_eval.RelatedRow) for f in (baseline, candidate))
    base_reg, cand_reg = (_rows(f, "regimes", EvalRow) for f in (baseline, candidate))

    def regime(rows, name):
        return evidence.per_doc(rows, where=lambda r: r.regime == name)

    return relatedness_eval.gain_changes(cand_rel, base_rel, arms=ARMS, rng=rng, n_boot=n_boot) + (
        evidence.gain_change(
            BETWEEN_SLICES,
            (regime(cand_reg, COLD_CARRY_OFF), regime(cand_reg, COLD_CARRY)),
            (regime(base_reg, COLD_CARRY_OFF), regime(base_reg, COLD_CARRY)),
            rng=rng, n_boot=n_boot,
        ),
    )


def own_gains(candidate: Path, *, n_boot: int) -> tuple[Estimate, Estimate]:
    """The candidate's related and between-slice gains, pooled, for the "ignores the carry" rule."""
    rng = NumpyRng(paper_eval_v1.BOOT_SEED)
    rel = _rows(candidate, "relatedness", relatedness_eval.RelatedRow)
    reg = _rows(candidate, "regimes", EvalRow)

    def pooled(measure, rows, field, ref, treated):
        return evidence.paired_estimates(
            measure,
            evidence.per_doc(rows, where=lambda r: getattr(r, field) == ref),
            evidence.per_doc(rows, where=lambda r: getattr(r, field) == treated),
            rng=rng, n_boot=n_boot,
        )[-1]

    return (
        pooled("related_gain", rel, "arm", relatedness_eval.RESET, relatedness_eval.RELATED),
        pooled("between_slices_gain", reg, "regime", COLD_CARRY_OFF, COLD_CARRY),
    )


def verdict(changes: tuple[Estimate, ...], gains: tuple[Estimate, Estimate]) -> str:
    by = {e.measure: e.interval for e in changes}
    other, chain, related = (by[f"{arm}_gain_change"] for arm in ARMS)
    if all(abs(g.interval.point) <= IGNORED_WITHIN for g in gains):
        return "the probe learned to ignore the carry"
    removes = other.low > 0 and chain.low > 0 and related.high >= 0
    return "training across documents removes the harm" if removes else "the harm is not shown to be removed"


def probe(baseline: Path, candidate: Path, *, n_boot: int, announce: Callable[[str], None] = print) -> Path:
    changes = contrasts(baseline, candidate, n_boot=n_boot)
    gains = own_gains(candidate, n_boot=n_boot)
    storage = LocalStorage(candidate)
    paper_eval_v1.write_estimates(storage, "tables/mixed_probe", changes + gains)
    rule = verdict(changes, gains)
    storage.write_bytes(
        "tables/mixed_probe_verdict.md",
        f"Baseline: `{baseline}`\n\nResamples: {n_boot}\n\nPreregistered rule (§7): **{rule}** "
        "(in a single-seed probe)\n".encode(),
    )
    announce(report.render(report.estimate_table(changes + gains)))
    announce(f"verdict: {rule}")
    return candidate / "tables"


def _rows(folder: Path, name: str, row_type) -> tuple:
    storage = LocalStorage(folder)
    path = f"raw/{name}.json"
    if not storage.exists(path):
        raise SystemExit(f"{folder} has no {path}")
    return evidence.rows_from(row_type, evidence.load_json(storage, path))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--n-boot", type=int, default=PREREGISTERED_N_BOOT)
    args = parser.parse_args(argv)
    probe(args.baseline, args.candidate, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
