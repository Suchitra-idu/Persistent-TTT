"""Two `paper_eval_v1` bundles, paired document by document. Local, no GPU.

    python -m ttt.experiments.compare_checkpoints_v1 \\
        results/lora-only/step_600/<stamp> results/ttt-v1.1/step_600/<stamp>

The first bundle is the baseline (a LoRA-only trained model, read through its
`lora_only` regime), the second the candidate. Positive Δ means the candidate
is lower. `fresh_sanity` should be zero: it is the same base model on the same
documents, so anything else means the holdouts differ.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from ttt.adapters.local_storage import LocalStorage
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import evidence
from ttt.core.types import EvalRow
from ttt.experiments import paper_eval_v1

COMPARE_DIR = "compare"
# Expected to differ between a baseline and a candidate; everything else should not.
EXPECTED_DIFFERENCES = frozenset({"run_name", "layer_start", "layer_stride", "layer_indices"})


def compare(
    baseline: Path,
    candidate: Path,
    *,
    out_dir: Path,
    n_boot: int = paper_eval_v1.DEFAULT_N_BOOT,
    announce: Callable[[str], None] = print,
) -> Path:
    base_rows, base_manifest = _load(baseline)
    cand_rows, cand_manifest = _load(candidate)
    estimates = evidence.checkpoint_estimates(
        base_rows, cand_rows, rng=NumpyRng(paper_eval_v1.BOOT_SEED), n_boot=n_boot
    )
    unmatched = evidence.unmatched_docs(base_rows, cand_rows)
    diff = config_diff(base_manifest.get("describe", {}), cand_manifest.get("describe", {}))

    folder = out_dir / f"{_label(base_manifest, baseline)}__vs__{_label(cand_manifest, candidate)}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    storage = LocalStorage(folder)
    paper_eval_v1.write_estimates(storage, "tables/checkpoints", estimates)
    evidence.save_json(
        storage,
        "manifest.json",
        {
            "baseline": str(baseline),
            "candidate": str(candidate),
            "n_boot": n_boot,
            "boot_seed": paper_eval_v1.BOOT_SEED,
            "unmatched_docs": unmatched,
            "config_diff": diff,
        },
    )
    figures = _figure(folder, estimates, announce)
    storage.write_bytes("SUMMARY.md", _summary(estimates, diff, unmatched, figures).encode())
    if unmatched:
        announce(f"warning: {unmatched} documents are in only one bundle; pairing is partial")
    unexpected = [row for row in diff if row[0] not in EXPECTED_DIFFERENCES]
    if unexpected:
        announce(f"warning: configs differ beyond the arm itself: {[row[0] for row in unexpected]}")
    announce(f"comparison -> {folder}")
    return folder


def config_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    return [(key, a.get(key), b.get(key)) for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]


def _load(folder: Path) -> tuple[tuple[EvalRow, ...], dict[str, Any]]:
    storage = LocalStorage(folder)
    if not storage.exists("raw/regimes.json"):
        raise SystemExit(f"{folder} has no raw/regimes.json; run paper_eval_v1 with the regimes suite")
    rows = evidence.rows_from(EvalRow, evidence.load_json(storage, "raw/regimes.json"))
    manifest = evidence.load_json(storage, "manifest.json") if storage.exists("manifest.json") else {}
    return rows, manifest


def _label(manifest: Mapping[str, Any], folder: Path) -> str:
    bundle = manifest.get("bundle", "")
    run_step = "/".join(bundle.split("/")[:-1]) if bundle else folder.parent.name
    return run_step.replace("/", "_")


def _figure(folder: Path, estimates, announce) -> list[str]:
    try:
        from ttt.experiments import _evidence_figures as figs
    except ImportError:
        announce("skipped figures: matplotlib not importable")
        return []
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    figs.forest(estimates, str(folder / "figures/checkpoints.png"), title="Candidate vs baseline checkpoint")
    return ["figures/checkpoints.png"]


def _summary(estimates, diff, unmatched: int, figures) -> str:
    from ttt.core import report

    lines = ["# Checkpoint comparison", "", "```", report.render(report.estimate_table(estimates)), "```", ""]
    lines += [f"Unmatched documents: {unmatched}", "", "## Config differences", ""]
    lines += [f"- `{key}`: `{a}` → `{b}`" for key, a, b in diff] or ["- none"]
    lines += ["", "## Figures", ""] + [f"- [{f}]({f})" for f in figures]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--n-boot", type=int, default=paper_eval_v1.DEFAULT_N_BOOT)
    parser.add_argument("--out", type=Path, default=Path(paper_eval_v1.RESULTS_DIR) / COMPARE_DIR)
    args = parser.parse_args(argv)
    compare(args.baseline, args.candidate, out_dir=args.out, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
