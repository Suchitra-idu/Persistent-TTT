"""Plot the compounding pilot's JSON.

    python -m ttt.experiments.plot_pilot compounding_RedPajamaArXiv_10_0729-1412.json

One facet per file, so passing one JSON per source compares compounding rates
across domains — the question the pilot exists to answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

METRICS = ("nll", "ppl")


def series(payload: Mapping, metric: str) -> dict[str, tuple[list[int], list[float]]]:
    """(positions, values) per regime, ordered by position."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}")
    by_regime: dict[str, list[Mapping]] = {}
    for row in payload["rows"]:
        by_regime.setdefault(row["regime"], []).append(row)
    return {
        regime: (
            [row["position"] for row in sorted(rows, key=_position)],
            [row[metric] for row in sorted(rows, key=_position)],
        )
        for regime, rows in by_regime.items()
    }


def title(payload: Mapping) -> str:
    sources = {row["source"] for row in payload["rows"]}
    label = sources.pop() if len(sources) == 1 else f"{len(sources)} sources"
    positions = {row["position"] for row in payload["rows"]}
    return f"{label}  (n_docs={len(positions)})"


def _position(row: Mapping) -> int:
    return row["position"]


def render(payloads: Sequence[Mapping], metric: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(payloads), figsize=(6 * len(payloads), 4.5), squeeze=False
    )
    for axis, payload in zip(axes[0], payloads, strict=True):
        for regime, (positions, values) in sorted(series(payload, metric).items()):
            axis.plot(positions, values, marker="o", linewidth=1.6, label=regime)
        axis.set_xlabel("document position in session")
        axis.set_ylabel(f"{metric.upper()} per token")
        axis.set_title(title(payload))
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plot_pilot")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--metric", choices=METRICS, default="nll")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payloads = [json.loads(path.read_text()) for path in args.paths]
    out = render(
        payloads, args.metric, args.out or args.paths[0].with_suffix(".png")
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
