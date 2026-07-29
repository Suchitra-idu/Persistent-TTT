"""Plot the compounding-dynamics pilot JSON produced by
`train_modal.py::compounding_pilot`.

Usage:
    python plot_pilot.py path/to/compounding_*.json [--out out.png] \\
        [--metric nll|ppl] [--overlay path/to/other.json ...]

Overlay is for comparing sources: pass one JSON per source and each gets
its own facet.
"""

import argparse
import json
from pathlib import Path


def _load(p: Path):
    with p.open() as f:
        return json.load(f)


def _plot_one(ax, data, metric: str):
    for name, rows in data["regimes"].items():
        if rows is None:
            continue
        xs = [r["pos"] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("document position in session")
    ax.set_ylabel(f"{metric.upper()} per token")
    ax.set_title(f"{data['source']}  (n_docs={data['n_docs']})")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path,
                    help="primary pilot JSON")
    ap.add_argument("--overlay", type=Path, nargs="*", default=[],
                    help="additional pilot JSONs (each rendered as a facet)")
    ap.add_argument("--metric", choices=["nll", "ppl"], default="nll")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: alongside the primary JSON)")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    paths = [args.json_path, *args.overlay]
    datas = [_load(p) for p in paths]
    n = len(datas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, data in zip(axes[0], datas):
        _plot_one(ax, data, args.metric)

    fig.tight_layout()
    out = args.out or args.json_path.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
