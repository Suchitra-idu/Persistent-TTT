"""Gallery for the token demo: the paper's own per-token losses on a random sample of cases.

    python -m ttt.experiments.token_demo_gallery_v1 --cases fresh.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttt.app import token_gain_analysis
from ttt.app.relatedness_eval import OTHER_SOURCE, RANDOM, RELATED, RESET, SELF
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR

BUNDLE = RESULTS_DIR / "hebbian-carry-600" / "step_600" / "20261103-164544-tokengain-fresh"
OUT = RESULTS_DIR / "paper" / "demo" / "gallery.json"
ARMS = (SELF, RELATED, RANDOM, OTHER_SOURCE)
EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
N_CASES = 12
PREVIEW_TOKENS = 160


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--n", type=int, default=N_CASES)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    cases = json.loads(args.cases.read_text())["cases"]
    rows = json.loads((BUNDLE / "raw" / "token_gain.json").read_text())
    losses = {(r["arm"], r["doc_idx"]): r["losses"] for r in rows}
    order, gains = token_gain_analysis.gains(rows, ARMS)
    average = {arm: token_gain_analysis.profile(gains[arm], EDGES).round(5).tolist() for arm in ARMS}
    picked = sorted(np.random.default_rng(0).choice(len(order), size=args.n, replace=False).tolist())
    gallery = []
    for i in picked:
        case = cases[order[i]]
        gallery.append({
            "index": case["index"],
            "title": case["target_title"],
            "tokens": [tokenizer.decode([t]) for t in case["target"][1:]],
            "context": tokenizer.decode(case["target"][:1]),
            "losses": {arm: [round(x, 4) for x in losses[(arm, case["index"])]] for arm in (RESET, *ARMS)},
            "primes": {
                arm: {"title": case["prime_titles"][arm],
                      "preview": tokenizer.decode(case["primes"][arm][:PREVIEW_TOKENS])}
                for arm in ARMS
            },
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "checkpoint": "hebbian-carry-600/step_600", "n_cases": len(order), "edges": EDGES,
        "average": average, "cases": gallery,
    }))
    print(f"{OUT}: {len(gallery)} cases, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
