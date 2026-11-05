"""Addendum A on a token_gain bundle: writes `tables/addendum_a.json` into it.

    python -m ttt.experiments.token_gain_analysis_v1 --cases fresh.json results/<run>/<step>/<bundle> ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ttt.app import token_gain_analysis
from ttt.experiments.paper_analysis_v1 import RESULTS_DIR

TITLES = RESULTS_DIR / "paper" / "inputs" / "fresh_case_titles.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("bundles", type=Path, nargs="+")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    if any(c["index"] != i for i, c in enumerate(cases)):
        raise ValueError("cases must be saved in index order")
    TITLES.parent.mkdir(parents=True, exist_ok=True)
    TITLES.write_text(json.dumps([c["prime_titles"] for c in cases]))
    for bundle in args.bundles:
        rows = json.loads((bundle / "raw" / "token_gain.json").read_text())
        result = token_gain_analysis.analyse(rows, cases)
        (bundle / "tables").mkdir(exist_ok=True)
        path = bundle / "tables" / "addendum_a.json"
        path.write_text(json.dumps(dataclasses.asdict(result), indent=1))
        print(f"{path}: A1 {result.a1}, A2 {result.a2}")


if __name__ == "__main__":
    main()
