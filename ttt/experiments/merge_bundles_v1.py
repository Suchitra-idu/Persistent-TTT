"""Shard bundles of one checkpoint merged into one bundle. Local, no GPU.

    python -m ttt.experiments.merge_bundles_v1 <out folder> <shard bundle> [<shard bundle> ...]

Raw rows are concatenated file by file, the shards are moved under
<out>/shards/, and the merged bundle is rendered like any other.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from ttt.app import dynamic_eval
from ttt.experiments import paper_eval_v1

SHARDS_DIR = "shards"


def merge(out: Path, shards: Sequence[Path]) -> Path:
    rows: dict[str, list] = {}
    for shard in shards:
        for raw in sorted((shard / "raw").glob("*.json")):
            rows.setdefault(raw.stem, []).extend(json.loads(raw.read_text()))
    (out / SHARDS_DIR).mkdir(parents=True, exist_ok=True)
    manifests = []
    for shard in shards:
        manifests.append(json.loads((shard / "manifest.json").read_text()))
        shutil.move(str(shard), str(out / SHARDS_DIR / shard.name))
    manifest = {**manifests[0], "bundle": str(out), "shards": [m["bundle"] for m in manifests]}
    paper_eval_v1.write_bundle(out, {"rows": rows}, manifest)
    return out


def chosen(rows_path: Path) -> dict[str, str]:
    """Per-objective best setting from merged tuning rows, as `--dyneval-fixed` text."""
    from ttt.app.evidence import rows_from

    tune_rows = rows_from(dynamic_eval.TuneRow, json.loads(rows_path.read_text()))
    return {o: dynamic_eval.best(tune_rows, o).text for o in dynamic_eval.OBJECTIVES}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m ttt.experiments.merge_bundles_v1 <out> <shard> [<shard> ...]")
    merge(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
