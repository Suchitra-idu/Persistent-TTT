"""Language-transfer eval's offline half, in two steps: `prepare` pulls each
language once into its own file (CPU + network only, resumable per
language); `combine` merges and shuffles them into the one file a
DatasetSpec actually reads. Two steps, not one write, because a DatasetSpec
concatenates shard files in filename order — leaving the per-language files
as DatasetSpec.source would make every language a contiguous block, and
`holdout_boundary`'s "the last N rows are a fair eval sample" assumes rows
are already mixed, not grouped by source."""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Mapping, Sequence

from ttt.core.config import lang_transfer
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage


def prepare(
    *,
    languages: Sequence[tuple[str, str]],
    snapshot: str,
    wiki_source: Any,
    storage: Storage,
    role: str,
    target_rows: int,
    min_chars: int = 200,
    announce: Callable[[str], None] = print,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code, name in languages:
        path = _raw_path(role, code)
        if storage.exists(path):
            announce(f"  {code} ({name}): already prepared")
            continue
        rows = wiki_source.load(code, snapshot, target_rows, min_chars=min_chars)
        if not rows:
            raise RuntimeError(f"no rows fetched for language {code!r} ({name})")
        storage.write_bytes(path, _to_parquet_bytes(rows, lang=code))
        counts[code] = len(rows)
        announce(f"  {code} ({name}): wrote {len(rows)} rows")
    storage.commit()
    return counts


def combine(
    *,
    languages: Sequence[tuple[str, str]],
    storage: Storage,
    role: str,
    rng: Rng,
    announce: Callable[[str], None] = print,
) -> int:
    """Always rebuilds: unlike `prepare()`'s per-language fetch, this is CPU
    + local reshuffling, cheap enough that a `languages` short-circuiting on
    a fixed output path would risk silently serving a stale mix once the
    list changes underneath it (docs/experiments-map.md)."""
    out_path = _combined_path(role)

    rows: list[dict[str, Any]] = []
    for code, name in languages:
        path = _raw_path(role, code)
        if not storage.exists(path):
            raise RuntimeError(
                f"language {code!r} ({name}) has not been prepared yet; run "
                "prepare() before combine()"
            )
        rows.extend(_read_parquet_rows(storage.read_bytes(path)))

    rng.shuffle(rows)
    storage.write_bytes(out_path, _rows_to_parquet_bytes(rows))
    storage.commit()
    announce(f"  combined ({role}): {len(rows)} rows from {len(languages)} languages")
    return len(rows)


def _raw_path(role: str, code: str) -> str:
    return f"{lang_transfer.corpus_dir(role)}/{code}.parquet"


def _combined_path(role: str) -> str:
    return f"{lang_transfer.combined_dir(role)}/all.parquet"


def _to_parquet_bytes(rows: Sequence[Mapping[str, Any]], *, lang: str) -> bytes:
    # A JSON string, not a struct: DatasetSpec.source_of parses either, and a
    # string column round-trips through every writer the same way.
    stamped = [{**row, "meta": json.dumps({"lang": lang})} for row in rows]
    return _rows_to_parquet_bytes(stamped)


def _rows_to_parquet_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "text": [row["text"] for row in rows],
            "title": [row.get("title", "") for row in rows],
            "meta": [row["meta"] for row in rows],
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _read_parquet_rows(data: bytes) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()
