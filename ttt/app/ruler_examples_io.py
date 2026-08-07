"""RulerExample <-> JSONL, through the Storage port checkpoints already use."""

from __future__ import annotations

import json

from ttt.core.config.ruler import RulerConfig
from ttt.core.ruler_types import RulerExample
from ttt.ports.storage import Storage

PREFIX = "ruler"

# Bump whenever the JSONL row shape below changes, so a reader can never
# mistake bytes written by an older `RulerExample` shape for the current one.
SCHEMA_VERSION = 2


def bucket_key(task: str, length: int) -> str:
    return f"{task}_{length}"


def path(cfg: RulerConfig, task: str, task_version: int, length: int) -> str:
    return f"{PREFIX}/v{SCHEMA_VERSION}/{cfg.key(task, length)}-t{task_version}.jsonl"


def write(
    storage: Storage,
    cfg: RulerConfig,
    task: str,
    task_version: int,
    length: int,
    examples,
) -> None:
    lines = (
        json.dumps(
            {
                "task": e.task,
                "length_bucket": e.length_bucket,
                "prompt": e.prompt,
                "answer_prefix": e.answer_prefix,
                "targets": list(e.targets),
            }
        )
        for e in examples
    )
    storage.write_bytes(
        path(cfg, task, task_version, length), ("\n".join(lines) + "\n").encode()
    )


def read(
    storage: Storage, cfg: RulerConfig, task: str, task_version: int, length: int
) -> tuple[RulerExample, ...]:
    p = path(cfg, task, task_version, length)
    if not storage.exists(p):
        raise FileNotFoundError(
            f"{p} not found; run ruler_prepare_v1 with the same ruler_flags first"
        )
    rows = (json.loads(line) for line in storage.read_bytes(p).decode().splitlines())
    return tuple(
        RulerExample(
            task=row["task"],
            length_bucket=row["length_bucket"],
            prompt=row["prompt"],
            answer_prefix=row["answer_prefix"],
            targets=tuple(row["targets"]),
        )
        for row in rows
        if row
    )
