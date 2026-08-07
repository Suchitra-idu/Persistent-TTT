"""The RulerTask protocol and the RULER_TASKS registry, same shape as
`extensions/datasets/_registry.py`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ttt.core.ruler_types import RulerContext, RulerExample


@runtime_checkable
class RulerTask(Protocol):
    name: str
    # Bump whenever this task's generator logic changes prompt content for
    # the same (task, length, num_samples, seed) — the cache key a schema
    # version alone can't catch, since the row *shape* hasn't changed.
    version: int

    def build(
        self, rng, tokenizer, seq_len: int, ctx: RulerContext
    ) -> RulerExample:
        """One example at roughly `seq_len` prompt tokens."""

    def score(self, prediction: str, example: RulerExample) -> float:
        """In [0, 1]. RULER's own metric for this task family."""


RULER_TASKS: dict[str, RulerTask] = {}


def register(task: RulerTask) -> RulerTask:
    if task.name in RULER_TASKS:
        raise ValueError(
            f"ruler task {task.name!r} is already registered; a name collision "
            "would make the resolved config ambiguous"
        )
    RULER_TASKS[task.name] = task
    return task


def get(name: str) -> RulerTask:
    if name not in RULER_TASKS:
        raise KeyError(f"unknown ruler task {name!r}. Known: {sorted(RULER_TASKS)}")
    return RULER_TASKS[name]
