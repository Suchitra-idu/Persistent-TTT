"""RULER's offline half: synthesize each task x length's examples once and
store them as JSONL. CPU + tokenizer only — never touches a GPU, and never
regenerates a set another run already scored against."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ttt.app import ruler_examples_io as io
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.ruler import RulerConfig
from ttt.core.ruler_types import RulerContext, RulerExample
from ttt.extensions.ruler_tasks._registry import RulerTask
from ttt.ports.data_source import DataSource
from ttt.ports.rng import Rng
from ttt.ports.storage import Storage
from ttt.ports.tokenizer import Tokenizer


def prepare(
    *,
    cfg: RulerConfig,
    tasks: Mapping[str, RulerTask],
    tokenizer: Tokenizer,
    ctx: RulerContext,
    rng: Rng,
    storage: Storage,
    announce: Callable[[str], None] = print,
) -> dict[str, tuple[RulerExample, ...]]:
    built: dict[str, tuple[RulerExample, ...]] = {}
    for task_name in cfg.tasks:
        task = tasks[task_name]
        for length in cfg.context_lengths:
            key = io.bucket_key(task_name, length)
            if storage.exists(io.path(cfg, task_name, task.version, length)):
                announce(f"  {key}: already prepared")
                built[key] = io.read(storage, cfg, task_name, task.version, length)
                continue
            examples = tuple(
                task.build(rng, tokenizer, length, ctx) for _ in range(cfg.num_samples)
            )
            io.write(storage, cfg, task_name, task.version, length, examples)
            announce(f"  {key}: wrote {len(examples)} examples")
            built[key] = examples
    storage.commit()
    return built


def build_context(
    *,
    source: DataSource,
    spec: DatasetSpec,
    min_words: int = 20_000,
    qa_source: Any = None,
    num_qa_pairs: int = 200,
) -> RulerContext:
    """Real material the generators need: haystack prose from `source`/`spec`
    (this repo's own corpus), QA pairs from `qa_source` if the `qa` task is in
    play."""
    table = source.load(spec)
    words: list[str] = []
    for index in range(len(table)):
        words.extend(table.row(index)[spec.text_column].split())
        if len(words) >= min_words:
            break
    qa_pairs = tuple(qa_source.load(num_qa_pairs)) if qa_source is not None else ()
    return RulerContext(haystack_words=tuple(words), qa_pairs=qa_pairs)
