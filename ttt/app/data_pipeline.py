"""The corpus, from a DataSource to the token ranges a session works on.

PIPELINE is the whole of what `load_token_dataset` used to do inline. The
holdout half lives here too, since both halves must agree on the split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ttt.app import stages
from ttt.app.stages import Data
from ttt.core import sampling, tokens
from ttt.core.config import presets
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.train import TrainConfig
from ttt.ports.data_source import DataSource
from ttt.ports.rng import Rng
from ttt.ports.table import SOURCE_COLUMN, Table
from ttt.ports.tokenizer import Tokenizer

PIPELINE = (
    stages.split_holdout,
    stages.filter_sources,
    stages.shuffle,
    stages.prefilter,
    stages.balance,
    stages.tokenize,
    stages.drop_short,
    stages.rebalance,
    stages.cap,
)


@dataclass(frozen=True)
class Doc:
    """A tokenized document. `source` is never empty (D9)."""

    index: int
    source: str
    token_ids: tuple[int, ...]

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class Holdout:
    docs: tuple[Doc, ...]
    shortfalls: tuple[sampling.SourceShortfall, ...]


def resolve_weights(
    cfg: TrainConfig, spec: DatasetSpec
) -> Mapping[str, int] | None:
    """None means no balancing: an explicit "none" or a spec with no default."""
    name = presets.resolve_preset_name(cfg.source_preset, spec.default_source_preset)
    return presets.get_preset(name) if name else None


def load(
    *,
    source: DataSource,
    spec: DatasetSpec,
    cfg: TrainConfig,
    rng: Rng,
    tokenizer: Tokenizer,
    limit_docs: int | None = None,
    weights: Mapping[str, int] | None = None,
    pipeline: Sequence = PIPELINE,
    announce: Callable[[str], None] = lambda _: None,
) -> Data:
    """`weights=None` resolves from cfg and spec; pass a mapping to override,
    which is how --only-sources reaches the balancer as plain data (D5)."""
    data = Data(
        table=source.load(spec),
        spec=spec,
        cfg=cfg,
        rng=rng,
        tokenizer=tokenizer,
        limit_docs=limit_docs,
        weights=weights if weights is not None else resolve_weights(cfg, spec),
        announce=announce,
    )
    return run(data, pipeline, announce=announce)


def run(
    data: Data,
    pipeline: Sequence = PIPELINE,
    *,
    announce: Callable[[str], None] = lambda _: None,
) -> Data:
    """Each stage timed and announced as it runs — the pipeline's own
    StageLog is only assembled afterward, too late to explain a slow stage
    while it's still running."""
    for stage in pipeline:
        before = len(data.table)
        t0 = time.time()
        data = stage(data)
        announce(
            f"  {stage.__name__}: {before:,d} -> {len(data.table):,d} rows "
            f"in {time.time() - t0:.1f}s"
        )
    return data


def documents(data: Data) -> tuple[Doc, ...]:
    rows = stages.rows_of(data.table, (SOURCE_COLUMN, stages.TOKENS_COLUMN))
    return tuple(
        Doc(index=index, source=source, token_ids=tuple(ids))
        for index, (source, ids) in enumerate(rows)
    )


def holdout(
    *,
    source: DataSource,
    spec: DatasetSpec,
    cfg: TrainConfig,
    rng: Rng,
    tokenizer: Tokenizer,
) -> Holdout:
    """The eval pool: the reserved tail, long enough to slice, sampled per source."""
    table = source.load(spec)
    held = table.select(range(spec.holdout_boundary(len(table)), len(table)))
    held = held.filter(SOURCE_COLUMN, spec.keeps)
    pool = _long_enough(held, spec, cfg.eval_min_tokens)

    labels = pool.column(SOURCE_COLUMN)
    picked, shortfalls = _pick(labels, cfg, rng)
    return Holdout(
        docs=_encode(pool, spec, cfg, tokenizer, picked),
        shortfalls=tuple(shortfalls),
    )


def _long_enough(table: Table, spec: DatasetSpec, min_tokens: int) -> Table:
    """Falls back to the unfiltered pool rather than evaluating on nothing."""
    minimum = tokens.min_chars_for(min_tokens, tokens.HOLDOUT_CHARS_PER_TOKEN)
    filtered = table.filter(spec.text_column, lambda text: len(text) >= minimum)
    return filtered if len(filtered) else table


def _pick(
    labels: Sequence[str], cfg: TrainConfig, rng: Rng
) -> tuple[list[int], list[sampling.SourceShortfall]]:
    if cfg.eval_n_docs_per_source > 0:
        return sampling.n_per_source_indices(labels, cfg.eval_n_docs_per_source, rng)
    return sampling.stratified_indices(labels, cfg.eval_n_docs, rng), []


def _encode(
    pool: Table,
    spec: DatasetSpec,
    cfg: TrainConfig,
    tokenizer: Tokenizer,
    picked: Sequence[int],
) -> tuple[Doc, ...]:
    rows = [pool.row(index) for index in picked]
    encoded = tokenizer.encode_batch(
        [row[spec.text_column] for row in rows], max_length=cfg.max_seq_len
    )
    return tuple(
        Doc(index=index, source=row[SOURCE_COLUMN], token_ids=tuple(ids))
        for index, (row, ids) in zip(picked, zip(rows, encoded, strict=True))
    )
