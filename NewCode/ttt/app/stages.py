"""The data pipeline's stages. Each is Data -> Data, and each is tested alone.

Every stage records what it kept, so the pipeline explains itself without
re-reading a corpus that no longer exists in the shape it was read in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ttt.core import balance as balance_math
from ttt.core import tokens
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.train import TrainConfig
from ttt.ports.rng import Rng
from ttt.ports.table import SOURCE_COLUMN, Table
from ttt.ports.tokenizer import Tokenizer

TOKENS_COLUMN = "input_ids"

# The pre-tokenize pass over-fetches because drop_short is source-biased: Books
# survive it, C4 does not, so a mix balanced before tokenizing is not one after.
PRESET_OVERSAMPLE = 3


@dataclass(frozen=True)
class StageLog:
    stage: str
    before: int
    after: int
    note: str = ""

    @property
    def dropped(self) -> int:
        return self.before - self.after


@dataclass(frozen=True)
class Data:
    table: Table
    spec: DatasetSpec
    cfg: TrainConfig
    rng: Rng
    tokenizer: Tokenizer
    limit_docs: int | None = None
    weights: Mapping[str, int] | None = None
    log: tuple[StageLog, ...] = ()

    @property
    def sources(self) -> list[str]:
        return self.table.column(SOURCE_COLUMN)

    def keeping(self, stage: str, table: Table, note: str = "") -> "Data":
        entry = StageLog(
            stage=stage, before=len(self.table), after=len(table), note=note
        )
        return replace(self, table=table, log=self.log + (entry,))


def split_holdout(data: Data) -> Data:
    """Reserve the newest rows for eval. The boundary is the spec's, computed
    once, or train and eval would disagree about what is contaminated."""
    boundary = data.spec.holdout_boundary(len(data.table))
    return data.keeping("split_holdout", data.table.select(range(boundary)))


def filter_sources(data: Data) -> Data:
    return data.keeping(
        "filter_sources", data.table.filter(SOURCE_COLUMN, data.spec.keeps)
    )


def shuffle(data: Data) -> Data:
    """Before every head-taking stage below, which is what makes those heads
    a uniform sample rather than the corpus's source-grouped natural order."""
    order = data.rng.permutation(len(data.table))
    return data.keeping("shuffle", data.table.select(order))


def prefilter(data: Data) -> Data:
    minimum = tokens.min_chars_for(data.cfg.min_doc_tokens)
    return data.keeping(
        "prefilter",
        data.table.filter(data.spec.text_column, lambda text: len(text) >= minimum),
        f">= {minimum} chars",
    )


def balance(data: Data) -> Data:
    target = (
        data.limit_docs * PRESET_OVERSAMPLE if data.limit_docs else len(data.table)
    )
    return _balanced(data, "balance", target)


def tokenize(data: Data) -> Data:
    """The one impure stage. Sized by limit_docs and the over-fetch above."""
    encoded = data.tokenizer.encode_batch(
        data.table.column(data.spec.text_column), max_length=data.cfg.max_seq_len
    )
    return data.keeping(
        "tokenize", data.table.with_column(TOKENS_COLUMN, encoded)
    )


def drop_short(data: Data) -> Data:
    return data.keeping(
        "drop_short",
        data.table.filter(
            TOKENS_COLUMN, lambda ids: len(ids) >= data.cfg.min_doc_tokens
        ),
        f">= {data.cfg.min_doc_tokens} tokens",
    )


def rebalance(data: Data) -> Data:
    return _balanced(data, "rebalance", data.limit_docs or len(data.table))


def cap(data: Data) -> Data:
    """Last, so limit_docs bounds the true final count rather than an estimate."""
    if not data.limit_docs:
        return data.keeping("cap", data.table)
    keep = range(min(data.limit_docs, len(data.table)))
    return data.keeping("cap", data.table.select(keep))


def _balanced(data: Data, stage: str, target: int) -> Data:
    if not data.weights:
        return data.keeping(stage, data.table, "no preset")
    result = balance_math.balanced_indices(
        data.sources, data.weights, min(target, len(data.table))
    )
    return data.keeping(
        stage, data.table.select(result.indices), _shortfall_note(result)
    )


def _shortfall_note(result: balance_math.BalanceResult) -> str:
    short = result.short_sources
    if not short:
        return "on ratio"
    return "short: " + ", ".join(
        f"{q.source} {q.took}/{q.target}" for q in sorted(short, key=_by_source)
    )


def _by_source(quota: balance_math.SourceQuota) -> str:
    return quota.source


def rows_of(table: Table, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    """Column-major reads zipped into rows — one pass per column, not per row."""
    return list(zip(*(table.column(name) for name in columns), strict=True))
