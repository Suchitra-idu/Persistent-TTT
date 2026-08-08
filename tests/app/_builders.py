"""Builders for the Ring 4 suites. Every adapter here is a fake; no GPU, no disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ttt.adapters.fake_compute import FakeCompute
from ttt.adapters.fake_data_source import FakeDataSource
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.fake_generation import FakeGeneration
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.in_memory_tracker import InMemoryTracker
from ttt.adapters.list_table import ListTable
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app.data_pipeline import Doc
from ttt.app.stages import Data
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.train import TrainConfig
from ttt.core.types import Carry

LAYERS = (1, 3)
CHUNK = 4
SOURCES = ("alpha", "beta")

SPEC = DatasetSpec(
    name="app-fixture",
    source="_fixture",
    source_meta_column="meta",
    source_meta_key="set_name",
    include_sources=SOURCES,
    holdout_last_n=2,
)

WEIGHTS = {"alpha": 1, "beta": 1}


def config(**overrides) -> TrainConfig:
    """Small, fast, and with the cadences off unless a test turns them on."""
    defaults = dict(
        max_seq_len=64,
        min_doc_tokens=1,
        grad_accum_steps=2,
        num_epochs=1,
        warmup_ratio=0.0,
        warmup_min_steps=0,
        log_every=1000,
        save_every=1000,
        eval_every=0,
        eval_n_docs_per_source=1,
        eval_min_tokens=1,
        strategy="everlasting",
    )
    return TrainConfig(**{**defaults, **overrides})


def doc(index: int, *, source: str = "alpha", n_tokens: int = 8, n_bytes: int = 0) -> Doc:
    return Doc(
        index=index,
        source=source,
        token_ids=tuple((index + offset) % 251 + 5 for offset in range(n_tokens)),
        n_bytes=n_bytes,
    )


def docs(n: int, *, sources: Sequence[str] = SOURCES, n_tokens: int = 8) -> tuple[Doc, ...]:
    return tuple(
        doc(index, source=sources[index % len(sources)], n_tokens=n_tokens)
        for index in range(n)
    )


def carry(value: float = 1.0, *, layers: Sequence[int] = LAYERS) -> Carry:
    import torch

    return Carry(deltas={index: torch.tensor([[value]]) for index in layers})


@dataclass
class Wiring:
    """One object graph of fakes, wired the way Ring 5 wires the real ones."""

    losses: Sequence[float] = (1.0,)
    layers: Sequence[int] = LAYERS
    chunk_size: int = CHUNK
    decay: float = 0.9
    fast_weights: FakeFastWeights = field(init=False)
    compute: FakeCompute = field(init=False)
    tracker: InMemoryTracker = field(init=False)
    rng: ScriptedRng = field(init=False)

    def __post_init__(self) -> None:
        self.fast_weights = FakeFastWeights(
            self.layers, chunk_size=self.chunk_size, decay=self.decay
        )
        self.compute = FakeCompute(
            losses=self.losses, fast_weights=self.fast_weights
        )
        self.tracker = InMemoryTracker()
        self.rng = ScriptedRng()


def chat_wiring(script: Sequence[int] = (7, 8, 9), **kwargs):
    """A FakeGeneration whose tokens stage into the same FakeFastWeights a real
    forward would, so the switch matrix moves real state."""
    wiring = Wiring(**kwargs)
    generation = FakeGeneration(
        script, vocab_size=261, fast_weights=wiring.fast_weights
    )
    return wiring, generation, FakeTokenizer()


def rows(sources: Sequence[str], *, length: int = 40) -> list[dict]:
    return [
        {"text": f"{source[0]}" * (length + index), "meta": {"set_name": source}}
        for index, source in enumerate(sources)
    ]


def pipeline_data(
    source_labels: Sequence[str],
    *,
    cfg: TrainConfig | None = None,
    limit_docs: int | None = None,
    weights: Mapping[str, int] | None = None,
    length: int = 40,
) -> Data:
    table = FakeDataSource({SPEC.name: rows(source_labels, length=length)}).load(SPEC)
    return Data(
        table=table,
        spec=SPEC,
        cfg=cfg or config(),
        rng=ScriptedRng(),
        tokenizer=FakeTokenizer(),
        limit_docs=limit_docs,
        weights=weights,
    )


def table_of(rows_in: Sequence[Mapping]) -> ListTable:
    return ListTable(list(rows_in))


def data_source(source_labels: Sequence[str], *, length: int = 40) -> FakeDataSource:
    return FakeDataSource({SPEC.name: rows(source_labels, length=length)})
