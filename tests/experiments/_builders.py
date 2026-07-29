"""An Engine built out of fakes, so an experiment runs with no GPU and no Modal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tests.app import _builders as app_builders
from ttt import cli
from ttt.adapters.fake_compute import FakeCompute
from ttt.adapters.fake_clock import FakeClock
from ttt.adapters.fake_data_source import FakeDataSource
from ttt.adapters.fake_fast_weights import FakeFastWeights
from ttt.adapters.fake_generation import FakeGeneration
from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.in_memory_tracker import InMemoryTracker
from ttt.core.types import Carry
from ttt.experiments._runtime import Engine

LAYERS = app_builders.LAYERS
SOURCES = ("FixtureProse", "FixtureCode", "FixtureMath")

# Long enough to clear the prefilter's 3.5 chars/token at min_doc_tokens=8.
DOC_CHARS = 120


def resolved(**overrides) -> cli.Resolved:
    """The fixture dataset and a tiny budget: everything else is the real default."""
    defaults = dict(
        dataset="fixture",
        strategy="everlasting",
        num_epochs=1,
        grad_accum_steps=2,
        min_doc_tokens=8,
        max_seq_len=64,
        eval_n_slices=2,
        eval_n_docs_per_source=1,
        eval_min_tokens=1,
        eval_every=0,
        save_every=1000,
        log_every=1000,
        warmup_min_steps=0,
        wandb_enabled=False,
    )
    return cli.resolve(**{**defaults, **overrides})


def rows(sources: Sequence[str] = SOURCES * 4) -> list[dict]:
    return [
        {"text": chr(ord("a") + index % 26) * DOC_CHARS, "meta": {"redpajama_set_name": source}}
        for index, source in enumerate(sources)
    ]


def data_source(sources: Sequence[str] = SOURCES * 4) -> FakeDataSource:
    return FakeDataSource({"fixture": rows(sources)})


@dataclass
class Saves:
    calls: list[dict] = field(default_factory=list)

    def __call__(self, **kwargs) -> None:
        self.calls.append(
            {**kwargs, "carries": dict(kwargs["carries"]), "n_updates": dict(kwargs["n_updates"])}
        )


def engine(
    resolution: cli.Resolved | None = None,
    *,
    losses: Sequence[float] = (1.4, 1.2, 1.0),
    script: Sequence[int] = (7, 8, 9),
    carries: Mapping[str, Carry] | None = None,
) -> Engine:
    resolution = resolution or resolved()
    fast_weights = FakeFastWeights(
        LAYERS, chunk_size=4, decay=resolution.ttt.carried_decay
    )
    built = Engine(
        resolved=resolution,
        tokenizer=FakeTokenizer(),
        compute=FakeCompute(losses=losses, fast_weights=fast_weights),
        fast_weights=fast_weights,
        generation=FakeGeneration(script, vocab_size=261, fast_weights=fast_weights),
        tracker=InMemoryTracker(),
        storage=InMemoryStorage(),
        clock=FakeClock(),
        root="/ckpt",
        save=Saves(),
    )
    built.carries = lambda: (dict(carries or {}), {})
    return built


def nothing_held_out(resolution: cli.Resolved) -> cli.Resolved:
    """A populated corpus whose holdout is empty — the shape a real run hits
    when holdout_last_n outruns the pool, not a corpus with no columns."""
    import dataclasses

    return dataclasses.replace(
        resolution, spec=dataclasses.replace(resolution.spec, holdout_last_n=0)
    )


def carry(value: float = 1.0) -> Carry:
    return app_builders.carry(value)
