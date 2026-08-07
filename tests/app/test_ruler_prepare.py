from __future__ import annotations

from ttt.adapters.fake_tokenizer import FakeTokenizer
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.numpy_rng import NumpyRng
from ttt.app import ruler_examples_io as io
from ttt.app import ruler_prepare
from ttt.core.config.ruler import RulerConfig
from ttt.core.ruler_types import RulerContext, RulerExample


class _FakeTask:
    name = "fake"
    version = 1

    def __init__(self) -> None:
        self.calls = 0

    def build(self, rng, tokenizer, seq_len, ctx) -> RulerExample:
        self.calls += 1
        return RulerExample(
            task=self.name,
            length_bucket=seq_len,
            prompt=f"p{self.calls}",
            answer_prefix=" Answer:",
            targets=("t",),
        )

    def score(self, prediction, example) -> float:
        return 1.0


def _prepare(cfg: RulerConfig, task: _FakeTask, storage: InMemoryStorage):
    return ruler_prepare.prepare(
        cfg=cfg,
        tasks={"fake": task},
        tokenizer=FakeTokenizer(),
        ctx=RulerContext(),
        rng=NumpyRng(0),
        storage=storage,
        announce=lambda _: None,
    )


def test_it_builds_num_samples_examples_per_bucket():
    task = _FakeTask()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=3)

    built = _prepare(cfg, task, InMemoryStorage())

    assert len(built["fake_128"]) == 3
    assert task.calls == 3


def test_it_builds_one_bucket_per_task_x_length():
    task = _FakeTask()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128, 256), num_samples=1)

    built = _prepare(cfg, task, InMemoryStorage())

    assert set(built) == {"fake_128", "fake_256"}


def test_it_does_not_regenerate_an_already_prepared_bucket():
    task = _FakeTask()
    storage = InMemoryStorage()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=2)

    _prepare(cfg, task, storage)
    _prepare(cfg, task, storage)

    assert task.calls == 2


def test_it_round_trips_through_storage():
    task = _FakeTask()
    storage = InMemoryStorage()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=1)

    built = _prepare(cfg, task, storage)

    assert io.read(storage, cfg, "fake", task.version, 128) == built["fake_128"]


def test_a_file_from_an_older_schema_version_does_not_block_regeneration():
    """Regression: a `RulerExample` field added after some sets were already
    prepared must not be read back from those older, now-incompatible bytes."""
    task = _FakeTask()
    storage = InMemoryStorage()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=1)
    stale_path = (
        f"ruler/v{io.SCHEMA_VERSION - 1}/{cfg.key('fake', 128)}-t{task.version}.jsonl"
    )
    storage.write_bytes(
        stale_path,
        b'{"task": "fake", "length_bucket": 128, "prompt": "old", "targets": ["t"]}\n',
    )

    built = _prepare(cfg, task, storage)

    assert task.calls == 1
    assert built["fake_128"][0].prompt == "p1"


def test_a_file_from_an_older_task_version_does_not_block_regeneration():
    """Regression: a task's own generator logic can change (a bugfix like
    vt's chain-order fix) without the row *shape* changing, so SCHEMA_VERSION
    alone can't catch it — this is what `RulerTask.version` is for."""
    task = _FakeTask()
    task.version = 2
    storage = InMemoryStorage()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=1)
    stale_path = f"ruler/v{io.SCHEMA_VERSION}/{cfg.key('fake', 128)}-t1.jsonl"
    storage.write_bytes(
        stale_path,
        b'{"task": "fake", "length_bucket": 128, "prompt": "old", '
        b'"answer_prefix": " Answer:", "targets": ["t"]}\n',
    )

    built = _prepare(cfg, task, storage)

    assert task.calls == 1
    assert built["fake_128"][0].prompt == "p1"


def test_it_commits_the_storage():
    storage = InMemoryStorage()
    cfg = RulerConfig(tasks=("fake",), context_lengths=(128,), num_samples=1)

    _prepare(cfg, _FakeTask(), storage)

    assert storage.commits == 1
