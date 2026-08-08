from __future__ import annotations

import io
import json

import pyarrow.parquet as pq
import pytest

from ttt.adapters.fake_wiki_lang_source import FakeWikiLangSource
from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.scripted_rng import ScriptedRng
from ttt.app import lang_corpus_prepare as prep

LANGUAGES = (("mg", "Malagasy"), ("sah", "Sakha"))


def _rows(n: int, *, prefix: str = "x") -> list[dict]:
    return [{"text": f"{prefix}{i}" * 100, "title": f"t{i}"} for i in range(n)]


def _prepare(*, storage=None, languages=LANGUAGES, target_rows=5, wiki_rows=None):
    storage = storage or InMemoryStorage()
    wiki = FakeWikiLangSource(wiki_rows or {"mg": _rows(5), "sah": _rows(5)})
    counts = prep.prepare(
        languages=languages,
        snapshot="20231101",
        wiki_source=wiki,
        storage=storage,
        role="train",
        target_rows=target_rows,
        announce=lambda _: None,
    )
    return counts, storage, wiki


def test_it_writes_one_parquet_per_language():
    _, storage, _ = _prepare()

    assert storage.exists("lang_corpus/train/mg.parquet")
    assert storage.exists("lang_corpus/train/sah.parquet")


def test_it_returns_the_row_count_written_per_language():
    counts, _, _ = _prepare(wiki_rows={"mg": _rows(3), "sah": _rows(5)})

    assert counts == {"mg": 3, "sah": 5}


def test_it_skips_a_language_already_prepared():
    storage = InMemoryStorage()
    storage.write_bytes("lang_corpus/train/mg.parquet", b"stale")

    _, _, wiki = _prepare(storage=storage)

    assert "mg" not in {code for code, _, _ in wiki.calls}
    assert storage.read_bytes("lang_corpus/train/mg.parquet") == b"stale"


def test_it_raises_when_a_language_has_no_rows():
    with pytest.raises(RuntimeError, match="no rows fetched"):
        _prepare(wiki_rows={"mg": [], "sah": _rows(5)})


def test_it_commits_once_after_the_whole_pass():
    _, storage, _ = _prepare()

    assert storage.commits == 1


def test_the_written_parquet_round_trips_with_a_lang_labelled_meta_column():
    _, storage, _ = _prepare(languages=(("mg", "Malagasy"),), wiki_rows={"mg": _rows(2)})

    table = pq.read_table(io.BytesIO(storage.read_bytes("lang_corpus/train/mg.parquet")))
    rows = table.to_pylist()

    assert len(rows) == 2
    assert {json.loads(row["meta"])["lang"] for row in rows} == {"mg"}
    assert all(row["text"] for row in rows)


def _combine(*, storage, languages=LANGUAGES, rng=None):
    return prep.combine(
        languages=languages,
        storage=storage,
        role="train",
        rng=rng or ScriptedRng(),
        announce=lambda _: None,
    )


class TestCombine:
    def test_it_writes_one_file_merging_every_language(self):
        _, storage, _ = _prepare(wiki_rows={"mg": _rows(3), "sah": _rows(5)})

        n = _combine(storage=storage)

        assert n == 8
        assert storage.exists("lang_corpus_combined/train/all.parquet")

    def test_it_raises_if_a_language_was_never_prepared(self):
        _, storage, _ = _prepare(languages=(("mg", "Malagasy"),), wiki_rows={"mg": _rows(2)})

        with pytest.raises(RuntimeError, match="has not been prepared"):
            _combine(storage=storage)

    def test_it_shuffles_using_the_given_rng_not_source_order(self):
        _, storage, _ = _prepare(
            wiki_rows={"mg": _rows(3, prefix="m"), "sah": _rows(3, prefix="s")}
        )
        # Reverses the 6 concatenated rows: sah, sah, sah, mg, mg, mg — the
        # opposite of mg-then-sah concatenation order.
        rng = ScriptedRng(permutations=[[5, 4, 3, 2, 1, 0]])

        _combine(storage=storage, rng=rng)

        table = pq.read_table(io.BytesIO(storage.read_bytes("lang_corpus_combined/train/all.parquet")))
        langs = [json.loads(m)["lang"] for m in table.column("meta").to_pylist()]

        assert langs == ["sah", "sah", "sah", "mg", "mg", "mg"]

    def test_it_rebuilds_even_if_a_combined_file_already_exists(self):
        """A stale file from a previous, differently-languaged run must not
        survive a re-run — the exact bug a naive exists() short-circuit
        caused in production (docs/experiments-map.md)."""
        _, storage, _ = _prepare(wiki_rows={"mg": _rows(3), "sah": _rows(5)})
        _combine(storage=storage)

        n = _combine(storage=storage)

        assert n == 8

    def test_it_commits_after_writing(self):
        _, storage, _ = _prepare(wiki_rows={"mg": _rows(3), "sah": _rows(5)})
        before = storage.commits

        _combine(storage=storage)

        assert storage.commits == before + 1
