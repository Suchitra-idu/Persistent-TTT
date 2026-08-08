"""WikipediaLangSource — streams a config, stops at the row limit, drops stubs."""

from __future__ import annotations

from ttt.adapters.wikipedia_lang_source import WikipediaLangSource


def _fake_stream(rows):
    def fake_load_dataset(repo, config, *, split, streaming):
        assert (repo, split, streaming) == ("wikimedia/wikipedia", "train", True)
        assert config == "20231101.mg"
        return iter(rows)

    return fake_load_dataset


def test_it_stops_at_the_row_limit(monkeypatch):
    rows = [{"text": "x" * 300, "title": f"t{i}"} for i in range(10)]
    monkeypatch.setattr("datasets.load_dataset", _fake_stream(rows))

    got = WikipediaLangSource().load("mg", "20231101", limit=3)

    assert len(got) == 3


def test_it_drops_rows_shorter_than_min_chars(monkeypatch):
    rows = [{"text": "short", "title": "a"}, {"text": "y" * 300, "title": "b"}]
    monkeypatch.setattr("datasets.load_dataset", _fake_stream(rows))

    got = WikipediaLangSource().load("mg", "20231101", limit=10, min_chars=200)

    assert [row["title"] for row in got] == ["b"]


def test_it_returns_fewer_rows_when_the_stream_runs_out(monkeypatch):
    rows = [{"text": "x" * 300, "title": "a"}]
    monkeypatch.setattr("datasets.load_dataset", _fake_stream(rows))

    got = WikipediaLangSource().load("mg", "20231101", limit=10)

    assert len(got) == 1
