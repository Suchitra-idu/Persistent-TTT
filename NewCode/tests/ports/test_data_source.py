from __future__ import annotations

import pytest

from tests.ports import _builders
from ttt.adapters.fake_data_source import FakeDataSource
from ttt.adapters.hf_data_source import HfDataSource
from ttt.ports.data_source import DataSource
from ttt.ports.table import SOURCE_COLUMN

SOURCES = ("alpha", "beta", "alpha", "gamma")


class DataSourceConformance:
    @pytest.fixture
    def source(self):
        raise NotImplementedError

    @pytest.fixture
    def spec(self):
        raise NotImplementedError

    @pytest.fixture
    def unlabelled_spec(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, source):
        assert isinstance(source, DataSource)

    def test_it_yields_one_row_per_document(self, source, spec):
        assert len(source.load(spec)) == len(SOURCES)

    def test_every_row_carries_a_source_label(self, source, spec):
        assert source.load(spec).column(SOURCE_COLUMN) == list(SOURCES)

    def test_the_text_column_survives(self, source, spec):
        assert spec.text_column in source.load(spec).column_names

    def test_rows_keep_corpus_order(self, source, spec):
        texts = source.load(spec).column(spec.text_column)

        assert texts == [f"doc {i}" for i in range(len(SOURCES))]

    def test_a_row_with_no_label_is_rejected(self, source, unlabelled_spec):
        with pytest.raises(ValueError, match="source label"):
            source.load(unlabelled_spec)


class TestFakeDataSource(DataSourceConformance):
    @pytest.fixture
    def spec(self):
        return _builders.FIXTURE_SPEC

    @pytest.fixture
    def unlabelled_spec(self):
        return _builders.UNLABELLED_SPEC

    @pytest.fixture
    def source(self, spec, unlabelled_spec):
        return FakeDataSource(
            {
                spec.name: _builders.spec_rows(SOURCES),
                unlabelled_spec.name: [{"text": "doc 0", "meta": {}}],
            }
        )

    def test_an_unscripted_spec_raises(self, source):
        with pytest.raises(KeyError, match="no rows scripted"):
            source.load(_builders.CONSTANT_SPEC)

    def test_a_constant_source_spec_stamps_every_row(self):
        spec = _builders.CONSTANT_SPEC
        source = FakeDataSource({spec.name: [{"text": "x"}, {"text": "y"}]})

        assert source.load(spec).column(SOURCE_COLUMN) == ["only", "only"]


@pytest.mark.integration
class TestHfDataSource(DataSourceConformance):
    @pytest.fixture
    def spec(self, tmp_path):
        return _builders.saved_to_disk(
            tmp_path / "labelled", _builders.FIXTURE_SPEC, _builders.spec_rows(SOURCES)
        )

    @pytest.fixture
    def unlabelled_spec(self, tmp_path):
        return _builders.saved_to_disk(
            tmp_path / "unlabelled",
            _builders.UNLABELLED_SPEC,
            [{"text": "doc 0", "meta": {}}],
        )

    @pytest.fixture
    def source(self):
        return HfDataSource()
