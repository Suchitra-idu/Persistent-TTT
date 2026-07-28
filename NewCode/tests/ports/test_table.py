from __future__ import annotations

import pytest

from tests.ports import _builders
from ttt.adapters.hf_table import HfTable
from ttt.adapters.list_table import ListTable
from ttt.ports.table import Table

ROWS = _builders.table_rows(5)


class TableConformance:
    @pytest.fixture
    def table(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, table):
        assert isinstance(table, Table)

    def test_length_is_the_row_count(self, table):
        assert len(table) == len(ROWS)

    def test_column_names_cover_every_key(self, table):
        assert set(table.column_names) == set(ROWS[0])

    def test_a_column_comes_back_in_row_order(self, table):
        assert table.column("n") == [row["n"] for row in ROWS]

    def test_an_absent_column_raises(self, table):
        with pytest.raises(KeyError):
            table.column("nope")

    def test_a_row_is_a_dict_of_every_column(self, table):
        assert table.row(2) == ROWS[2]

    def test_select_filters(self, table):
        assert table.select([0, 2]).column("n") == [0, 2]

    def test_select_reorders(self, table):
        assert table.select([3, 1]).column("n") == [3, 1]

    def test_select_leaves_the_original_alone(self, table):
        table.select([0])

        assert len(table) == len(ROWS)

    def test_select_of_nothing_is_empty(self, table):
        assert len(table.select([])) == 0

    def test_with_column_adds_it(self, table):
        added = table.with_column("extra", list(range(len(ROWS))))

        assert "extra" in added.column_names and added.column("extra")[3] == 3

    def test_with_column_replaces_a_column_of_the_same_name(self, table):
        replaced = table.with_column("n", [9] * len(ROWS))

        assert replaced.column("n") == [9] * len(ROWS)
        assert replaced.column_names.count("n") == 1

    def test_with_column_leaves_the_original_alone(self, table):
        table.with_column("n", [9] * len(ROWS))

        assert table.column("n") == [row["n"] for row in ROWS]

    def test_a_wrong_length_column_is_rejected(self, table):
        with pytest.raises(ValueError):
            table.with_column("extra", [1])


class TestListTable(TableConformance):
    @pytest.fixture
    def table(self):
        return ListTable(ROWS)

    def test_a_ragged_table_is_rejected(self):
        with pytest.raises(ValueError, match="ragged"):
            ListTable([{"a": 1}, {"a": 1, "b": 2}])

    def test_an_empty_table_has_no_columns(self):
        assert ListTable([]).column_names == ()


@pytest.mark.integration
class TestHfTable(TableConformance):
    @pytest.fixture
    def table(self):
        datasets = pytest.importorskip("datasets")
        return HfTable(datasets.Dataset.from_list(ROWS))
