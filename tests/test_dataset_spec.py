"""Tests for the dataset-spec abstraction: registry lookup, source
extraction (both struct and JSON-string meta), and include-source
membership. No `datasets` dep -- the pure logic is tested; the ds.map
integration lives in the load path and is exercised at run time."""

import json

import pytest

from data_utils import extract_source_from_row
from ttt_config import DATASETS, DatasetSpec, get_dataset_spec


# ---------- registry ----------

def test_registry_has_arxiv_and_slimpajama():
    assert "arxiv" in DATASETS
    assert "slimpajama-6b" in DATASETS


def test_get_dataset_spec_returns_spec():
    spec = get_dataset_spec("arxiv")
    assert spec.name == "arxiv"
    assert spec.text_column == "text"


def test_get_dataset_spec_unknown_raises():
    with pytest.raises(KeyError, match="Unknown TTT_DATASET"):
        get_dataset_spec("does-not-exist")


def test_slimpajama_spec_excludes_commoncrawl():
    spec = get_dataset_spec("slimpajama-6b")
    assert "RedPajamaCommonCrawl" not in spec.include_sources
    # And every other advertised source is included:
    for src in ("RedPajamaC4", "RedPajamaGithub", "RedPajamaBook",
                "RedPajamaArXiv", "RedPajamaWikipedia",
                "RedPajamaStackExchange"):
        assert src in spec.include_sources


def test_slimpajama_spec_meta_field_wiring():
    spec = get_dataset_spec("slimpajama-6b")
    assert spec.source_meta_column == "meta"
    assert spec.source_meta_key == "redpajama_set_name"


# ---------- extract_source_from_row ----------

def _slim_spec():
    return get_dataset_spec("slimpajama-6b")


def test_extract_source_from_struct_meta():
    row = {"text": "hello", "meta": {"redpajama_set_name": "RedPajamaC4"}}
    assert extract_source_from_row(row, _slim_spec()) == "RedPajamaC4"


def test_extract_source_from_json_string_meta():
    row = {"text": "hello",
           "meta": json.dumps({"redpajama_set_name": "RedPajamaGithub"})}
    assert extract_source_from_row(row, _slim_spec()) == "RedPajamaGithub"


def test_extract_source_missing_meta_returns_empty():
    row = {"text": "hello"}
    assert extract_source_from_row(row, _slim_spec()) == ""


def test_extract_source_malformed_json_returns_empty():
    row = {"text": "hello", "meta": "not valid json {"}
    assert extract_source_from_row(row, _slim_spec()) == ""


def test_extract_source_wrong_type_returns_empty():
    row = {"text": "hello", "meta": 42}
    assert extract_source_from_row(row, _slim_spec()) == ""


def test_extract_source_missing_key_returns_empty():
    row = {"text": "hello", "meta": {"other_field": "x"}}
    assert extract_source_from_row(row, _slim_spec()) == ""


def test_extract_source_no_config_returns_empty():
    """A spec with no source_meta_column/key returns "" regardless."""
    spec = DatasetSpec(name="test", source="x/y")
    row = {"text": "hello", "meta": {"redpajama_set_name": "RedPajamaC4"}}
    assert extract_source_from_row(row, spec) == ""


# ---------- DatasetSpec sanity ----------

def test_spec_is_frozen_and_hashable():
    spec = DatasetSpec(name="test", source="a/b", include_sources=("x",))
    with pytest.raises(Exception):
        spec.name = "other"
    assert hash(spec) is not None


def test_active_dataset_spec_matches_env_default():
    """When TTT_DATASET is unset (test env), the active spec is arxiv."""
    from ttt_config import DATASET_NAME, DATASET_SPEC
    assert DATASET_NAME == DATASET_SPEC.name
