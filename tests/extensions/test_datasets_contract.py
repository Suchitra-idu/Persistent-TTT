"""Contract suite for the datasets registry. Every plugin auto-enrols.

D9's "every row carries a source label" lives here, as a registration-time
contract rather than the runtime check `everlasting_carry` used to make.
"""

from __future__ import annotations

import pytest

from ttt.extensions.datasets import DATASETS, DEFAULT, get, register
from tests.extensions import _builders

CASES = [pytest.param(spec, id=name) for name, spec in sorted(DATASETS.items())]

MULTI_SOURCE_CASES = [
    pytest.param(spec, id=name)
    for name, spec in sorted(DATASETS.items())
    if spec.is_multi_source
]


def test_the_registry_has_the_corpus_and_the_fixture():
    assert sorted(DATASETS) == [
        "fixture",
        "lang-transfer-eval",
        "lang-transfer-train",
        "slimpajama-6b",
    ]


def test_the_default_spec_is_registered():
    assert get(DEFAULT).name == DEFAULT


@pytest.mark.parametrize("spec", CASES)
def test_every_spec_is_registered_under_its_own_name(spec):
    assert get(spec.name) is spec


@pytest.mark.parametrize("spec", CASES)
def test_every_spec_labels_every_row(spec):
    source = _builders.a_source_of(spec)

    assert spec.source_of(_builders.labelled_row(spec, source)) == source


@pytest.mark.parametrize("spec", CASES)
def test_every_specs_own_sources_pass_its_filter(spec):
    assert spec.keeps(_builders.a_source_of(spec))


@pytest.mark.parametrize("spec", CASES)
def test_no_spec_keeps_a_source_it_does_not_declare(spec):
    assert not spec.keeps("NotASource")


@pytest.mark.parametrize("spec", CASES)
def test_the_holdout_never_exceeds_the_pool(spec):
    assert spec.holdout_boundary(3) == 0


@pytest.mark.parametrize("spec", CASES)
def test_the_holdout_is_the_newest_rows(spec):
    n_rows = spec.holdout_last_n + 17

    assert spec.holdout_boundary(n_rows) == 17


@pytest.mark.parametrize("spec", MULTI_SOURCE_CASES)
def test_a_multi_source_spec_rejects_a_row_with_no_label(spec):
    with pytest.raises(ValueError, match="source label"):
        spec.source_of({spec.text_column: "x", spec.source_meta_column: {}})


@pytest.mark.parametrize("spec", MULTI_SOURCE_CASES)
def test_a_multi_source_spec_rejects_a_row_missing_the_meta_column(spec):
    with pytest.raises(ValueError, match="no usable"):
        spec.source_of({spec.text_column: "x"})


@pytest.mark.parametrize("spec", MULTI_SOURCE_CASES)
def test_a_multi_source_spec_reads_a_json_string_meta_column(spec):
    source = _builders.a_source_of(spec)
    encoded = f'{{"{spec.source_meta_key}": "{source}"}}'

    assert spec.source_of({spec.source_meta_column: encoded}) == source


def test_registering_a_name_twice_is_an_error():
    with pytest.raises(ValueError, match="already registered"):
        register(get(DEFAULT))


def test_get_rejects_an_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        get("arxiv")
