"""HfDataSource — a Hub dataset can name a config, not just a repo."""

from __future__ import annotations

from datasets import Dataset

from ttt.adapters.hf_data_source import HfDataSource
from ttt.core.config.dataset import DatasetSpec


def _spec(**overrides) -> DatasetSpec:
    defaults = dict(
        name="wiki-mg",
        source="wikimedia/wikipedia",
        source_meta_column="meta",
        source_meta_key="lang",
    )
    return DatasetSpec(**{**defaults, **overrides})


def test_it_passes_the_specs_config_to_load_dataset(monkeypatch):
    seen = {}

    def fake_load_dataset(source, config, split):
        seen.update(source=source, config=config, split=split)
        return Dataset.from_dict({"text": ["a"], "meta": ['{"lang": "mg"}']})

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    HfDataSource().load(_spec(config="20231101.mg"))

    assert seen == {"source": "wikimedia/wikipedia", "config": "20231101.mg", "split": "train"}


def test_an_unset_config_reaches_load_dataset_as_none(monkeypatch):
    seen = {}

    def fake_load_dataset(source, config, split):
        seen.update(source=source, config=config, split=split)
        return Dataset.from_dict({"text": ["a"], "meta": ['{"lang": "mg"}']})

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    HfDataSource().load(_spec())

    assert seen["config"] is None
