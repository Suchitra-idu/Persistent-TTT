"""The DATASETS registry of `DatasetSpec`s (D5)."""

from __future__ import annotations

from ttt.core.config.dataset import DatasetSpec

DEFAULT = "slimpajama-6b"

DATASETS: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    if spec.name in DATASETS:
        raise ValueError(
            f"dataset spec {spec.name!r} is already registered; a name collision "
            "would make the resolved config ambiguous"
        )
    DATASETS[spec.name] = spec
    return spec


def get(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}. Known: {sorted(DATASETS)}")
    return DATASETS[name]
