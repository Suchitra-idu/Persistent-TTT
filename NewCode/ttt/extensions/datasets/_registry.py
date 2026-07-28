"""The DATASETS registry.

The spec type is `core.config.dataset.DatasetSpec` — a frozen dataclass, so
there is no separate protocol to declare (D5). What a plugin adds is one
instance; what the registry adds is a name and the D9 guarantee, checked at
registration rather than at container spin-up.
"""

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
