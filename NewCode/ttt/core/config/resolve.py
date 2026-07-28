"""Pure config resolution: defaults + overrides -> a new frozen config (D3).

Generic, so a strategy plugin's own config resolves through the same function.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, TypeVar

T = TypeVar("T")

_SIMPLE_TYPES = {"int": int, "float": float, "bool": bool, "str": str}


def merge(defaults: T, overrides: Mapping[str, Any]) -> T:
    """Strict: an unknown key is an error naming the valid fields, not a
    silently ignored typo that costs six GPU-hours at the default LR."""
    if not dataclasses.is_dataclass(defaults):
        raise TypeError(
            f"merge needs a dataclass instance, got {type(defaults).__name__}"
        )
    if not overrides:
        return defaults

    fields = {f.name: f for f in dataclasses.fields(defaults)}
    unknown = sorted(set(overrides) - set(fields))
    if unknown:
        raise ValueError(
            f"unknown config field(s) {unknown} for "
            f"{type(defaults).__name__}; valid fields are {sorted(fields)}"
        )
    for name, value in overrides.items():
        _check_type(type(defaults).__name__, name, fields[name].type, value)
    return dataclasses.replace(defaults, **dict(overrides))


def only_set(overrides: Mapping[str, Any], unset: Iterable[Any] = (None,)) -> dict:
    """Drop the sentinels a CLI that cannot express Optional[int] uses for
    "leave this alone". Keeps False distinct from 0."""
    sentinels = list(unset)
    return {
        name: value
        for name, value in overrides.items()
        if not any(_is_sentinel(value, sentinel) for sentinel in sentinels)
    }


def describe(config: Any) -> tuple[tuple[str, str], ...]:
    """Sorted (field, repr) pairs — what an experiment records to be reproducible."""
    if not dataclasses.is_dataclass(config):
        raise TypeError(
            f"describe needs a dataclass instance, got {type(config).__name__}"
        )
    return tuple(
        (f.name, repr(getattr(config, f.name)))
        for f in sorted(dataclasses.fields(config), key=lambda f: f.name)
    )


def _is_sentinel(value: Any, sentinel: Any) -> bool:
    if value is sentinel:
        return True
    if isinstance(value, bool) != isinstance(sentinel, bool):
        return False
    return type(value) is type(sentinel) and value == sentinel


def _check_type(owner: str, name: str, declared: Any, value: Any) -> None:
    """Only simply-typed fields are checked; a half-enforced check that looked
    complete would be worse than an honest partial one."""
    annotation = declared if isinstance(declared, str) else getattr(
        declared, "__name__", ""
    )
    expected = _SIMPLE_TYPES.get(annotation)
    if expected is None:
        return
    if expected is bool and not isinstance(value, bool):
        raise TypeError(f"{owner}.{name} must be a bool, got {value!r}")
    if expected is not bool and isinstance(value, bool):
        raise TypeError(f"{owner}.{name} must be {annotation}, got {value!r}")
    if expected is float and isinstance(value, int):
        return
    if not isinstance(value, expected):
        raise TypeError(f"{owner}.{name} must be {annotation}, got {value!r}")
