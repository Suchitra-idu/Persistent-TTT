"""The single CLI surface: arguments in, frozen config out, once (D3).

`resolve` takes None for "not given". In `from_flags` every value in the flags
string is taken as given, zero included.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from ttt.core.config import presets, resolve as resolve_mod
from ttt.core.config.dataset import DatasetSpec
from ttt.core.config.train import TrainConfig
from ttt.core.config.ttt import TTTConfig, derive_layer_indices
from ttt.extensions import datasets, strategies
from ttt.extensions.strategies import Strategy

DEFAULT_LIMIT_DOCS = 4000
# Options that no longer exist but appear in recorded run commands, with the
# only value those commands used.
RETIRED = {"state_aware_gate": False}

DEFAULT_MODEL_SIZE = "0.6B"
ENV_DATASET = "TTT_DATASET"
ENV_MODEL_SIZE = "TTT_MODEL_SIZE"
ENV_BASE_MODEL = "TTT_BASE_MODEL"

def _settable(config_type, *, handled: frozenset[str]) -> tuple[str, ...]:
    """Every field the CLI may set, derived so a new field is reachable at once."""
    return tuple(
        field.name
        for field in dataclasses.fields(config_type)
        if field.name not in handled
    )


# strategy is a named argument of `resolve`; micro_batch_size is fixed at 1.
_TRAIN_FIELDS = _settable(TrainConfig, handled=frozenset({"strategy", "micro_batch_size"}))

_TTT_FIELDS = _settable(TTTConfig, handled=frozenset({"layer_indices"}))


def _strategy_fields() -> tuple[str, ...]:
    """The union over registered strategies, so a new plugin's knobs auto-enrol."""
    return tuple(
        sorted(
            {
                field.name
                for strategy in strategies.STRATEGIES.values()
                for field in dataclasses.fields(strategy)
                if field.name != "name"
            }
        )
    )


@dataclass(frozen=True)
class Resolved:
    """Everything a run needs, decided once. Nothing downstream re-resolves."""

    train: TrainConfig
    ttt: TTTConfig
    spec: DatasetSpec
    strategy: Strategy
    base_model: str = ""
    weights: Mapping[str, int] | None = None
    limit_docs: int | None = None
    resume_from: str = ""

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Sorted (field, repr) pairs — what an experiment records to be reproducible."""
        return (
            resolve_mod.describe(self.train)
            + resolve_mod.describe(self.ttt)
            + (
                ("base_model", self.base_model),
                ("dataset", self.spec.name),
                ("strategy", self.strategy.describe()),
            )
        )


def resolve(
    *,
    dataset: str | None = None,
    strategy: str | None = None,
    base_model: str | None = None,
    model_size: str | None = None,
    limit_docs: int | None = None,
    resume_from: str = "",
    layer_stride: int | None = None,
    layer_start: int | None = None,
    num_layers: int | None = None,
    **overrides: Any,
) -> Resolved:
    """Unknown keys raise, naming the valid ones: a typo costs GPU-hours."""
    overrides = _without_retired(overrides)
    _check_known(overrides)
    spec = datasets.get(dataset or datasets.DEFAULT)
    picked = strategies.get(strategy or TrainConfig().strategy)
    train = _train_config(overrides, strategy=strategy)

    return Resolved(
        train=train,
        ttt=_ttt_config(overrides, layer_stride, layer_start, num_layers),
        spec=spec,
        strategy=_strategy(picked, overrides),
        base_model=base_model or f"Qwen/Qwen3-{model_size or DEFAULT_MODEL_SIZE}",
        weights=_weights(spec),
        limit_docs=limit_docs,
        resume_from=resume_from,
    )


def from_flags(*, limit_docs: int = 0, **flags: Any) -> Resolved:
    """Every value in `flags` is taken as given. `limit_docs` unset is the paper's 4000."""
    return resolve(limit_docs=limit_docs or DEFAULT_LIMIT_DOCS, **flags)


def _train_config(
    overrides: Mapping[str, Any],
    *,
    strategy: str | None,
) -> TrainConfig:
    given = _taken(overrides, _TRAIN_FIELDS)
    if strategy:
        given["strategy"] = strategy
    return resolve_mod.merge(TrainConfig(), given)


def _ttt_config(
    overrides: Mapping[str, Any],
    layer_stride: int | None,
    layer_start: int | None,
    num_layers: int | None,
) -> TTTConfig:
    """Layer indices stay None until model depth is known, unless num_layers
    is given — the one place a config depends on a model that is not loaded yet."""
    cfg = resolve_mod.merge(TTTConfig(), _taken(overrides, _TTT_FIELDS))
    if num_layers is None:
        return cfg
    return dataclasses.replace(
        cfg,
        layer_indices=derive_layer_indices(
            num_layers,
            stride=layer_stride or 2,
            start=layer_start if layer_start is not None else 1,
        ),
    )


def _strategy(picked: Strategy, overrides: Mapping[str, Any]) -> Strategy:
    """Strategy knobs live on the plugin, not on TrainConfig (D4), so a knob
    the chosen strategy does not have is an error rather than a silent no-op."""
    given = _taken(overrides, _strategy_fields())
    if not given:
        return picked
    return resolve_mod.merge(picked, given)


def _weights(spec: DatasetSpec) -> Mapping[str, int] | None:
    return presets.get_preset(spec.default_source_preset) if spec.default_source_preset else None


def _without_retired(overrides: Mapping[str, Any]) -> dict[str, Any]:
    kept = dict(overrides)
    for name, only in RETIRED.items():
        if name in kept and _as_bool(name, kept.pop(name)) != only:
            raise ValueError(f"{name} was removed; only {name}={str(only).lower()} is still accepted")
    return kept


def _check_known(overrides: Mapping[str, Any]) -> None:
    known = _TRAIN_FIELDS + _TTT_FIELDS + _strategy_fields()
    unknown = sorted(set(overrides) - set(known))
    if unknown:
        raise ValueError(
            f"unknown option(s) {unknown}; the CLI surface is {sorted(known)}"
        )


def _taken(overrides: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: value
        for name, value in overrides.items()
        if name in fields and value is not None
    }


TRUTHY = ("1", "true", "yes", "on")
FALSY = ("0", "false", "no", "off")


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    raise ValueError(f"{name}={value!r} is not a boolean; use one of {TRUTHY + FALSY}")


def _flag_types() -> dict[str, type]:
    """name -> type, read off each field's default so a new config field becomes
    coercible the moment it exists."""
    types: dict[str, type] = {}
    for owner in (TrainConfig, TTTConfig, *strategies.STRATEGIES.values()):
        for field in dataclasses.fields(owner):
            if field.default not in (dataclasses.MISSING, None):
                types[field.name] = type(field.default)
    return types


_FLAG_TYPES = _flag_types()


def parse_flags(text: str) -> dict[str, Any]:
    """`"num_epochs=2,strategy=hybrid"` -> kwargs for `from_flags`.

    Modal builds an entrypoint's CLI from its signature and cannot express
    `**kwargs`, so the whole surface has to arrive as one string. Unknown names
    still raise in `resolve`, before anything reaches a GPU.
    """
    given: dict[str, Any] = {}
    for part in text.split(","):
        pair = part.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"expected name=value, got {pair!r}")
        name, _, value = pair.partition("=")
        given[name.strip()] = _flag_value(name.strip(), value.strip())
    return given


def _flag_value(name: str, text: str) -> Any:
    kind = _FLAG_TYPES.get(name)
    if kind is bool or name in RETIRED:
        return _as_bool(name, text)
    if kind in (int, float):
        return kind(text)
    if kind is str:
        return text
    return _guessed(text)


def _guessed(text: str) -> Any:
    """For the flags that are not config fields — `session`, `dataset`."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def env_defaults(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The three settings the laptop and the container must agree on. Read here
    and nowhere else (D3); an explicit argument still wins."""
    source = os.environ if environ is None else environ
    return resolve_mod.only_set(
        {
            "dataset": source.get(ENV_DATASET),
            "model_size": source.get(ENV_MODEL_SIZE),
            "base_model": source.get(ENV_BASE_MODEL),
        }
    )


def step_path(run_name: str, step: int) -> str:
    return f"{run_name}/step_{step}"


def invocation() -> str:
    """The command line that launched this process, for run provenance.

    Read on the laptop, in a local entrypoint — a Modal remote function has
    its own argv, not the `modal run ...` the user typed. Callers thread the
    result through as an explicit argument to reach a remote tracker.
    """
    return " ".join(sys.argv)


def resume_path(resume_from: str, run_name: str) -> str:
    """`step_N` is this run's; `other_run/step_N` is another's. Empty is no resume."""
    if not resume_from:
        return ""
    if "/" not in resume_from and not run_name:
        raise ValueError(f"resume_from={resume_from!r} names no run; pass run/{resume_from} or set run_name")
    return resume_from if "/" in resume_from else f"{run_name}/{resume_from}"
