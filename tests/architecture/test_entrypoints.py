"""Modal builds an entrypoint's CLI from its signature: `**kwargs` makes it unrunnable,
so the config surface travels as one `--flags` string (`cli.parse_flags`)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import ttt.experiments

VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD


def entrypoint_modules():
    for info in pkgutil.iter_modules(ttt.experiments.__path__):
        module = importlib.import_module(f"ttt.experiments.{info.name}")
        if hasattr(module, "app") and callable(getattr(module, "main", None)):
            yield module


MODULES = sorted(entrypoint_modules(), key=lambda m: m.__name__)


def signature_of(module):
    """The wrapped function, which Modal builds the CLI from."""
    return inspect.signature(module.main.info.raw_f, eval_str=True)


def test_every_entrypoint_takes_a_flags_string_and_no_kwargs():
    wrong = [
        m.__name__ for m in MODULES
        if VAR_KEYWORD in [p.kind for p in signature_of(m).parameters.values()]
        or getattr(signature_of(m).parameters.get("flags"), "annotation", None) is not str
    ]

    assert not wrong, f"entrypoints Modal cannot run: {wrong}"


def test_the_experiments_that_run_on_modal_were_actually_found():
    assert {m.__name__.rsplit(".", 1)[-1] for m in MODULES} >= {"train_v1", "paper_eval_v1", "sanity_check_v1"}
