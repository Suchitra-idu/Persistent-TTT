"""Every Modal entrypoint has to be runnable from the command line.

Modal builds an entrypoint's CLI from its signature. `**kwargs` becomes a single
required `--flags ANY` that no valid invocation can satisfy, so an entrypoint
that takes one is unrunnable — and nothing catches it until someone pays for a
GPU. The whole config surface travels as one `--flags` string instead
(`cli.parse_flags`).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import ttt.experiments

VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD


def entrypoint_modules():
    for info in pkgutil.iter_modules(ttt.experiments.__path__):
        module = importlib.import_module(f"ttt.experiments.{info.name}")
        if hasattr(module, "app") and callable(getattr(module, "main", None)):
            yield module


MODULES = sorted(entrypoint_modules(), key=lambda m: m.__name__)


def signature_of(module):
    """`local_entrypoint` returns a wrapper whose own signature is `(*args,
    **kwargs)`; the function Modal builds the CLI from is `.info.raw_f`.
    `eval_str` because these modules use PEP 563.
    """
    return inspect.signature(module.main.info.raw_f, eval_str=True)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_an_entrypoint_does_not_take_kwargs(module):
    kinds = [p.kind for p in signature_of(module).parameters.values()]

    assert VAR_KEYWORD not in kinds


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_an_entrypoint_accepts_a_flags_string(module):
    assert signature_of(module).parameters["flags"].annotation is str


def test_the_experiments_that_run_on_modal_were_actually_found():
    assert len(MODULES) >= 6
