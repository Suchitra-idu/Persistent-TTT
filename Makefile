# The repo-root virtualenv. Override on the command line to use another:
#   make check PYTHON=python LINT_IMPORTS=lint-imports
VENV         ?= $(abspath $(CURDIR)/.venv)
PYTHON       ?= $(VENV)/bin/python
LINT_IMPORTS ?= $(VENV)/bin/lint-imports

export PYTHONPATH := .

# Silences mkdocs-material's advocacy banner about a future MkDocs 2.0. It is
# not a build warning; `docs-build` is --strict and clean.
export NO_MKDOCS_2_WARNING := 1

.DEFAULT_GOAL := check

## check: the gate. Dependency Rule + the fast test tier. Nothing merges red.
check: lint test

## lint: the Dependency Rule, enforced (spec §9). A boundary violation is a failure.
lint:
	$(LINT_IMPORTS) --config .importlinter

## test: rings 0-4, excluding gpu/integration/slow (see pyproject markers).
test:
	$(PYTHON) -m pytest

## test-gpu-modal: the gpu tier on a real H100. The supported path; costs money.
test-gpu-modal:
	$(VENV)/bin/modal run tests/gpu/run_on_modal.py

## test-gpu: the same tier against a local GPU. Only meaningful on sm_80 or newer
## — the base model is bfloat16, so anything older measures nothing.
# Overrides addopts wholesale: the default tier's -q and its `not gpu` filter are
# both wrong here, and -s is what lets the download's progress bar reach you.
test-gpu:
	$(PYTHON) -m pytest tests/gpu -o addopts="" -m gpu -v -s

## gpu-warm: pull the base model into the local HF cache. Resumable, shows a bar.
gpu-warm:
	$(PYTHON) -c "from huggingface_hub import snapshot_download; \
	from ttt import cli; print(snapshot_download(cli.resolve().base_model))"

## test-all: every tier including the quarantined ones.
test-all:
	$(PYTHON) -m pytest -m ""

## docs: serve docs/ at http://127.0.0.1:8000 with live reload.
docs:
	$(VENV)/bin/mkdocs serve

## docs-build: build the site into site/. --strict fails on a broken link.
docs-build:
	$(VENV)/bin/mkdocs build --strict

## install-dev: the dev tier (pytest, hypothesis, import-linter, mkdocs).
install-dev:
	$(PYTHON) -m pip install -e ".[dev,docs]"

## guard: prove the enforcement bites — plant a banned import in each inner
## ring, expect `make lint` to fail on it, then remove it. Should print PASS.
guard: guard-core guard-app

guard-core:
	@$(MAKE) --no-print-directory _guard \
		FILE=ttt/core/_guard_violation.py LINE="import transformers" \
		WHAT="core -> transformers"

guard-app:
	@$(MAKE) --no-print-directory _guard \
		FILE=ttt/app/_guard_violation.py LINE="import torch" \
		WHAT="app -> torch"

_guard:
	@echo "$(LINE)" > $(FILE)
	@if $(MAKE) --no-print-directory lint >/dev/null 2>&1; then \
		rm -f $(FILE); \
		echo "GUARD FAILED: $(WHAT) was not caught"; \
		exit 1; \
	else \
		rm -f $(FILE); \
		echo "GUARD PASS: $(WHAT) is rejected"; \
	fi

.PHONY: check lint test test-gpu test-all docs docs-build install-dev guard \
        guard-core guard-app _guard
