# The repo-root virtualenv. Override on the command line to use another:
#   make check PYTHON=python LINT_IMPORTS=lint-imports
VENV         ?= $(abspath $(CURDIR)/../.venv)
PYTHON       ?= $(VENV)/bin/python
LINT_IMPORTS ?= $(VENV)/bin/lint-imports

export PYTHONPATH := .

.DEFAULT_GOAL := check

## check: the gate. Dependency Rule + the fast test tier. Nothing merges red.
check: lint test

## lint: the Dependency Rule, enforced (spec §9). A boundary violation is a failure.
lint:
	$(LINT_IMPORTS) --config .importlinter

## test: rings 0-4, excluding gpu/integration/slow (see pyproject markers).
test:
	$(PYTHON) -m pytest

## test-gpu: manual and gated. Needs a real H100; never runs in CI.
## Until Phase 6 populates tests/gpu/ this exits 5 ("no tests collected").
test-gpu:
	$(PYTHON) -m pytest -m gpu

## test-all: every tier including the quarantined ones.
test-all:
	$(PYTHON) -m pytest -m ""

## install-dev: the dev tier (pytest, hypothesis, import-linter).
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

## guard: prove the enforcement bites — plant a core -> transformers import,
## expect `make lint` to fail, then remove it. Should print PASS.
guard:
	@echo "import transformers" > ttt/core/_guard_violation.py
	@if $(MAKE) --no-print-directory lint >/dev/null 2>&1; then \
		rm -f ttt/core/_guard_violation.py; \
		echo "GUARD FAILED: a core -> transformers import was not caught"; \
		exit 1; \
	else \
		rm -f ttt/core/_guard_violation.py; \
		echo "GUARD PASS: core -> transformers is rejected"; \
	fi

.PHONY: check lint test test-gpu test-all install-dev guard
