SHELL := /bin/bash
PYTHON := .venv/bin/python
BOOTSTRAP_PYTHON := uv run --no-project --python 3.13 python
STACK := $(PYTHON) -m scripts.dev_stack

.PHONY: bootstrap prepare-artifacts dev up down restart status logs check test e2e publication-check

bootstrap:
	@$(BOOTSTRAP_PYTHON) -m scripts.bootstrap

prepare-artifacts:
	@$(PYTHON) -m scripts.prepare_artifacts

dev: up

up:
	@$(STACK) up

down:
	@$(STACK) down

restart:
	@$(STACK) restart

status:
	@$(STACK) status

logs:
	@$(STACK) logs

check:
	@$(PYTHON) -m scripts.validate_manifests
	@$(PYTHON) -m scripts.publication_check
	@uv run --all-packages --all-groups ruff check .
	@uv run --all-packages --all-groups ruff format --check .
	@npm --prefix apps/web run lint
	@npm --prefix apps/web run typecheck
	@npm --prefix apps/web run format:check

test:
	@uv run --all-packages --all-groups pytest
	@npm --prefix apps/web test
	@npm --prefix apps/web run build

e2e:
	@uv run --all-packages --all-groups pytest -m e2e

publication-check:
	@python3 -m scripts.publication_check
