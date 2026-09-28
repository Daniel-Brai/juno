.RECIPEPREFIX = >
.DEFAULT_GOAL := help
.PHONY: help setup fmt lint typecheck imports test test-all cov hooks check release

UV ?= uv
PUSH ?= false

help:
> @printf "Available targets:\n"
> @printf "  make setup\t\t\tInstall all dependency groups and the git hooks\n"
> @printf "  make fmt\t\t\tFormat and auto-fix lint issues\n"
> @printf "  make lint\t\t\tCheck lint and formatting\n"
> @printf "  make typecheck\t\tRun mypy in strict mode\n"
> @printf "  make imports\t\t\tCheck the layering contracts\n"
> @printf "  make test\t\t\tRun the fast test suite in parallel\n"
> @printf "  make test-all\t\t\tAlso run slow, gpu and live_api tests\n"
> @printf "  make cov\t\t\tRun tests with coverage\n"
> @printf "  make hooks\t\t\tRun every pre-commit hook on all files\n"
> @printf "  make check\t\t\tWhat CI runs: lint, typecheck, imports and test\n"
> @printf "  make release VERSION=<v> PUSH=<true|false>\tChangelog, release commit and tag\n"

setup:
> $(UV) sync --all-groups
> $(UV) run prek install

fmt:
> $(UV) run ruff format .
> $(UV) run ruff check --fix .

lint:
> $(UV) run ruff check .
> $(UV) run ruff format --check .

typecheck:
> $(UV) run mypy

imports:
> $(UV) run lint-imports

test:
> $(UV) run pytest -n auto -m "not slow and not gpu and not live_api and not e2e"

test-all:
> $(UV) run pytest -n auto --live --gpu

cov:
> $(UV) run coverage run -m pytest -m "not slow and not gpu and not live_api"
> $(UV) run coverage combine
> $(UV) run coverage report

hooks:
> $(UV) run prek run --all-files

check: lint typecheck imports test

release:
> $(UV) version $(VERSION)
> $(UV) run git-changelog --bump $(VERSION) --convention conventional --output CHANGELOG.md .
> git add pyproject.toml uv.lock CHANGELOG.md
> git commit -m "chore(release): $(VERSION)"
> git tag $(VERSION)
ifeq ($(PUSH),true)
> git push
> git push --tags
endif
