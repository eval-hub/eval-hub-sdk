.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	uv run pytest

.PHONY: test-e2e
test-e2e:
	uv run pytest --e2e -s -x -q

.PHONY: ruff
ruff:
	uv run ruff check --fix src/evalhub
	uv run ruff format src tests

.PHONY: mypy
mypy:
	uv run mypy --config-file=pyproject.toml src tests

.PHONY: tidy
tidy: ruff mypy
