.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	uv run pytest --color=yes -ra

.PHONY: test-e2e
test-e2e:
	@echo "*** WARN: Running E2E with uv run --no-sync so not to override any replacement for eval-hub-server ***"
	uv run --no-sync uv pip show eval-hub-server
	uv run --no-sync pytest --e2e -s -x --color=yes -ra

.PHONY: ruff
ruff:
	uv run ruff check --fix src/evalhub
	uv run ruff format src tests

.PHONY: mypy
mypy:
	uv run mypy --config-file=pyproject.toml src tests

.PHONY: tidy
tidy: ruff mypy

.PHONY: start-oci-registry
start-oci-registry:
	docker run -d -p 5001:5000 --name eval-hub-oci-registry docker.io/library/registry:2

.PHONY: stop-oci-registry
stop-oci-registry:
	docker stop eval-hub-oci-registry
	docker rm eval-hub-oci-registry
