SHELL := /bin/bash
COMPOSE := docker compose
UV := uv
MIGRATIONS_DIR := infra/migrations

.DEFAULT_GOAL := help
.PHONY: help dev down migrate migrate-status lint fmt typecheck test check logs psql clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- stack ------------------------------------------------------------------

dev: ## Build and start Postgres + API, waiting until both report healthy
	$(COMPOSE) up -d --build --wait
	@echo "api:      http://localhost:$${API_PORT:-8000}/health"
	@echo "postgres: localhost:$${POSTGRES_PORT:-5432}"
	@echo "next:     make migrate"

down: ## Stop the stack (keeps the pgdata volume)
	$(COMPOSE) down

logs: ## Follow API logs
	$(COMPOSE) logs -f api

psql: ## Open a psql shell against the running database
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-rag} -d $${POSTGRES_DB:-rag}

clean: ## Stop the stack and destroy the database volume
	$(COMPOSE) down -v

# --- database ---------------------------------------------------------------

migrate: ## Apply pending SQL migrations (forward-only)
	$(UV) run python -m rag.index.migrate --dir $(MIGRATIONS_DIR)

migrate-status: ## List applied and pending migrations without applying anything
	$(UV) run python -m rag.index.migrate --dir $(MIGRATIONS_DIR) --status

# --- quality ----------------------------------------------------------------

lint: ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Autofix lint findings and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy --strict over packages/rag (and apps/api)
	$(UV) run mypy

test: ## Run the test suite (integration tests need `make dev` first)
	$(UV) run pytest

check: lint typecheck test ## Everything CI runs
