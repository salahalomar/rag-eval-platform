# Build context is the repo root, not apps/api — the API depends on the `rag`
# workspace member, so the whole workspace has to be visible to uv.
#
#   docker build -f apps/api/Dockerfile .

FROM python:3.12-slim-bookworm

# Pinned uv, copied in rather than pip-installed so the version is visible and fixed.
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependency layer first: only the manifests are copied, so editing source does not
# invalidate the (slow) dependency install. This matters more from Phase 2, when torch
# joins the tree.
COPY pyproject.toml uv.lock ./
COPY packages/rag/pyproject.toml packages/rag/pyproject.toml
COPY apps/api/pyproject.toml apps/api/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Then the workspace itself.
COPY packages/ packages/
COPY apps/ apps/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

# docker-compose overrides this with --reload for local development.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
