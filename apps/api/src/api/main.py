"""HTTP surface. Phase 0 exposes health only; /query arrives in Phase 5."""

from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from rag import __version__ as rag_version
from rag.db import DatabaseHealth, check_health

app = FastAPI(
    title="rag-eval-platform",
    version=rag_version,
    description="RAG over arXiv ML papers. The evaluation harness is the product.",
)


class HealthResponse(BaseModel):
    """Payload for GET /health."""

    status: Literal["ok", "degraded"]
    db: DatabaseHealth
    version: str


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
def health(response: Response) -> HealthResponse:
    """Report process and database state.

    Defined with `def` rather than `async def` on purpose: FastAPI runs synchronous
    endpoints in a threadpool, which lets the whole library stay synchronous and keeps
    the API and the eval runner on one code path.

    `db.pgvector_version` is null until migration 001 has run, so this endpoint
    distinguishes "database reachable" from "stack actually ready" — useful when
    `make dev` has completed but `make migrate` has not.
    """
    db = check_health()
    if not db.connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", db=db, version=rag_version)
    return HealthResponse(status="ok", db=db, version=rag_version)
