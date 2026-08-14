# rag-eval-platform

Retrieval-augmented generation over arXiv ML papers, built so that its **evaluation
harness is the deliverable** and the chat interface is only what makes it demoable. The
repository measures its own retrieval and answer quality on every commit and publishes
an ablation table — including the arms that lost.

**Status: Phase 0 — scaffold.** No retrieval logic yet. Infrastructure, configuration,
migrations and CI only.

---

## Quickstart

```bash
cp .env.example .env
make dev        # Postgres 16 + pgvector, and the API, both healthy
make migrate    # apply forward-only SQL migrations
curl -s localhost:8000/health | python3 -m json.tool
```

`make help` lists the rest.

## Layout

| Path | What lives here |
|---|---|
| `packages/rag/` | The library — ingestion, indexing, retrieval, generation. Imports nothing from `apps/`. |
| `apps/api/` | FastAPI. A transport layer with no retrieval logic in it. |
| `infra/migrations/` | Numbered, forward-only SQL. Never edited once applied. |
| `eval/` | Golden set, metrics, runner, ablation matrix. *(Phases 6–7)* |

The rule that matters: the evaluation harness calls the same `rag` code path the API
calls. `tests/test_layering.py` enforces the direction of that dependency.

[`ENGINEERING.md`](ENGINEERING.md) states the principles this repo is built under —
what gets measured, what counts as honest naming, and what is not allowed to drift.

## Phase status

- [x] **0** Scaffold — workspace, compose, migrations, health, CI
- [ ] **1** Ingestion — arXiv fetch, PDF parse, section-aware chunking
- [ ] **2** Dense retrieval — bge-small embeddings, HNSW
- [ ] **3** Lexical + RRF fusion
- [ ] **4** Cross-encoder reranking
- [ ] **5** Generation, citations, refusal
- [ ] **6** Golden set
- [ ] **7** Eval harness and ablation table
- [ ] **8** Frontend
- [ ] **9** Ship

The ablation table replaces this section once Phase 7 lands.
