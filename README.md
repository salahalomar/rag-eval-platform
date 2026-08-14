# rag-eval-platform

Retrieval-augmented generation over arXiv ML papers, built so that its **evaluation
harness is the deliverable** and the chat interface is only what makes it demoable. The
repository measures its own retrieval and answer quality on every commit and publishes
an ablation table — including the arms that lost.

**Status: Phase 1 — ingestion.** arXiv fetch, PDF parsing, section detection and
structure-aware chunking. No retrieval yet.

---

## Quickstart

```bash
cp .env.example .env
make dev        # Postgres 16 + pgvector, and the API, both healthy
make migrate    # apply forward-only SQL migrations
curl -s localhost:8000/health | python3 -m json.tool
```

Then build the corpus (~10 minutes: arXiv asks for one request every three seconds, and
PDFs are cached on disk so a re-run resumes rather than restarts):

```bash
uv run rag ingest --limit 150
```

```bash
uv run rag stats --sample 3
```

`make help` lists the rest.

## How ingestion works

| Stage | What it does | Why it is not the obvious thing |
|---|---|---|
| `arxiv.py` | Metadata + PDF, cached by id | One request per 3s, resumable — a corpus build is paid once, offline |
| `parse.py` | Text, page numbers, char offsets | Reconstructs **two-column reading order**; PyMuPDF's default block order interleaves the columns into alternating half-sentences |
| `sections.py` | `3 Method > 3.2 Training` | Requires **both** typography and numbering; either signal alone produces false headings, and a false boundary permanently severs a table from its caption |
| `chunk.py` | 512 tokens, 15% overlap | Never crosses a section, never splits a sentence, and counts tokens with **bge-small's own tokenizer** so chunks cannot silently overflow the model's window |

Two behaviours worth knowing about:

- **`chunk_tokens` budgets `embed_input`, not content.** With contextual headers on, the
  header is charged against the same budget, so headers cost ~6% of content at the
  default chunk size. The alternative — budgeting content alone — pushes the embedded
  string past the model's 512-token window and truncates every long chunk with no error
  raised. The Phase 7 headers arm should be run below the model window to isolate it.
- **Chunkings are keyed, not overwritten.** `chunk_config_sha256` identifies the settings
  that produced a chunk, so the Phase 7 chunk-size sweep can hold several chunkings of
  one corpus side by side. Re-ingesting is a no-op only when the PDF *and* the chunking
  config are both unchanged.

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
- [x] **1** Ingestion — arXiv fetch, PDF parse, section-aware chunking
- [ ] **2** Dense retrieval — bge-small embeddings, HNSW
- [ ] **3** Lexical + RRF fusion
- [ ] **4** Cross-encoder reranking
- [ ] **5** Generation, citations, refusal
- [ ] **6** Golden set
- [ ] **7** Eval harness and ablation table
- [ ] **8** Frontend
- [ ] **9** Ship

The ablation table replaces this section once Phase 7 lands.
