# rag-eval-platform

Retrieval-augmented generation over arXiv ML papers, built so that its **evaluation
harness is the deliverable** and the chat interface is only what makes it demoable. The
repository measures its own retrieval and answer quality on every commit and publishes
an ablation table — including the arms that lost.

**Status: Phase 2 — dense retrieval.** Local bge-small embeddings with a content-addressed
cache, an HNSW index measured against exact-scan ground truth, and cosine top-k search.
Lexical search, fusion and reranking are next.

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

Then embed and search (~6 minutes on CPU for 6,386 chunks; re-running is a no-op):

```bash
uv run rag embed
```

```bash
uv run rag search "What is the effect of learning rate warmup on transformer training?"
```

The corpus is pinned in [`infra/corpus/`](infra/corpus/) — rebuild the exact same one with
`rag ingest --ids-file infra/corpus/cs-lg-cs-cl-150.txt`. A category search returns "the
most recent 150", which is a different set of papers every day; the golden set will bind
questions to chunk ids, so the corpus behind them has to be nameable.

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

## Dense retrieval

Embeddings are `BAAI/bge-small-en-v1.5` run locally on CPU. Queries get bge's instruction
prefix and passages do not — the asymmetry is worth several points of recall and is
invisible when wrong, so a test asserts it rather than a comment claiming it.

`ef_search` is chosen from measurement, not from the pgvector default. Against exact-scan
ground truth over 50 queries on the 6,386-chunk corpus (`rag bench-index`):

| ef_search | recall@10 | recall@50 | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|
| 40 | 0.830 | 0.860 | 6.9 | 11.6 |
| 100 | 0.952 | 0.921 | 5.1 | 8.2 |
| 200 | 0.980 | 0.973 | 6.7 | 12.4 |
| **400** | **1.000** | **1.000** | **7.1** | **8.2** |
| 800 | 1.000 | 1.000 | 7.4 | 8.0 |

Index: 12.5 MiB over 6,386 vectors, 0.5s to build. Warm end-to-end search (query
embedding + retrieval, model resident) is **p50 33ms / p95 38ms**.

Two honest notes on that table:

- **The index barely earns its place at this scale.** An exhaustive scan of the same
  corpus runs at p50 8.0ms — HNSW at ef_search=400 is 7.1ms. The index is here because it
  is the thing that keeps working as the corpus grows, not because it is winning today.
- **400 is tuned to *this* corpus.** It buys exact agreement with a full scan for about
  2ms, which matters because approximation error is indistinguishable from retrieval error
  in a published metric: at ef_search=100 roughly one true neighbour in twenty is missed,
  and every ablation arm would silently carry that deficit. It will not stay exact at ten
  times the rows and must be re-measured.

Retrieval quality itself is not claimed here. There is no golden set yet, so there is no
Recall@k against ground truth — only Phase 6 and 7 can supply that, and until they do the
right number of quality claims to make is zero.

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
- [x] **2** Dense retrieval — bge-small embeddings, HNSW
- [ ] **3** Lexical + RRF fusion
- [ ] **4** Cross-encoder reranking
- [ ] **5** Generation, citations, refusal
- [ ] **6** Golden set
- [ ] **7** Eval harness and ablation table
- [ ] **8** Frontend
- [ ] **9** Ship

The ablation table replaces this section once Phase 7 lands.
