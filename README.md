# rag-eval-platform

Retrieval-augmented generation over arXiv ML papers, built so that its **evaluation
harness is the deliverable** and the chat interface is only what makes it demoable. The
repository measures its own retrieval and answer quality on every commit and publishes
an ablation table — including the arms that lost.

**Status: Phase 4 — reranked hybrid retrieval.** Dense (bge-small over HNSW) and lexical
(`ts_rank_cd`) arms behind one `retrieve()` entry point, fused with Reciprocal Rank
Fusion and reordered by a `bge-reranker-base` cross-encoder, with refusal below a score
floor. Generation is next.

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
uv run rag search "What is the effect of learning rate warmup on transformer training?" --compare
```

`--compare` runs all three retrieval modes over the same query side by side. Any one of
them is reachable on its own with `--mode {rrf,dense_only,lexical_only}`.

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

## Lexical retrieval and fusion

**The lexical arm is `ts_rank_cd`, and it is not BM25.** `ts_rank_cd` is a
coverage-density ranking: it scores by how many query lexemes a chunk contains and how
tightly they cluster, with no document-frequency term and no length-saturation curve.
That is a different function from Okapi BM25, not an implementation of it. Phase 6 adds a
real BM25 arm with `bm25s` and the ablation table reports both.

Query lexemes are OR-ed, not conjoined. `plainto_tsquery` would require all six content
words of a typical question inside one 512-token chunk, which returns nothing on this
corpus; lexical search is here as the recall complement to an arm that already handles
semantic similarity.

Measured over 8 questions on the 6,386-chunk corpus, warm:

| mode | p50 ms | p95 ms |
|---|---:|---:|
| `dense_only` | 13.6 | 16.0 |
| `lexical_only` | 17.2 | 51.3 |
| `rrf` | 32.1 | 66.2 |

**The arms overlap on a mean of 11.4 of their top 50** (min 6, max 21) — so roughly
three-quarters of each list is unique to that arm. That is the measured case for fusing
them at all. It is *not* a quality claim: complementarity is necessary for fusion to help,
not sufficient, and whether it actually helps is a Recall@k question that only Phase 7 can
answer.

**Why lexical costs more than dense, and what it implies.** The OR query matches a mean of
43% of the corpus (min 21%, max 72%), because stemmed terms like `train`, `rate` and
`learn` appear nearly everywhere. The GIN index finds those matches in 0.4ms; the expense
is computing `ts_rank_cd` for all ~3,500 of them and then top-N sorting. Suppressing
common terms is precisely what an inverse-document-frequency weight does — which makes
this a *measured* reason to want the BM25 arm in Phase 6 rather than a theoretical one.

RRF's p50 is close to the sum of the two arms because they run sequentially. Running them
concurrently is a Phase 9 concern, not a correctness one.

## Cross-encoder reranking

A bi-encoder embeds query and passage independently — neither vector is computed knowing
the other exists. A cross-encoder concatenates them and attends across the pair, which is
why it reorders results a bi-encoder got wrong, and why it cannot be indexed: every pair
needs its own forward pass. That is the entire trade, and this is what it costs here.

| `rerank_top_n` | p50 ms | p95 ms | top-5 kept vs n=50 | mean rank movement |
|---:|---:|---:|---:|---:|
| 5 | 566 | 602 | 0.8 / 5 | 1.1 |
| 10 | 1,132 | 1,144 | 1.0 / 5 | 2.8 |
| 25 | 2,538 | 2,775 | 2.8 / 5 | 10.1 |
| **50** | **5,316** | **5,591** | 5.0 / 5 | 23.4 |

**Reranking is by far the most expensive stage in the system** — roughly 106ms per pair
on CPU, against 32ms for the *entire* hybrid retrieval that feeds it. And there is no
cheap cut: at `top_n=25` only 2.8 of the 5 final results survive.

It stays at 50 anyway. The number that should decide this is Recall@5 at each setting,
and no golden set exists yet; trimming it now would be tuning against the one axis that
happens to be easy to measure. Phase 7 sweeps it and the table will show what the latency
bought.

Apple's MPS backend runs the same model at 56ms/pair — 1.7× faster — and is deliberately
not used. The deployment target is Linux CPU, so a latency figure measured on an Apple GPU
would describe hardware this system never runs on, and mixing MPS locally with CPU in CI
puts cross-machine determinism at risk for no benefit to the published numbers.

### The reranker sees about 80% of each chunk

| | tokenizer | vocab | p50 tokens per chunk |
|---|---|---|---:|
| embedder | BERT WordPiece | 30,522 | 460 |
| reranker | XLM-RoBERTa | 250,002 | 511 |

`bge-reranker-base` is multilingual and tokenises English into **1.19× more tokens** than
`bge-small` does. Chunks sized to fill the embedder's 512-token window therefore overflow
the reranker's, and **48% of corpus chunks — 78% of the pairs in the query above — lose
their tail**.

This is the same class of failure the chunker was built to prevent, on a stage that did
not exist when the budget was set. It is reported per query (`truncated_pairs`) rather
than hidden. It is not fixed by guesswork: shrinking `chunk_tokens` to about 430 would fit
both windows, and whether that trade is worth making is a Recall@5 question for Phase 7,
where it becomes an ablation arm rather than a hunch.

### Refusal

Below `score_floor` the result comes back empty with `reason="below_score_floor"`, and
Phase 5 declines to call the LLM at all. Refusal is a measured behaviour, not an error
path — the golden set scores how often it was the right call.

Scores are raw logits (roughly [-11, +11], zero being neutral), not probabilities. A
sigmoid would put every score above zero and silently turn the default floor of 0.0 into
"never refuse". The consequence, stated rather than discovered: **an arm with reranking
disabled cannot refuse on score**, because cosine similarity and `ts_rank_cd` are on
scales no single floor value can serve.

## Layout

| Path | What lives here |
|---|---|
| `packages/rag/` | The library — ingestion, indexing, retrieval, generation. Imports nothing from `apps/`. |
| `apps/api/` | FastAPI. A transport layer with no retrieval logic in it. |
| `infra/migrations/` | Numbered, forward-only SQL. Never edited once applied. |
| `eval/` | Golden set, metrics, runner, ablation matrix. *(Phases 6–7)* |

The rule that matters: the evaluation harness calls the same `rag` code path the API
calls. `tests/test_layering.py` enforces the direction of that dependency, and
`rag.retrieve.retrieve()` is the single door — dispatch is on `RetrievalConfig.fusion`
alone, so an ablation arm is a different configuration and never a different code path.

[`ENGINEERING.md`](ENGINEERING.md) states the principles this repo is built under —
what gets measured, what counts as honest naming, and what is not allowed to drift.

## Phase status

- [x] **0** Scaffold — workspace, compose, migrations, health, CI
- [x] **1** Ingestion — arXiv fetch, PDF parse, section-aware chunking
- [x] **2** Dense retrieval — bge-small embeddings, HNSW
- [x] **3** Lexical + RRF fusion
- [x] **4** Cross-encoder reranking
- [ ] **5** Generation, citations, refusal
- [ ] **6** Golden set
- [ ] **7** Eval harness and ablation table
- [ ] **8** Frontend
- [ ] **9** Ship

The ablation table replaces this section once Phase 7 lands.
