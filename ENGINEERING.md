# Engineering principles — rag-eval-platform

The contract this repository is built under. Every design decision below is meant to be
defensible out loud, not just implemented.

---

## What this project is

A retrieval-augmented generation platform over arXiv ML papers whose **primary
deliverable is the evaluation harness, not the chatbot**. The chat interface exists to
make the system demoable; the reason the repo exists is that it measures its own
retrieval and answer quality on every commit and publishes an ablation table.

If a change makes the demo prettier but the evaluation weaker or less honest, it is the
wrong change.

## Non-negotiable principles

1. **Every retrieval claim must be measured.** Never write "improves relevance" in a
   comment, commit message or README. Write the delta on a named metric against a named
   baseline, or say nothing.
2. **No answer without a citation.** The generator must emit claims bound to
   `chunk_id`s. If retrieval returns nothing above threshold, the correct behaviour is
   refusal, not a plausible guess.
3. **Determinism in the eval path.** Temperature 0, fixed seeds, pinned model versions,
   cached embeddings. Two runs of `make eval` on the same commit must produce identical
   retrieval metrics.
4. **The golden set is sacred.** Never tune a parameter and then regenerate the golden
   set. Never let the system under test influence the ground truth. Changes to
   `eval/golden/*.jsonl` require an explicit, separate commit that touches nothing else.
5. **Honest naming.** If we use Postgres `ts_rank_cd`, we call it lexical search and say
   so — we do not call it BM25. If the LLM judge is the same model family as the
   generator, the README says that and quantifies the bias.
6. **Cost and latency are features.** Every retrieval path records per-stage timings and
   token counts.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Matches existing skills |
| Package manager | `uv` | Fast, lockfile, workspace support |
| API | FastAPI + Pydantic v2 | Typed contracts end to end |
| Database | Postgres 16 + `pgvector` | One datastore for vectors, lexical and metadata |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) via sentence-transformers, local | Free, deterministic, CPU-viable, swappable for ablation |
| Reranker | `BAAI/bge-reranker-base` cross-encoder, local | Free, biggest single quality win |
| Generation | Anthropic API behind an `LLMClient` protocol | Cheap; the protocol keeps it swappable |
| Frontend | React 18 + TypeScript + Vite | Matches existing skills |
| Container | Docker + Docker Compose | Reproducible local stack |
| CI | GitHub Actions | Public, visible, badge-able |

## Repo layout

```
rag-eval-platform/
├── ENGINEERING.md
├── Makefile                      # dev, test, lint, ingest, eval, ablate
├── docker-compose.yml
├── pyproject.toml                # uv workspace root
├── packages/
│   └── rag/                      # the library — no FastAPI imports allowed in here
│       └── src/rag/
│           ├── ingest/           # arxiv fetch, pdf parse, section-aware chunking
│           ├── index/            # embedding, lexical index, migration runner
│           ├── retrieve/         # dense, lexical, rrf fusion, rerank
│           ├── generate/         # prompt assembly, citation binding, refusal
│           ├── config.py         # single typed RetrievalConfig — see below
│           ├── db.py             # connection handling
│           ├── settings.py       # the only place the library reads the environment
│           └── telemetry.py      # per-stage timings, token counts, cost
├── apps/
│   ├── api/                      # FastAPI: thin HTTP layer over packages/rag
│   └── web/                      # React + Vite
├── eval/
│   ├── golden/                   # versioned JSONL ground truth — treat as read-only
│   ├── generate_golden.py        # candidate generation (human verification required)
│   ├── verify_cli.py             # TUI for human verification pass
│   ├── metrics/                  # recall@k, mrr, ndcg, faithfulness, citation acc
│   ├── runner.py                 # runs one config against the golden set
│   ├── ablate.py                 # runs the full matrix, writes results/
│   └── results/                  # committed markdown + json, one file per run
├── infra/
│   ├── migrations/               # plain SQL, numbered, forward-only
│   ├── fly.toml
│   └── k6/                       # load test scripts
└── .github/workflows/
    ├── ci.yml                    # lint, typecheck, unit tests, smoke eval
    └── eval.yml                  # nightly full ablation, commits results
```

**Hard rule:** `packages/rag` must never import from `apps/`. The library is usable from
a script, a notebook, the API and the eval runner identically. The eval harness calls
exactly the same code path the API does — no parallel implementation.
`tests/test_layering.py` enforces the direction of that dependency.

## The central abstraction

Everything configurable about retrieval lives in one frozen Pydantic model. The ablation
runner works by instantiating variants of it. Nothing in the retrieval path may read an
environment variable directly.

```python
class RetrievalConfig(BaseModel, frozen=True):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_tokens: int = 512
    chunk_overlap_pct: float = 0.15
    contextual_headers: bool = True  # prepend paper+section title to chunk before embedding
    dense_enabled: bool = True
    dense_top_k: int = 50
    lexical_enabled: bool = True
    lexical_top_k: int = 50
    fusion: Literal["rrf", "dense_only", "lexical_only"] = "rrf"
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    final_top_k: int = 5
    score_floor: float = 0.0  # below this -> refuse
```

Every eval result JSON embeds the full config that produced it. A result without its
config is worthless.

## Conventions

- **Commits:** conventional commits, one logical change each. Never a single giant
  commit — the commit history is part of the record.
- **Types:** full annotations; `mypy --strict` on `packages/rag`. `ruff` for lint and
  format.
- **Tests:** `pytest`. Pure functions (chunking boundaries, RRF math, metric
  calculations) get unit tests with hand-computed expected values. Retrieval gets
  integration tests against a seeded 20-paper fixture corpus.
- **Migrations:** forward-only numbered SQL in `infra/migrations`. No ORM
  auto-migration, and an applied migration is never edited — the runner enforces this
  with a checksum guard.
- **Secrets:** `.env` only, `.env.example` committed, never a key in code or test
  fixtures.
- **Docstrings:** every public function in `packages/rag` gets one explaining *why*, not
  what.
