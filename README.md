# minirag

A compact retrieval-augmented generation (RAG) API: ingest documents, search them
with **hybrid retrieval**, rerank the candidates, and get answers grounded in the
passages that were actually retrieved.

Built with **FastAPI**, **Qdrant** and **Pydantic**, with **96% test coverage** and CI on
every push.

> **Clone and run it in 30 seconds — no API keys, no database, no signup.**
> External providers sit behind interfaces with deterministic offline
> implementations, so the full pipeline runs anywhere.

## Quickstart

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for the interactive OpenAPI docs, or:

```bash
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"title":"Qdrant","text":"Qdrant is an open source vector database for dense embeddings."}'

curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"query":"What is Qdrant?"}'
```

With Docker, including a real Qdrant instance:

```bash
docker compose up --build
```

## How it works

Ingestion writes to two indexes; retrieval reads from both and narrows in stages.

```
  POST /documents
        │
        ▼
    chunking ──────────► embeddings ──────────► Qdrant      (dense index)
   word windows          Gemini · offline        cosine
   with overlap                                     │
        │                                           │
        └──────────────────────────────────────► BM25       (lexical index)
                                                    │
  ────────────────────────────────────────────────────────────────────────
  POST /search · POST /ask
        │
        ├─► dense search   ─┐
        │   top_k × 4       │
        │                   ├─►  RRF fusion  ─►  rerank  ─►  top_k passages
        │                   │    merge ranks     cross-        │
        └─► BM25 search    ─┘                    encoder       ▼
            top_k × 4                                      generation
                                                    Anthropic · offline
```

**Recall first, then precision.** Each retriever returns a pool several times
larger than the answer size, because nothing downstream can recover a passage
that retrieval never returned. Fusion merges the two rankings; reranking orders
what survived.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, active providers, retrieval mode, both index sizes |
| `POST /documents` | Chunk, embed and index a document in both indexes |
| `POST /search` | Hybrid search over indexed chunks (`mode` overridable per call) |
| `POST /ask` | Retrieve passages and answer from them, with citations |

Any request may override the strategy, which makes the three modes directly
comparable on the same corpus:

```bash
curl -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"ES-2024-01847","mode":"dense"}'   # misses the exact reference
```

## Design decisions

**Providers are interfaces, not imports.** `EmbeddingProvider` and
`GenerationProvider` are Protocols. Concrete vendors are selected in a single
composition root (`app/dependencies.py`), so business logic never imports a
vendor SDK. Swapping Gemini for another embedder is a config change.

**The offline stack is a feature, not a mock.** `FakeEmbeddings` produces
deterministic, L2-normalised vectors from a hash of the input. It is
semantically meaningless but structurally faithful, which means the retrieval
pipeline is exercised end to end in CI with no secrets. Reviewers can run the
project without being asked for a credit card.

**Asymmetric embeddings.** Documents and queries are embedded with different
task types (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`), which improves retrieval
over embedding both sides identically.

**Hybrid retrieval, because dense search fails predictably.** Vector search
matches meaning, so it is weakest where meaning is not the point: product codes,
error identifiers, version numbers, surnames. A query for `ES-2024-01847`
retrieves passages about reference numbers in general. BM25 covers exactly that
gap by scoring literal term overlap and weighting rare terms heavily. Neither
retriever subsumes the other, which is the argument for running both.

**Fusion on rank, not on score.** Cosine similarity is bounded in [-1, 1]; BM25
is unbounded and scales with corpus statistics. Normalising one against the other
needs distributional assumptions that do not hold, so Reciprocal Rank Fusion
discards the scores and combines positions instead: each hit accumulates
`1 / (k + rank)` per list. Less information, no assumptions — a trade made
deliberately.

**Reranking is a separate stage from retrieval.** A bi-encoder embeds query and
passage independently, which is what makes the index precomputable and also what
caps its precision. A cross-encoder scores the pair together: much more accurate,
far too slow for a whole corpus. So retrieval casts a wide cheap net and
reranking reorders only the survivors. The default reranker is a lexical
baseline that runs offline; a real cross-encoder (Cohere) sits behind the same
Protocol and is a config change.

**Fusion alone is not enough, and there is a test that says so.** Passages found
by both retrievers accumulate two contributions and outrank a passage found by
BM25 alone — so fusion gets an exact-match passage into the candidate pool but
not to the top of it. Reranking is what closes that gap. The behaviour is pinned
down in `tests/test_hybrid.py` rather than left as folklore.

**Chunking is deliberately simple and heavily tested.** Poor chunking is the
most common cause of poor RAG quality, so it is dependency-free, deterministic,
and covered by tests for overlap correctness, content preservation and invalid
parameters.

**Grounding is enforced in the prompt.** The model is instructed to answer only
from numbered context passages, to cite them, and to decline when the context is
insufficient — a confidently wrong answer is worse than an admitted gap.

## Configuration

Copy `.env.example` to `.env`. Everything defaults to the offline stack; set
`EMBEDDING_PROVIDER=gemini` or `GENERATION_PROVIDER=anthropic` with the matching
API key to use real providers.

## Development

```bash
make test    # pytest with coverage
make lint    # ruff
make run     # dev server with reload
```

## Limitations

Honest scope notes, since this is a reference implementation rather than a
production service:

- **The BM25 index is in-process and in-memory.** It is rebuilt from Qdrant at
  startup (`warm_lexical_index`), so a restart does not silently degrade hybrid
  search to dense-only — but it does not survive as a shared index across
  workers, and rebuild time grows with the corpus. The production route is
  Qdrant's native sparse vectors, which keeps one source of truth and removes
  the rebuild entirely.
- **The default reranker is a baseline, not a cross-encoder.** `LexicalReranker`
  scores query-term coverage and knows no synonyms and no word order. It exists
  so the default configuration reranks with something honest offline. Set
  `RERANKER_PROVIDER=cohere` where reranking quality matters.
- **No retrieval evaluation.** There is no labelled query set, so the claim that
  hybrid beats dense rests on a constructed test case rather than on measured
  recall@k or MRR. Adding that harness is the next thing worth doing.
- No stemming or stopword handling in the lexical tokenizer — deliberate, since
  either would need a language assumption the rest of the pipeline avoids.
- No authentication; add an API key or JWT layer before exposing it publicly.
- Ingestion is synchronous. Large documents should move to a background worker.

## Licence

MIT — see [LICENSE](LICENSE).
