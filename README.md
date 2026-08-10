# minirag

A compact retrieval-augmented generation (RAG) API: ingest documents, search them
semantically, and get answers grounded in the passages that were actually retrieved.

Built with **FastAPI**, **Qdrant** and **Pydantic**, with **89% test coverage** and CI on
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

```
                 ┌──────────────┐
   POST /documents │   chunking   │  word windows with overlap
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  embeddings  │  Gemini  ·  offline fake
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │    Qdrant    │  cosine similarity
                 └──────┬───────┘
                        │
   POST /ask    ┌───────▼──────┐
   POST /search │  generation  │  Anthropic  ·  offline fake
                 └──────────────┘
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness plus active providers and indexed chunk count |
| `POST /documents` | Chunk, embed and index a document |
| `POST /search` | Semantic search over indexed chunks |
| `POST /ask` | Retrieve passages and answer from them, with citations |

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

- No authentication; add an API key or JWT layer before exposing it publicly.
- No reranking stage — retrieval is pure vector similarity. A cross-encoder
  reranker would improve precision on larger corpora.
- Ingestion is synchronous. Large documents should move to a background worker.
- No hybrid (keyword + vector) search, which typically helps with rare terms
  and exact identifiers.

## Licence

MIT — see [LICENSE](LICENSE).
