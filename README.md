# MemoryForge 🔍

**Production-grade Agentic Conversational RAG API for Research Documents**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-compose-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-orange?style=flat-square)

> Built with LangChain · LangGraph · BM25 · LLMLingua · MCP Tools · LangSmith · FastAPI · Docker · Redis · Qdrant

---

## What is MemoryForge?

MemoryForge is an **agentic RAG API** that answers questions about research documents — not with a fixed retrieve-then-generate chain, but with a **LangGraph state machine** that decides retrieval strategy, compresses context, calls external tools when needed, and remembers conversation history across sessions.

Feed it PDFs. Ask questions. It remembers what you asked before.

---

## Architecture

```
User Query → FastAPI
               → LangGraph Agent
                     → load memory         (Redis session history)
                     → BM25 Retriever      (keyword + metadata filter)
                     → LLMLingua           (4–6× context compression)
                     → MCP Tools           (arxiv — when local index insufficient)
                     → LLM                 (GPT-4o-mini / any OpenAI-compatible)
               → Answer + Sources + save turn to memory
```

Async document ingestion — uploads never block the API:

```
POST /ingest → Redis Queue → Worker → BM25 Index (persisted to disk)
```

---

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | async, Pydantic validation |
| Agent Orchestration | LangGraph | stateful decision-making, not a fixed chain |
| Retrieval | BM25 (`rank_bm25`) + metadata filtering | keyword search + filter by year/author/topic |
| Context Compression | LLMLingua 2 | 4–6× token reduction before LLM call |
| Memory | Redis (dual-layer) | short-term MessagesState + long-term session store |
| External Tools | MCP (arxiv API) | live paper search when local index is insufficient |
| Vector Store | Qdrant (ready) | hybrid BM25 + dense retrieval at scale |
| Observability | LangSmith + OpenTelemetry | full agent trace — latency, tokens, retrieval quality |
| Deployment | Docker Compose | 4 services: app + worker + redis + qdrant |

---

## Quickstart

```bash
# clone and configure
git clone https://github.com/divyat2605/MemoryForge
cd MemoryForge

cp .env.example .env
# add OPENAI_API_KEY and LANGCHAIN_API_KEY to .env

# start all 4 services
docker compose up --build
```

API available at `http://localhost:8000`

> **Note on first run:** LLMLingua model (~400MB) downloads on first startup. An extractive fallback kicks in automatically while it downloads — no code change needed.

---

## API Endpoints

### `POST /ingest`
Upload a research PDF, TXT, or MD for async indexing.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@attention_is_all_you_need.pdf"

# returns: { "job_id": "abc123", "status": "queued" }
```

### `GET /ingest/status/{job_id}`
Check ingestion progress.

```bash
curl http://localhost:8000/ingest/status/abc123

# returns: { "status": "done", "chunks": 42, "doc": "attention..." }
```

### `POST /query`
Query the indexed documents. Pass `session_id` for multi-turn memory.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer attention mechanism?",
    "session_id": "my-session",
    "filters": { "year": "2017" },
    "top_k": 5
  }'
```

Follow-up queries with the same `session_id` retain full conversation context.

```bash
# agent remembers the previous turn
curl -X POST http://localhost:8000/query \
  -d '{"query": "How does that compare to RNN?", "session_id": "my-session"}'
```

### `GET /documents`
List all indexed documents and their metadata.

```bash
curl http://localhost:8000/documents
```

### `DELETE /memory/{session_id}`
Clear conversation history for a session.

```bash
curl -X DELETE http://localhost:8000/memory/my-session
```

---

## Memory Layer

MemoryForge uses a **dual-layer memory system** — Redis is already in `docker-compose`, so this is zero extra infra.

| Layer | Implementation | Scope |
|-------|---------------|-------|
| Short-term | LangGraph `MessagesState` | last 6 turns, in-state per request |
| Long-term | Redis, keyed by `session_id` | full history, 7-day TTL, survives restarts |

Redis serves **two purposes in one service**: async ingestion job queue + session memory store.

---

## Observability

Set these in `.env` — zero code changes needed:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```

LangSmith automatically traces every LangGraph node (retrieve → compress → generate), capturing LLM token usage, per-query latency, and retrieval quality across all agent steps.

---

## Scaling to 1M+ Documents

Current implementation uses in-memory BM25 (`rank_bm25`).

| Bottleneck | Current | At Scale |
|------------|---------|----------|
| Keyword search | `rank_bm25` (in-memory) | **Elasticsearch** — BM25 built-in, distributed, zero retrieval logic change |
| Vector search | single Qdrant container | **Qdrant cluster** — hybrid BM25 + dense ANN |
| API throughput | single FastAPI instance | `docker compose up --scale app=3` |
| Ingestion speed | Redis queue (already async) | increase worker replicas |

**Interview answer:** *"Current impl uses rank_bm25 — for 1M docs I'd swap to Elasticsearch which has BM25 built-in and handles distributed search natively. The retrieval interface doesn't change, just the backend. Qdrant is already in docker-compose for hybrid search when needed."*

---

## Project Structure

```
MemoryForge/
├── main.py              # FastAPI — /ingest /query /memory /documents
├── agent.py             # LangGraph state machine — retrieve → compress → generate
├── memory.py            # dual-layer memory — LangGraph MessagesState + Redis
├── retriever.py         # BM25 keyword search + metadata filtering
├── compressor.py        # LLMLingua context compression + extractive fallback
├── mcp_tools.py         # MCP tool definitions (arxiv search + paper fetch)
├── ingestor.py          # doc chunking + auto metadata extraction
├── queue_worker.py      # Redis async ingestion worker
├── monitoring.py        # LangSmith + OpenTelemetry tracing
├── Dockerfile
├── docker-compose.yml   # app + worker + redis + qdrant
├── requirements.txt
├── .env.example
├── LICENSE              # Apache 2.0
└── README.md
```

---

## Environment Variables

```env
# required
OPENAI_API_KEY=sk-...

# observability (optional but recommended)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=memoryforge

# redis
REDIS_URL=redis://redis:6379

# qdrant
QDRANT_URL=http://qdrant:6333
```

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

*Built by [Divya Tripathi](https://github.com/divyat2605)*