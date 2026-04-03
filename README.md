# QueryForge 🔍

**Production-grade Agentic RAG API for Research Documents**

> Built with LangChain · LangGraph · BM25 · LLMLingua · MCP Tools · LangSmith · FastAPI · Docker

---

## Architecture

```
User Query → FastAPI
               → LangGraph Agent
                     → BM25 Retriever (keyword search + metadata filtering)
                     → LLMLingua (context compression ~4x token reduction)
                     → MCP Tools (arxiv search, paper fetch — on demand)
                     → LLM (GPT-4o-mini / any OpenAI-compatible)
               → Answer + Sources
```

Async document ingestion via Redis queue — uploads never block the API.

```
POST /ingest → Redis Queue → Worker → BM25 Index (persisted)
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Agent Orchestration | LangGraph |
| Retrieval | BM25 (rank_bm25) + metadata filtering |
| Context Compression | LLMLingua 2 |
| External Tools | MCP tools (arxiv API) |
| Async Queue | Redis |
| Vector Store (ready) | Qdrant |
| Monitoring | LangSmith + OpenTelemetry |
| Deployment | Docker + docker-compose |

---

## Quickstart

```bash
git clone https://github.com/yourusername/QueryForge
cd QueryForge

cp .env.example .env
# add your OPENAI_API_KEY and LANGCHAIN_API_KEY

docker compose up --build
```

API available at `http://localhost:8000`

---

## API Endpoints

### `POST /ingest`
Upload a research PDF/TXT/MD for async indexing.
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@attention_is_all_you_need.pdf"
# returns job_id
```

### `GET /ingest/status/{job_id}`
Check ingestion status.
```bash
curl http://localhost:8000/ingest/status/<job_id>
```

### `POST /query`
Query the indexed documents.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer attention mechanism?",
    "filters": {"year": "2017"},
    "top_k": 5
  }'
```

### `GET /documents`
List all indexed documents and metadata.

---

## Scaling to 1M Documents

Current implementation uses in-memory BM25 (`rank_bm25`).

To scale:
1. Replace `rank_bm25` with **Elasticsearch** — BM25 built-in, distributed, zero retrieval logic change needed
2. Enable **Qdrant** (already in docker-compose) for dense vector search alongside BM25 (hybrid search)
3. Add horizontal FastAPI replicas via `docker compose up --scale app=3`

---

## Monitoring

Set `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true` in `.env`.

LangSmith automatically traces:
- Every LangGraph node (retrieve → compress → generate)
- LLM token usage and latency
- Retrieval quality per query

---

## Project Structure

```
QueryForge/
├── main.py           # FastAPI routes
├── agent.py          # LangGraph graph definition
├── retriever.py      # BM25 + metadata filtering
├── compressor.py     # LLMLingua context compression
├── mcp_tools.py      # MCP tool definitions (arxiv)
├── ingestor.py       # Doc chunking + metadata extraction
├── queue_worker.py   # Redis async ingestion worker
├── monitoring.py     # LangSmith + OpenTelemetry
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Resume Talking Points

- **Agentic RAG** using LangGraph state machine with tool-use capability
- **BM25 keyword retrieval** with metadata-aware pre-filtering (year, author, topics)
- **LLMLingua context compression** — 4-6x token reduction before LLM generation
- **MCP tool integration** — agent can call external APIs (arxiv) when local index is insufficient
- **Async ingestion pipeline** via Redis queue — non-blocking uploads at scale
- **Production observability** via LangSmith tracing across all agent nodes
- **Dockerized** multi-service deployment (app + worker + redis + qdrant)
- **Scale path defined**: BM25 → Elasticsearch, Qdrant for hybrid search at 1M+ docs