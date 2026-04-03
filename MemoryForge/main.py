from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import redis.asyncio as aioredis
import json
import uuid

from ingestor import ingest_document
from agent import run_agent
from monitoring import get_tracer

app = FastAPI(title="QueryForge", description="Agentic RAG API for Research Documents", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = aioredis.from_url("redis://redis:6379", decode_responses=True)

@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()


# ---------- schemas ----------

class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    top_k: Optional[int] = 5
    session_id: Optional[str] = "default"  # NEW — memory key

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    trace_id: Optional[str] = None

class IngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str


# ---------- routes ----------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "QueryForge"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a research PDF/txt/md for async ingestion.
    Job is queued in Redis and processed by the worker.
    """
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, MD supported")

    contents = await file.read()
    job_id = str(uuid.uuid4())

    job_payload = {
        "job_id": job_id,
        "filename": file.filename,
        "content": contents.decode("utf-8", errors="ignore"),
    }

    await redis_client.rpush("ingest_queue", json.dumps(job_payload))
    await redis_client.set(f"job:{job_id}", "queued", ex=3600)

    return IngestResponse(job_id=job_id, status="queued", filename=file.filename)


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    status = await redis_client.get(f"job:{job_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": status}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run agentic RAG query: BM25 retrieval → metadata filtering → LLMLingua compression → LLM answer.
    """
    with get_tracer().start_as_current_span("queryforge.query") as span:
        span.set_attribute("query", request.query)
        span.set_attribute("filters", str(request.filters))

        result = await run_agent(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
            session_id=request.session_id,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            trace_id=span.get_span_context().trace_id.__str__() if span else None,
        )


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    from memory import clear_session
    await clear_session(session_id)
    return {"session_id": session_id, "status": "cleared"}


@app.get("/documents")
async def list_documents():
    """List all ingested documents and their metadata."""
    from retriever import list_all_docs
    docs = list_all_docs()
    return {"documents": docs, "total": len(docs)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)