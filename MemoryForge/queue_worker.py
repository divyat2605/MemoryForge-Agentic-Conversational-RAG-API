"""
queue_worker.py — Redis queue worker for async document ingestion

Run alongside the FastAPI app:
    python queue_worker.py

Picks jobs from the "ingest_queue" Redis list and processes them
via the ingestion pipeline.
"""

import asyncio
import json
import os
import redis.asyncio as aioredis
from ingestor import ingest_document


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QUEUE_KEY = "ingest_queue"
POLL_INTERVAL = 1  # seconds


async def process_job(redis_client, raw_job: str):
    job = json.loads(raw_job)
    job_id = job["job_id"]
    filename = job["filename"]
    content = job["content"]

    print(f"[worker] processing job {job_id} — {filename}")

    try:
        await redis_client.set(f"job:{job_id}", "processing", ex=3600)
        result = await ingest_document(filename, content)
        await redis_client.set(f"job:{job_id}", "completed", ex=3600)
        print(f"[worker] ✅ {filename} — {result['chunks']} chunks indexed")
    except Exception as e:
        await redis_client.set(f"job:{job_id}", f"failed: {str(e)}", ex=3600)
        print(f"[worker] ❌ job {job_id} failed: {e}")


async def main():
    print(f"[worker] connecting to Redis at {REDIS_URL}")
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    print(f"[worker] listening on queue: {QUEUE_KEY}")
    while True:
        try:
            # blocking pop with 2s timeout
            result = await redis_client.blpop(QUEUE_KEY, timeout=2)
            if result:
                _, raw_job = result
                await process_job(redis_client, raw_job)
        except Exception as e:
            print(f"[worker] error: {e}")
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())