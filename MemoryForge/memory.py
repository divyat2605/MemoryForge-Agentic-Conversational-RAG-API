
"""
memory.py — Dual-layer memory for MemoryForge

Short-term : last N conversation turns (in-process, per session)
Long-term  : Redis-persisted message history keyed by session_id
"""

import json
import os
from typing import Optional
import redis.asyncio as aioredis
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MAX_SHORT_TERM_TURNS = 6          # last 6 turns kept in context
LONG_TERM_TTL = 60 * 60 * 24 * 7  # 7 days TTL per session

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


# ---------- long-term (Redis) ----------

async def load_history(session_id: str) -> list[BaseMessage]:
    """Load full conversation history for a session from Redis."""
    r = await get_redis()
    raw = await r.get(f"memory:{session_id}")
    if not raw:
        return []

    messages = []
    for item in json.loads(raw):
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


async def save_history(session_id: str, messages: list[BaseMessage]):
    """Persist conversation history to Redis with TTL."""
    r = await get_redis()
    serialized = [
        {
            "role": "human" if isinstance(m, HumanMessage) else "ai",
            "content": m.content,
        }
        for m in messages
    ]
    await r.set(f"memory:{session_id}", json.dumps(serialized), ex=LONG_TERM_TTL)


async def append_turn(session_id: str, query: str, answer: str):
    """Append a single Q&A turn to session history."""
    history = await load_history(session_id)
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=answer))
    await save_history(session_id, history)


async def clear_session(session_id: str):
    """Clear memory for a session."""
    r = await get_redis()
    await r.delete(f"memory:{session_id}")


# ---------- short-term (windowed) ----------

def get_short_term(history: list[BaseMessage], n_turns: int = MAX_SHORT_TERM_TURNS) -> list[BaseMessage]:
    """
    Return last N turns from history for LLM context window.
    Each turn = 1 human + 1 AI message = 2 items, so slice by n_turns * 2.
    """
    return history[-(n_turns * 2):]