"""
memory_summarizer.py — Summarization and semantic memory for MemoryForge

- Summarizes long conversations for context windowing
- Embedding-based semantic memory retrieval
- Memory importance scoring (recency, LLM, user feedback)
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util

# Use a lightweight LLM for summarization (can swap for OpenAI, etc.)
def summarize_history(history: List[Dict], max_tokens: int = 256) -> str:
    """
    Summarize a list of chat turns (dicts with 'role' and 'content').
    """
    # Simple extractive: join last N turns, or use LLM for abstractive
    if len(history) <= 4:
        return "\n".join([f"{h['role']}: {h['content']}" for h in history])
    # Placeholder: join last 2, summarize earlier
    head = history[:-2]
    tail = history[-2:]
    # For now, just join head as a block
    summary = "... " + " ".join([h['content'] for h in head])[:max_tokens]
    return summary + "\n" + "\n".join([f"{h['role']}: {h['content']}" for h in tail])

# --- Semantic memory retrieval ---
_memory_model = None
_memory_index = []  # List[dict]: {"id", "content", "embedding", "importance"}

def get_memory_model():
    global _memory_model
    if _memory_model is None:
        _memory_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _memory_model

def build_memory_index(turns: List[Dict]):
    """Build semantic memory index from chat turns."""
    model = get_memory_model()
    for t in turns:
        if "embedding" not in t:
            t["embedding"] = model.encode(t["content"], normalize_embeddings=True)
    global _memory_index
    _memory_index = turns

def semantic_memory_retrieve(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve most relevant past turns to query."""
    model = get_memory_model()
    q_emb = model.encode(query, normalize_embeddings=True)
    scores = [float(util.dot_score(q_emb, t["embedding"])) for t in _memory_index]
    ranked = sorted(zip(scores, _memory_index), key=lambda x: x[0], reverse=True)
    return [t for s, t in ranked[:top_k]]

# --- Importance scoring ---
def score_importance(turn: Dict, query: Optional[str] = None) -> float:
    """
    Score memory turn importance (recency + relevance).
    """
    recency = 1.0 / (1 + turn.get("age", 0))
    relevance = 1.0
    if query and "embedding" in turn:
        model = get_memory_model()
        q_emb = model.encode(query, normalize_embeddings=True)
        relevance = float(util.dot_score(q_emb, turn["embedding"]))
    return 0.7 * recency + 0.3 * relevance
