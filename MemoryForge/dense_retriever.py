"""
dense_retriever.py — Dense embedding retrieval and hybrid search for MemoryForge

- Uses SentenceTransformers for dense vector search
- Supports hybrid retrieval (BM25 + dense)
- Reranking with cross-encoder (bge-reranker or Cohere)
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util

# For reranking
try:
    from sentence_transformers import CrossEncoder
    _has_cross_encoder = True
except ImportError:
    _has_cross_encoder = False

# --- Embedding Model ---
_dense_model = None
_dense_index = []  # List[dict]: {"id", "text", "metadata", "embedding"}


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _dense_model


def build_dense_index(docs: List[dict]):
    """Build dense index from docs (id, text, metadata)."""
    model = get_dense_model()
    for doc in docs:
        if "embedding" not in doc:
            doc["embedding"] = model.encode(doc["text"], normalize_embeddings=True)
    global _dense_index
    _dense_index = docs


def dense_retrieve(query: str, top_k: int = 5, filters: Optional[dict] = None) -> List[dict]:
    """Dense embedding retrieval with optional metadata filtering."""
    model = get_dense_model()
    q_emb = model.encode(query, normalize_embeddings=True)
    # Metadata filter
    candidates = [d for d in _dense_index if _meta_match(d["metadata"], filters)] if filters else _dense_index
    if not candidates:
        return []
    scores = [float(np.dot(q_emb, d["embedding"])) for d in candidates]
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{"id": d["id"], "text": d["text"], "metadata": d["metadata"], "score": round(s, 4)} for s, d in ranked[:top_k]]


def _meta_match(meta: dict, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        val = meta.get(k)
        if isinstance(val, list):
            if v not in val:
                return False
        else:
            if str(val).lower() != str(v).lower():
                return False
    return True


def hybrid_retrieve(query: str, top_k: int = 5, filters: Optional[dict] = None, bm25_fn=None) -> List[dict]:
    """Combine BM25 and dense retrieval, deduplicate, rerank."""
    bm25_results = bm25_fn(query, top_k=top_k, filters=filters) if bm25_fn else []
    dense_results = dense_retrieve(query, top_k=top_k, filters=filters)
    # Merge by id, keep best score
    merged = {}
    for r in bm25_results + dense_results:
        if r["id"] not in merged or r["score"] > merged[r["id"]]["score"]:
            merged[r["id"]] = r
    merged_list = list(merged.values())
    # Rerank if cross-encoder available
    if _has_cross_encoder and len(merged_list) > 1:
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [(query, d["text"]) for d in merged_list]
        ce_scores = reranker.predict(pairs)
        for d, s in zip(merged_list, ce_scores):
            d["rerank_score"] = float(s)
        merged_list.sort(key=lambda x: x["rerank_score"], reverse=True)
    return merged_list[:top_k]
