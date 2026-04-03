"""
retriever.py — BM25 keyword retrieval + metadata filtering

Scale note: rank_bm25 is in-memory. For 1M+ docs, swap BM25Okapi
for Elasticsearch which has BM25 built-in and handles distributed
search natively — zero change needed to the metadata filtering logic.
"""


import pickle
import os
from typing import List, Optional
from rank_bm25 import BM25Okapi
import threading
from dense_retriever import build_dense_index, hybrid_retrieve

# ---------- in-memory store ----------
_lock = threading.Lock()
_documents: list[dict] = []   # {"id", "text", "metadata"}
_bm25_index: Optional[BM25Okapi] = None
_INDEX_PATH = "/app/data/bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rebuild_index():
    global _bm25_index
    if not _documents:
        _bm25_index = None
        return
    corpus = [_tokenize(doc["text"]) for doc in _documents]
    _bm25_index = BM25Okapi(corpus)


def _save_index():
    os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
    with open(_INDEX_PATH, "wb") as f:
        pickle.dump({"documents": _documents}, f)


def _load_index():
    global _documents
    if os.path.exists(_INDEX_PATH):
        with open(_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            _documents = data.get("documents", [])
        _rebuild_index()


# load on import
_load_index()


# ---------- public API ----------

def add_documents_to_index(docs: list[dict]):
    """Add new docs to BM25 and dense index. Thread-safe."""
    with _lock:
        existing_ids = {d["id"] for d in _documents}
        new_docs = [d for d in docs if d["id"] not in existing_ids]
        _documents.extend(new_docs)
        _rebuild_index()
        _save_index()
        # Also update dense index
        build_dense_index(_documents)


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
    hybrid: bool = True,
) -> list[dict]:
    """
    Hybrid retrieval (BM25 + dense) with reranking. Set hybrid=False for BM25 only.
    """
    with _lock:
        if not _documents or _bm25_index is None:
            return []
        if hybrid:
            return hybrid_retrieve(query, top_k=top_k, filters=filters, bm25_fn=lambda q, top_k, filters: retrieve(q, top_k, filters, hybrid=False))
        # --- BM25 only ---
        # ...existing code for BM25 retrieval...
        if filters:
            candidate_docs = []
            candidate_indices = []
            for i, doc in enumerate(_documents):
                meta = doc.get("metadata", {})
                match = True
                for key, val in filters.items():
                    doc_val = meta.get(key)
                    if doc_val is None:
                        match = False
                        break
                    if isinstance(doc_val, list):
                        if val not in doc_val:
                            match = False
                            break
                    else:
                        if str(doc_val).lower() != str(val).lower():
                            match = False
                            break
                if match:
                    candidate_docs.append(doc)
                    candidate_indices.append(i)
        else:
            candidate_docs = _documents
            candidate_indices = list(range(len(_documents)))

        if not candidate_docs:
            return []

        tokenized_query = _tokenize(query)
        candidate_corpus = [_tokenize(doc["text"]) for doc in candidate_docs]
        local_bm25 = BM25Okapi(candidate_corpus)
        scores = local_bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in ranked[:top_k]:
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(float(score), 4),
            })

        return results


def list_all_docs() -> list[dict]:
    """Return unique documents (by source) with metadata."""
    with _lock:
        seen = set()
        unique = []
        for doc in _documents:
            src = doc["metadata"].get("source", doc["id"])
            if src not in seen:
                seen.add(src)
                unique.append({
                    "source": src,
                    "metadata": doc["metadata"],
                    "chunks": sum(1 for d in _documents if d["metadata"].get("source") == src),
                })
        return unique