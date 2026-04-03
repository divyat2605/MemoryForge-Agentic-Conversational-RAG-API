"""
decision.py — Query classification and routing logic for agentic RAG

This module decides:
- Whether retrieval is needed
- Whether to use local BM25 or external tool (arXiv)
- Confidence scoring for routing

Extendable: swap out rule-based logic for LLM/embedding classifier as needed.
"""

from typing import Dict, Tuple
import re

# --- Simple rule-based classifier (can be replaced with LLM/embedding model) ---

def classify_query(query: str) -> Tuple[str, float]:
    """
    Classify query type and return (route, confidence).
    Routes: 'bm25', 'arxiv', 'none'
    """
    q = query.lower().strip()
    # Example rules (replace with LLM/embedding classifier for prod)
    if any(x in q for x in ["arxiv", "latest paper", "find paper", "recent research"]):
        return ("arxiv", 0.95)
    if len(q) < 8 or re.match(r"^(hi|hello|thanks|bye)$", q):
        return ("none", 0.99)
    # Default: use BM25
    return ("bm25", 0.8)


def decision_node(state: Dict) -> Dict:
    """
    Decide retrieval route and attach to state.
    """
    query = state.get("query", "")
    route, confidence = classify_query(query)
    state = dict(state)
    state["decision_route"] = route
    state["decision_confidence"] = confidence
    return state
