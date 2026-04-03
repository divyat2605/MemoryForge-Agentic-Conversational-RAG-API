"""
eval.py — Evaluation framework for MemoryForge

- Retrieval recall@k
- Answer faithfulness (LLM-based or QA-based)
- Latency tracking
- Structured logging (JSONL)
"""

import time
import json
from typing import List, Dict, Callable, Optional

# --- Retrieval Recall@k ---
def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Compute recall@k: fraction of ground truth items in top-k retrieved.
    """
    retrieved_k = set(retrieved[:k])
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return len(retrieved_k & gt) / len(gt)

# --- Answer Faithfulness (LLM-based, placeholder) ---
def answer_faithfulness(answer: str, context: str, llm_judge: Optional[Callable] = None) -> float:
    """
    Score answer faithfulness to context (0-1). Uses LLM judge if provided.
    """
    if llm_judge:
        return llm_judge(answer, context)
    # Placeholder: simple string overlap
    overlap = len(set(answer.split()) & set(context.split()))
    return min(1.0, overlap / (len(answer.split()) + 1e-6))

# --- Latency Tracking ---
class LatencyTracker:
    def __init__(self):
        self.times = {}
    def start(self, key):
        self.times[key] = time.time()
    def stop(self, key):
        if key in self.times:
            self.times[key] = time.time() - self.times[key]
    def get(self, key):
        return self.times.get(key, None)
    def as_dict(self):
        return self.times

# --- Structured Logging ---
def log_metrics(metrics: Dict, log_path: str = "eval_metrics.jsonl"):
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# --- Example batch eval ---
def batch_eval(queries: List[Dict], retrieve_fn: Callable, answer_fn: Callable, k: int = 5, log_path: str = "eval_metrics.jsonl"):
    """
    queries: [{"query": str, "ground_truth": [str], "context": str}]
    retrieve_fn: (query) -> [doc_id]
    answer_fn: (query, context) -> answer
    """
    for q in queries:
        tracker = LatencyTracker()
        tracker.start("retrieval")
        retrieved = retrieve_fn(q["query"])
        tracker.stop("retrieval")
        recall = recall_at_k(retrieved, q["ground_truth"], k)
        tracker.start("generation")
        answer = answer_fn(q["query"], q["context"])
        tracker.stop("generation")
        faith = answer_faithfulness(answer, q["context"])
        metrics = {
            "query": q["query"],
            "recall@k": recall,
            "faithfulness": faith,
            "retrieval_latency": tracker.get("retrieval"),
            "generation_latency": tracker.get("generation"),
        }
        log_metrics(metrics, log_path=log_path)
