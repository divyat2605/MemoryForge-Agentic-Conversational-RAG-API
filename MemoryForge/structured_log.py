"""
structured_log.py — Structured JSONL logging for agent observability

- Logs per-node events, decision path, retrieval source, token usage, latency
- Use in each LangGraph node and FastAPI route
"""

import json
import time
from typing import Dict, Any

LOG_PATH = "agent_structured.log"

def log_event(event: Dict[str, Any], log_path: str = LOG_PATH):
    event["timestamp"] = time.time()
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")

# Example usage:
# log_event({"node": "retrieve", "decision": "bm25", "latency": 0.12, ...})
