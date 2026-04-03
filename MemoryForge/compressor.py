"""
compressor.py — LLMLingua context compression

Compresses retrieved chunks before sending to LLM.
Typical compression ratio: 4x-6x token reduction with <5% quality loss.

If llmlingua is not installed / model not available, falls back to
simple extractive compression (top sentences by keyword overlap).
"""

from typing import List
import re


def _extractive_fallback(chunks: list[str], query: str, target_ratio: float = 0.5) -> str:
    """
    Fallback compressor: keeps sentences with highest keyword overlap with query.
    No model required.
    """
    query_tokens = set(query.lower().split())
    all_sentences = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sent in sentences:
            score = len(set(sent.lower().split()) & query_tokens)
            all_sentences.append((score, sent))

    all_sentences.sort(key=lambda x: x[0], reverse=True)
    target_count = max(1, int(len(all_sentences) * target_ratio))
    compressed = " ".join(s for _, s in all_sentences[:target_count])
    return compressed


def compress_context(chunks: list[dict], query: str) -> str:
    """
    Compress retrieved chunks using LLMLingua.
    Falls back to extractive compression if LLMLingua unavailable.

    Args:
        chunks: list of retriever result dicts with "text" key
        query: original user query for compression guidance

    Returns:
        compressed context string
    """
    texts = [c["text"] for c in chunks]
    full_context = "\n\n".join(texts)

    try:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )

        result = compressor.compress_prompt(
            full_context,
            instruction=f"Answer the query: {query}",
            question=query,
            target_token=512,
            condition_compare=True,
        )

        compressed = result["compressed_prompt"]
        original_tokens = result.get("origin_tokens", "?")
        compressed_tokens = result.get("compressed_tokens", "?")
        print(f"[LLMLingua] {original_tokens} → {compressed_tokens} tokens")
        return compressed

    except ImportError:
        print("[compressor] LLMLingua not installed, using extractive fallback")
        return _extractive_fallback(texts, query)
    except Exception as e:
        print(f"[compressor] LLMLingua error: {e}, using extractive fallback")
        return _extractive_fallback(texts, query)