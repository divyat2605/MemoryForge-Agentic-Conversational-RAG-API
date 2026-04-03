"""
mcp_tools.py — MCP-compatible tool definitions for the LangGraph agent.

Tools exposed:
- search_arxiv: fetch recent papers by keyword from arxiv API
- fetch_paper_abstract: get abstract for a given arxiv ID
- summarize_topic: generate a brief topic summary using LLM

These follow the MCP tool schema so they can be served via an MCP server
endpoint if needed (POST /mcp/tools).
"""

import httpx
from langchain.tools import tool
from typing import Optional


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arxiv for recent research papers by keyword.
    Returns titles, authors, and arxiv IDs.
    Use this when the user asks about recent papers or specific research topics.
    """
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()

        # naive parse — extract titles and ids from XML
        import re
        titles = re.findall(r"<title>(.*?)</title>", resp.text, re.DOTALL)[1:]  # skip feed title
        ids = re.findall(r"<id>http://arxiv.org/abs/(.*?)</id>", resp.text)

        if not titles:
            return f"No results found for: {query}"

        results = []
        for i, (title, arxiv_id) in enumerate(zip(titles, ids)):
            results.append(f"{i+1}. [{arxiv_id}] {title.strip()}")

        return "\n".join(results)

    except Exception as e:
        return f"arxiv search failed: {str(e)}"


@tool
def fetch_paper_abstract(arxiv_id: str) -> str:
    """
    Fetch the abstract of a paper given its arxiv ID (e.g. '2310.06825').
    Use this when the user wants details about a specific paper.
    """
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()

        import re
        summary_match = re.search(r"<summary>(.*?)</summary>", resp.text, re.DOTALL)
        title_match = re.findall(r"<title>(.*?)</title>", resp.text, re.DOTALL)

        title = title_match[1].strip() if len(title_match) > 1 else "Unknown"
        abstract = summary_match.group(1).strip() if summary_match else "Abstract not found"

        return f"Title: {title}\n\nAbstract: {abstract}"

    except Exception as e:
        return f"Failed to fetch paper: {str(e)}"


# registry for MCP endpoint
MCP_TOOLS = [search_arxiv, fetch_paper_abstract]


def get_tools():
    return MCP_TOOLS