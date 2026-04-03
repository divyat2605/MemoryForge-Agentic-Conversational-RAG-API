"""
agent.py — LangGraph agentic RAG orchestration

Graph:
  START → retrieve → compress → generate → END
               ↑
         (tools available at generate node via MCP tools)

The agent can:
1. Retrieve from BM25 index with metadata filters
2. Compress context via LLMLingua
3. Call MCP tools (arxiv search, paper fetch) if needed
4. Generate final answer
"""

import os
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode


from retriever import retrieve
from compressor import compress_context
from mcp_tools import get_tools
from memory import load_history, append_turn, get_short_term, get_summarized_history, get_semantic_memory, get_memory_importance
from monitoring import traceable
from decision import decision_node


# ---------- state ----------

class AgentState(TypedDict):
    query: str
    filters: Optional[dict]
    top_k: int
    session_id: str                  # NEW — memory key
    chat_history: list[BaseMessage]  # NEW — short-term window
    retrieved_chunks: list
    compressed_context: str
    answer: str
    sources: list


# ---------- LLM ----------

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(get_tools())


# ---------- nodes ----------

@traceable(name="retrieve_node")
def retrieve_node(state: AgentState) -> AgentState:
    """BM25/hybrid retrieval with metadata filtering. Handles retrieval failure."""
    try:
        chunks = retrieve(
            query=state["query"],
            top_k=state["top_k"],
            filters=state.get("filters"),
        )
        if not chunks:
            # Fallback: mark for tool/arxiv fallback
            state = dict(state)
            state["retrieved_chunks"] = []
            state["retrieval_failed"] = True
            return state
        return {**state, "retrieved_chunks": chunks, "retrieval_failed": False}
    except Exception as e:
        # Log error, fallback
        state = dict(state)
        state["retrieved_chunks"] = []
        state["retrieval_failed"] = True
        state["retrieval_error"] = str(e)
        return state


@traceable(name="compress_node")
def compress_node(state: AgentState) -> AgentState:
    """LLMLingua context compression. Handles compression failure and empty context."""
    if not state["retrieved_chunks"]:
        # If retrieval failed, fallback message
        if state.get("retrieval_failed"):
            return {**state, "compressed_context": "No relevant documents found. Fallback to external tools if available."}
        return {**state, "compressed_context": "No relevant documents found in the index."}
    try:
        compressed = compress_context(state["retrieved_chunks"], state["query"])
        # Check if compression removed all context
        if not compressed or len(compressed.strip()) < 10:
            return {**state, "compressed_context": "Compression removed all key context. Please try rephrasing or use external tools."}
        return {**state, "compressed_context": compressed}
    except Exception as e:
        return {**state, "compressed_context": f"Compression failed: {str(e)}"}


@traceable(name="generate_node")
def generate_node(state: AgentState) -> AgentState:
    """LLM generation with compressed context + MCP tools available."""
    import time
    t0 = time.time()
    llm = get_llm()

    # --- Memory summarization and semantic memory ---
    full_history = state.get("chat_history", [])
    summarized = get_summarized_history(full_history, max_tokens=256) if full_history else ""
    semantic_turns = get_semantic_memory(full_history, state["query"], top_k=2) if full_history else []
    semantic_context = "\n".join([f"{t['role']}: {t['content']}" for t in semantic_turns])

    system_prompt = """You are MemoryForge, an expert research assistant specializing in academic papers.

Answer the user's question using the provided context from retrieved research documents.
You also have access to a summary and semantic memory of the recent conversation history — use it to resolve references like "this", "it", "that paper" etc.
If the context is insufficient, you may use the available tools (search_arxiv, fetch_paper_abstract) to find additional information.

Guidelines:
- Be precise and cite sources when possible
- Use conversation history to maintain continuity across turns
- If you use a tool, incorporate its output into your final answer
- Mention the paper source (filename/arxiv id) when referencing specific claims
- If no relevant context exists, say so clearly
"""

    user_message = f"""Context from retrieved research documents:
---
{state['compressed_context']}
---

Conversation summary:
{summarized}

Relevant past turns:
{semantic_context}

Question: {state['query']}"""

    messages = [SystemMessage(content=system_prompt)]
    messages += state.get("chat_history", [])    # short-term memory window
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    latency = time.time() - t0
    from structured_log import log_event
    log_event({
        "node": "generate",
        "token_usage": getattr(response, "usage", None),
        "latency": latency,
        "decision_path": {
            "decision_route": state.get("decision_route"),
            "retrieval_failed": state.get("retrieval_failed"),
            "use_arxiv_tool": state.get("use_arxiv_tool"),
        },
    })

    sources = []
    for c in state["retrieved_chunks"]:
        src = {
            "id": c["id"],
            "source": c["metadata"].get("source", "unknown"),
            "score": c["score"],
            "year": c["metadata"].get("year"),
            "author": c["metadata"].get("author"),
            "topics": c["metadata"].get("topics", []),
        }
        if "rerank_score" in c:
            src["rerank_score"] = c["rerank_score"]
        sources.append(src)

    return {
        **state,
        "answer": response.content,
        "sources": sources,
    }


# ---------- graph ----------


def arxiv_tool_node(state: AgentState) -> AgentState:
    """Node for arxiv/tool fallback. Handles tool failure gracefully."""
    state = dict(state)
    try:
        # Mark that arxiv/tool should be used in generate_node
        state["use_arxiv_tool"] = True
        state["arxiv_tool_failed"] = False
        return state
    except Exception as e:
        state["use_arxiv_tool"] = False
        state["arxiv_tool_failed"] = True
        state["arxiv_tool_error"] = str(e)
        return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decision", decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("arxiv_tool", arxiv_tool_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)
    graph.add_node("tools", ToolNode(get_tools()))

    # Routing: decision_node decides which retrieval path to take
    graph.add_edge(START, "decision")
    graph.add_conditional_edges(
        "decision",
        lambda state: state.get("decision_route", "bm25"),
        {
            "bm25": "retrieve",
            "arxiv": "arxiv_tool",
            "none": "compress",  # skip retrieval, compress empty context
        },
    )
    graph.add_edge("retrieve", "compress")
    graph.add_edge("arxiv_tool", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_agent(query: str, filters: Optional[dict] = None, top_k: int = 5, session_id: str = "default") -> dict:
    graph = get_graph()

    # load long-term history, trim to short-term window
    full_history = await load_history(session_id)
    short_term = get_short_term(full_history)

    initial_state: AgentState = {
        "query": query,
        "filters": filters or {},
        "top_k": top_k,
        "session_id": session_id,
        "chat_history": short_term,
        "retrieved_chunks": [],
        "compressed_context": "",
        "answer": "",
        "sources": [],
    }

    result = await graph.ainvoke(initial_state)

    # persist turn to Redis long-term memory
    await append_turn(session_id, query, result["answer"])

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }