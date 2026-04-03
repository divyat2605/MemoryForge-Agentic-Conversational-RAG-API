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
from memory import load_history, append_turn, get_short_term
from monitoring import traceable


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
    """BM25 retrieval with metadata filtering."""
    chunks = retrieve(
        query=state["query"],
        top_k=state["top_k"],
        filters=state.get("filters"),
    )
    return {**state, "retrieved_chunks": chunks}


@traceable(name="compress_node")
def compress_node(state: AgentState) -> AgentState:
    """LLMLingua context compression."""
    if not state["retrieved_chunks"]:
        return {**state, "compressed_context": "No relevant documents found in the index."}

    compressed = compress_context(state["retrieved_chunks"], state["query"])
    return {**state, "compressed_context": compressed}


@traceable(name="generate_node")
def generate_node(state: AgentState) -> AgentState:
    """LLM generation with compressed context + MCP tools available."""
    llm = get_llm()

    system_prompt = """You are MemoryForge, an expert research assistant specializing in academic papers.

Answer the user's question using the provided context from retrieved research documents.
You also have access to the recent conversation history — use it to resolve references like "this", "it", "that paper" etc.
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

Question: {state['query']}"""

    messages = [SystemMessage(content=system_prompt)]
    messages += state.get("chat_history", [])    # short-term memory window
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)

    sources = [
        {
            "id": c["id"],
            "source": c["metadata"].get("source", "unknown"),
            "score": c["score"],
            "year": c["metadata"].get("year"),
            "author": c["metadata"].get("author"),
            "topics": c["metadata"].get("topics", []),
        }
        for c in state["retrieved_chunks"]
    ]

    return {
        **state,
        "answer": response.content,
        "sources": sources,
    }


# ---------- graph ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)
    graph.add_node("tools", ToolNode(get_tools()))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "compress")
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