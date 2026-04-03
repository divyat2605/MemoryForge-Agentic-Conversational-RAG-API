import re
import json
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from retriever import add_documents_to_index


def extract_metadata(filename: str, content: str) -> dict:
    """
    Extract metadata from research doc content heuristically.
    In production: use a dedicated metadata extraction LLM call.
    """
    metadata = {
        "source": filename,
        "type": "research_paper",
    }

    # try to extract year from filename or content
    year_match = re.search(r"(20\d{2}|19\d{2})", filename + content[:500])
    if year_match:
        metadata["year"] = year_match.group(1)

    # try to extract arxiv id
    arxiv_match = re.search(r"arxiv[:\s]*([\d]{4}\.[\d]{4,5})", content[:1000], re.IGNORECASE)
    if arxiv_match:
        metadata["arxiv_id"] = arxiv_match.group(1)

    # naive author extraction — first "Author:" or "by " mention
    author_match = re.search(r"(?:Author[s]?:|by\s)([A-Z][a-z]+ [A-Z][a-z]+)", content[:2000])
    if author_match:
        metadata["author"] = author_match.group(1)

    # topic tag from filename keywords
    topics = ["transformer", "diffusion", "llm", "vision", "rag", "agent", "bert", "gpt"]
    detected = [t for t in topics if t in filename.lower() or t in content[:500].lower()]
    if detected:
        metadata["topics"] = detected

    return metadata


def chunk_document(content: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(content)


async def ingest_document(filename: str, content: str) -> dict:
    """
    Full ingestion pipeline:
    1. Extract metadata
    2. Chunk document
    3. Add to BM25 index with metadata
    """
    metadata = extract_metadata(filename, content)
    chunks = chunk_document(content)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": f"{filename}__chunk_{i}",
            "text": chunk,
            "metadata": {**metadata, "chunk_index": i, "total_chunks": len(chunks)},
        })

    add_documents_to_index(docs)

    return {
        "filename": filename,
        "chunks": len(chunks),
        "metadata": metadata,
    }