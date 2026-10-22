"""
rag.py - Document loading, chunking, and ChromaDB vector store.

Free stack:
- pypdf for PDF
- python-docx for DOCX
- plain text for TXT / MD
- ChromaDB with its default embedding function (all-MiniLM-L6-v2)
  so we avoid a heavy torch / sentence-transformers install.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "rag_documents"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def load_docx(file_bytes: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    # Tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n\n".join(parts)


def load_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def load_document(file_bytes: bytes, filename: str) -> str:
    """Load a document from bytes based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_bytes)
    elif ext == ".docx":
        return load_docx(file_bytes)
    elif ext in {".txt", ".md", ".markdown"}:
        return load_text(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, MD, or DOCX.")


def load_document_from_path(path: str | Path) -> str:
    path = Path(path)
    return load_document(path.read_bytes(), path.name)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple character-based splitter with overlap.
    Tries to break on paragraph / sentence boundaries when close.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to find a nice break point in the last 20% of the chunk
        if end < length:
            break_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            # Only use a candidate if it's reasonably far into the chunk
            for cand in break_candidates:
                if cand > start + chunk_size * 0.6:
                    end = cand + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    """
    Get or create the collection.
    Chroma's default embedding function uses all-MiniLM-L6-v2
    (downloaded once, runs locally, free).
    """
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_document(text: str, source_name: str) -> int:
    """
    Chunk a document and add all chunks to Chroma.
    Returns number of chunks added.
    Each chunk id: f"{source}_{i}", metadata: {"source": source_name}
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection = get_collection()

    # Make source safe for ids
    safe_source = source_name.replace(" ", "_").replace("/", "_")
    ids = [f"{safe_source}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    # Upsert so re-uploading the same file replaces old chunks
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def search(query: str, n_results: int = 4) -> List[Dict[str, Any]]:
    """
    Search the vector store.
    Returns list of dicts: {"text": str, "source": str, "distance": float}
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    n = min(n_results, count)
    results = collection.query(query_texts=[query], n_results=n)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        out.append(
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "distance": dist,
            }
        )
    return out


def clear_collection() -> None:
    """Delete and recreate the collection (clears all documents)."""
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(name=COLLECTION_NAME)


def count_documents() -> int:
    """Return number of chunks currently stored."""
    try:
        collection = get_collection()
        return collection.count()
    except Exception:
        return 0
