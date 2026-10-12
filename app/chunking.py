"""Split documents into overlapping chunks suitable for embedding."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int, source_id: str
) -> list[Document]:
    """Split page-level documents into chunks.

    Metadata from each page is copied onto its chunks; ``chunk_index`` and a
    stable ``chunk_id`` (used as the FAISS docstore id) are added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["source_id"] = source_id
        digest = hashlib.sha1(f"{source_id}|{index}".encode()).hexdigest()[:20]
        chunk.metadata["chunk_id"] = digest
    return chunks
