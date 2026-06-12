"""
text_splitter.py
----------------
Splits LangChain Documents into smaller chunks suitable for embedding.

Chunk size and overlap are tunable parameters that affect retrieval quality:
- Larger chunks → more context per chunk, fewer total chunks, but coarser retrieval
- Smaller chunks → more precise retrieval, but may lose surrounding context
- Overlap ensures that sentences near chunk boundaries are not missed
"""

from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Default chunking configuration — tweak these based on your use case
DEFAULT_CHUNK_SIZE = 1000       # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200     # characters shared between adjacent chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks,
    then sentence boundaries, then word boundaries — preserving natural
    text structure as much as possible.

    Args:
        documents:     List of LangChain Document objects to split.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A new list of Document objects with smaller text content.
        Each chunk retains the metadata (source, page) from its parent.
    """
    if not documents:
        raise ValueError("No documents provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split priority: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "Text splitting produced no chunks. "
            "The PDF may be empty or contain only images."
        )

    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Return basic statistics about a list of chunks.
    Useful for displaying info in the Streamlit sidebar.

    Args:
        chunks: List of split Document objects.

    Returns:
        Dictionary with count, avg_length, min_length, max_length.
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(chunk.page_content) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
