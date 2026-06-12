"""
vector_store.py
---------------
Manages the FAISS vector store: creation, persistence, and loading.

FAISS (Facebook AI Similarity Search) is an in-memory vector database that
supports fast approximate nearest-neighbour search. We persist it to disk so
users don't have to re-embed documents on every app restart.

Directory layout on disk:
    vectorstore/
        index.faiss   ← the binary FAISS index
        index.pkl     ← the docstore (text chunks + metadata)
"""

import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Default folder where the FAISS index is saved/loaded
VECTORSTORE_DIR = "vectorstore"


def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    save_dir: str = VECTORSTORE_DIR,
) -> FAISS:
    """
    Embed a list of document chunks and store them in a new FAISS index.

    This overwrites any existing index in save_dir.

    Args:
        chunks:    List of text chunks (LangChain Documents) to embed.
        embeddings: The embedding model to use.
        save_dir:  Directory path where the FAISS index will be saved.

    Returns:
        A FAISS vector store loaded with all embedded chunks.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty list of chunks.")

    # Build the FAISS index from documents — this is the expensive embedding step
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Persist to disk so we can reload without re-embedding
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)

    return vector_store


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    load_dir: str = VECTORSTORE_DIR,
) -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.

    Args:
        embeddings: The same embedding model used when the index was created.
                    The model must match exactly, or similarity scores will be wrong.
        load_dir:   Directory path where the FAISS index is stored.

    Returns:
        A FAISS vector store, or None if no saved index exists.
    """
    index_file = os.path.join(load_dir, "index.faiss")

    if not os.path.exists(index_file):
        return None

    # allow_dangerous_deserialization=True is required by LangChain when
    # loading pickled data — only load indexes you created yourself
    vector_store = FAISS.load_local(
        folder_path=load_dir,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def add_documents_to_store(
    vector_store: FAISS,
    new_chunks: List[Document],
    save_dir: str = VECTORSTORE_DIR,
) -> FAISS:
    """
    Add new document chunks to an existing FAISS vector store without
    rebuilding the entire index.

    Args:
        vector_store: Existing FAISS store to update.
        new_chunks:   Additional chunks to embed and add.
        save_dir:     Directory to persist the updated index.

    Returns:
        The updated FAISS vector store.
    """
    if not new_chunks:
        return vector_store

    vector_store.add_documents(new_chunks)
    vector_store.save_local(save_dir)
    return vector_store


def vector_store_exists(load_dir: str = VECTORSTORE_DIR) -> bool:
    """
    Check whether a saved FAISS index exists on disk.

    Args:
        load_dir: Directory to check.

    Returns:
        True if index files are present, False otherwise.
    """
    return os.path.exists(os.path.join(load_dir, "index.faiss"))


def get_retriever(
    vector_store: FAISS,
    k: int = 4,
    search_type: str = "similarity",
):
    """
    Create a LangChain retriever from the FAISS vector store.

    Args:
        vector_store: The FAISS store to build the retriever from.
        k:            Number of top-matching chunks to retrieve per query.
        search_type:  "similarity" (cosine) or "mmr" (Max Marginal Relevance,
                      which improves diversity of retrieved chunks).

    Returns:
        A LangChain VectorStoreRetriever ready for use in a chain.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
