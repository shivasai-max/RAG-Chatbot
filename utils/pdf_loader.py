"""
pdf_loader.py
-------------
Handles loading and extracting text from PDF files.
Supports both uploaded files (via Streamlit) and file paths on disk.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile


def load_pdf_from_path(file_path: str) -> List[Document]:
    """
    Load a PDF from a file path on disk.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects, one per page.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is not a PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Attach the source filename to each document's metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    return documents


def load_pdf_from_upload(uploaded_file) -> List[Document]:
    """
    Load a PDF from a Streamlit UploadedFile object.

    Streamlit uploaded files are in-memory bytes objects, so we write them
    to a temporary file on disk before passing to PyPDFLoader.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        List of LangChain Document objects, one per page.
    """
    # Write the uploaded bytes to a named temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        documents = load_pdf_from_path(tmp_path)

        # Replace the temp path with the real filename in metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)

    return documents


def get_pdf_page_count(uploaded_file) -> int:
    """
    Return the number of pages in a PDF without loading full content.
    Useful for showing quick stats in the UI.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        Number of pages as an integer.
    """
    docs = load_pdf_from_upload(uploaded_file)
    return len(docs)
