"""
rag_chain.py
------------
Builds the RAG (Retrieval-Augmented Generation) chain using LangChain + Gemini.

Flow:
  User question
      ↓
  Retriever (FAISS) → top-k relevant chunks
      ↓
  Prompt template (context + question)
      ↓
  Gemini LLM
      ↓
  Answer + source documents
"""

import os
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# This prompt instructs Gemini to stay grounded in the retrieved context
# and honestly admit when the answer isn't available.

RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided document context.

Use the following retrieved context to answer the question. If the answer is not contained
in the context, say "I don't have enough information in the uploaded documents to answer
that question" — do not make up an answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (be concise, accurate, and cite the relevant section when helpful):"""


def build_rag_chain(
    vector_store: FAISS,
    api_key: str,
    k: int = 4,
    temperature: float = 0.3,
    model_name: str = "gemini-2.0-flash",
) -> ConversationalRetrievalChain:
    """
    Build and return a ConversationalRetrievalChain backed by Gemini.

    A ConversationalRetrievalChain:
    1. Condenses the user's question + chat history into a standalone question
    2. Retrieves top-k chunks from FAISS
    3. Passes chunks + question to the LLM for a grounded answer

    Args:
        vector_store: Populated FAISS vector store.
        api_key:      Google Gemini API key.
        k:            Number of chunks to retrieve per query.
        temperature:  LLM temperature (0 = deterministic, 1 = creative).
        model_name:   Gemini model to use.

    Returns:
        A ready-to-invoke ConversationalRetrievalChain.
    """
    # Gemini LLM via LangChain
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,   # Gemini requires this flag
    )

    # Retriever — fetches the k most relevant chunks for each query
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Custom answer-generation prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # Conversation memory — keeps track of past turns so Gemini can handle
    # follow-up questions that reference earlier answers
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",        # store only the final answer in memory
    )

    # Assemble the full chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,   # include retrieved chunks in the response
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    return chain


def query_rag_chain(
    chain: ConversationalRetrievalChain,
    question: str,
) -> Dict[str, Any]:
    """
    Run a user question through the RAG chain.

    Args:
        chain:    The ConversationalRetrievalChain to query.
        question: The user's natural-language question.

    Returns:
        A dictionary with keys:
            "answer"           — Gemini's response string
            "source_documents" — list of retrieved LangChain Documents
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    result = chain.invoke({"question": question.strip()})

    return {
        "answer": result.get("answer", "No answer returned."),
        "source_documents": result.get("source_documents", []),
    }


def format_source_documents(source_docs: List[Any]) -> str:
    """
    Format retrieved source documents into a readable string for display.

    Args:
        source_docs: List of LangChain Document objects from the chain response.

    Returns:
        A formatted string listing each source chunk with its metadata.
    """
    if not source_docs:
        return "No source documents retrieved."

    lines = []
    seen = set()

    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}-p{page}"

        if key not in seen:
            seen.add(key)
            lines.append(f"📄 **{source}** — Page {page}")

    return "\n".join(lines) if lines else "No source documents retrieved."
