from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions
based only on the provided document context.

If the answer is not present in the context,
say:
"I don't have enough information in the uploaded documents."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""


def build_rag_chain(
    vector_store: FAISS,
    llm_provider: str = "ollama",
    groq_api_key: str = "",
    k: int = 4,
    temperature: float = 0.3,
) -> ConversationalRetrievalChain:

    if llm_provider == "groq":

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=temperature,
        )

    else:

        llm = ChatOllama(
            model="llama3",
            temperature=temperature,
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    return chain


def query_rag_chain(
    chain: ConversationalRetrievalChain,
    question: str,
) -> Dict[str, Any]:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    result = chain.invoke({
        "question": question.strip()
    })

    return {
        "answer": result.get("answer", "No answer returned."),
        "source_documents": result.get("source_documents", []),
    }


def format_source_documents(source_docs: List[Any]) -> str:

    if not source_docs:
        return "No source documents retrieved."

    lines = []
    seen = set()

    for doc in source_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        key = f"{source}-p{page}"

        if key not in seen:
            seen.add(key)
            lines.append(f"📄 {source} — Page {page}")

    return "\n".join(lines)