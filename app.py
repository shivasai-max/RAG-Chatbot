import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_pdf_from_upload
from utils.text_splitter import split_documents, get_chunk_stats
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    load_vector_store,
    add_documents_to_store,
    vector_store_exists,
)

from utils.rag_chain import (
    build_rag_chain,
    query_rag_chain,
    format_source_documents,
)

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

def init_session_state():

    defaults = {
        "chat_history": [],
        "rag_chain": None,
        "vector_store": None,
        "embeddings": None,
        "processed_files": set(),
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_embeddings_cached():

    return get_embedding_model()


def render_sidebar():

    with st.sidebar:

        st.title("🤖 RAG Chatbot")

        llm_provider = st.radio(
            "Choose Model",
            ["ollama", "groq"],
            horizontal=True
        )

        st.markdown("### Current Status")

        if llm_provider == "ollama":

            st.success("🟢 Running on Ollama Local")

        else:

            st.success("🟢 Running on Groq API")

        st.divider()

        groq_api_key = ""

        if llm_provider == "groq":

            groq_api_key = st.text_input(
                "Groq API Key",
                type="password"
            )

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:

            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.processed_files
            ]

            if new_files:

                if st.button("⚡ Process Documents"):

                    process_uploaded_files(
                        new_files,
                        llm_provider,
                        groq_api_key
                    )

            else:

                st.success(
                    f"{len(uploaded_files)} file(s) already processed"
                )

        st.divider()

        if vector_store_exists():

            col1, col2 = st.columns(2)

            with col1:

                if st.button("📂 Load Index"):

                    load_existing_index(
                        llm_provider,
                        groq_api_key
                    )

            with col2:

                if st.button("🗑️ Clear Index"):

                    clear_vector_store()

        st.divider()

        if st.button("🧹 Clear Chat"):

            st.session_state.chat_history = []

            if st.session_state.rag_chain:
                st.session_state.rag_chain.memory.clear()

            st.rerun()


def process_uploaded_files(
    files,
    llm_provider,
    groq_api_key
):

    if st.session_state.embeddings is None:

        st.session_state.embeddings = load_embeddings_cached()

    all_chunks = []

    for file in files:

        try:

            documents = load_pdf_from_upload(file)

            chunks = split_documents(documents)

            all_chunks.extend(chunks)

            stats = get_chunk_stats(chunks)

            st.sidebar.success(
                f"{file.name} → {stats['count']} chunks"
            )

            st.session_state.processed_files.add(file.name)

        except Exception as e:

            st.sidebar.error(str(e))

    if not all_chunks:

        return

    try:

        if st.session_state.vector_store is None:

            st.session_state.vector_store = create_vector_store(
                all_chunks,
                st.session_state.embeddings
            )

        else:

            st.session_state.vector_store = add_documents_to_store(
                st.session_state.vector_store,
                all_chunks
            )

        st.session_state.rag_chain = build_rag_chain(
            vector_store=st.session_state.vector_store,
            llm_provider=llm_provider,
            groq_api_key=groq_api_key,
        )

        st.sidebar.success("Documents processed successfully!")

    except Exception as e:

        st.sidebar.error(str(e))


def load_existing_index(
    llm_provider,
    groq_api_key
):

    try:

        if st.session_state.embeddings is None:

            st.session_state.embeddings = load_embeddings_cached()

        vs = load_vector_store(
            st.session_state.embeddings
        )

        if vs is None:

            st.sidebar.error("No saved index found.")

            return

        st.session_state.vector_store = vs

        st.session_state.rag_chain = build_rag_chain(
            vector_store=vs,
            llm_provider=llm_provider,
            groq_api_key=groq_api_key,
        )

        st.sidebar.success("Index loaded!")

    except Exception as e:

        st.sidebar.error(str(e))


def clear_vector_store():

    try:

        if os.path.exists("vectorstore"):

            shutil.rmtree("vectorstore")

        st.session_state.vector_store = None
        st.session_state.rag_chain = None
        st.session_state.processed_files = set()

        st.sidebar.success("Vector store cleared.")

        st.rerun()

    except Exception as e:

        st.sidebar.error(str(e))


def render_chat():

    st.title("💬 RAG Chatbot")

    st.caption("Ask questions about your uploaded PDFs")

    if st.session_state.rag_chain is None:

        st.info("Upload PDFs and process them first.")

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander("📚 Sources"):

                    st.markdown(message["sources"])

    user_input = st.chat_input(
        "Ask a question...",
        disabled=(st.session_state.rag_chain is None),
    )

    if user_input:

        handle_user_message(user_input)


def handle_user_message(user_input):

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):

        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        placeholder.markdown("⏳ Thinking...")

        try:

            result = query_rag_chain(
                st.session_state.rag_chain,
                user_input
            )

            answer = result["answer"]

            sources = format_source_documents(
                result["source_documents"]
            )

            placeholder.markdown(answer)

            with st.expander("📚 Sources"):

                st.markdown(sources)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:

            placeholder.error(str(e))


def main():

    init_session_state()

    render_sidebar()

    render_chat()


if __name__ == "__main__":

    main()