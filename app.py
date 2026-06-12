"""
app.py
------
Main Streamlit application for the RAG Chatbot.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  Sidebar: API key input + PDF upload + index mgmt   │
  │  Main area: Chat interface with history             │
  └─────────────────────────────────────────────────────┘

Session state keys used:
  - chat_history    : list of {"role": "user"/"assistant", "content": str}
  - rag_chain       : built ConversationalRetrievalChain (or None)
  - vector_store    : FAISS index (or None)
  - embeddings      : loaded HuggingFaceEmbeddings (or None)
  - processed_files : set of filenames already embedded
"""

import os
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
from utils.rag_chain import build_rag_chain, query_rag_chain, format_source_documents

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Chat message bubbles */
        .user-message {
            background: #1e3a5f;
            color: #ffffff;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
        }
        .assistant-message {
            background: #2d2d2d;
            color: #e8e8e8;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 80%;
            word-wrap: break-word;
        }
        /* Source section styling */
        .source-box {
            background: #1a1a2e;
            border-left: 3px solid #4a9eff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #aaa;
            margin-top: 6px;
        }
        /* Remove default Streamlit padding in main area */
        .block-container { padding-top: 2rem; }

        /* Sidebar header */
        .sidebar-header {
            font-size: 1.1em;
            font-weight: 600;
            color: #4a9eff;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: initialise session state defaults
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helper: lazy-load the embedding model (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings_cached():
    return get_embedding_model()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("*Powered by Gemini + FAISS*")
        st.divider()

        # ── API Key ──────────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">🔑 Gemini API Key</p>', unsafe_allow_html=True)
        api_key = st.text_input(
            label="API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            help="Get your key at https://aistudio.google.com/app/apikey",
            label_visibility="collapsed",
        )
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        st.divider()

        # ── PDF Upload ────────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">📄 Upload Documents</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            label="Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to query",
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.processed_files
            ]

            if new_files:
                if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
                    process_uploaded_files(new_files, api_key)
            else:
                st.success(f"✅ {len(uploaded_files)} file(s) already processed")

        st.divider()

        # ── Saved Index ───────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-header">💾 Vector Store</p>', unsafe_allow_html=True)

        if vector_store_exists():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load Index", use_container_width=True):
                    load_existing_index(api_key)
            with col2:
                if st.button("🗑️ Clear Index", use_container_width=True):
                    clear_vector_store()
        else:
            st.info("No saved index found. Upload and process PDFs first.")

        st.divider()

        # ── Stats ─────────────────────────────────────────────────────────────
        if st.session_state.vector_store:
            st.markdown('<p class="sidebar-header">📊 Index Info</p>', unsafe_allow_html=True)
            try:
                n_vectors = st.session_state.vector_store.index.ntotal
                st.metric("Vectors stored", n_vectors)
            except Exception:
                pass
            if st.session_state.processed_files:
                st.markdown("**Indexed files:**")
                for fname in st.session_state.processed_files:
                    st.markdown(f"• {fname}")

        # ── Clear Chat ────────────────────────────────────────────────────────
        st.divider()
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            # Reset chain memory too so Gemini forgets prior turns
            if st.session_state.rag_chain:
                st.session_state.rag_chain.memory.clear()
            st.rerun()

    return api_key


# ---------------------------------------------------------------------------
# Document processing pipeline
# ---------------------------------------------------------------------------
def process_uploaded_files(files, api_key: str):
    """Run the full ingestion pipeline: load → split → embed → store."""
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key first.")
        return

    # Ensure embeddings are loaded
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_cached()

    all_chunks = []
    progress_bar = st.sidebar.progress(0, text="Starting…")

    for i, file in enumerate(files):
        try:
            progress_bar.progress(
                int((i / len(files)) * 60),
                text=f"Loading {file.name}…",
            )

            # 1. Extract text from PDF
            documents = load_pdf_from_upload(file)

            # 2. Split into chunks
            chunks = split_documents(documents)
            all_chunks.extend(chunks)

            stats = get_chunk_stats(chunks)
            st.sidebar.caption(
                f"✓ {file.name}: {len(documents)} pages → {stats['count']} chunks"
            )
            st.session_state.processed_files.add(file.name)

        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {e}")
            continue

    if not all_chunks:
        progress_bar.empty()
        st.sidebar.error("No text could be extracted from the uploaded files.")
        return

    try:
        progress_bar.progress(70, text="Building FAISS index…")

        # 3. Create or update vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_vector_store(
                all_chunks, st.session_state.embeddings
            )
        else:
            st.session_state.vector_store = add_documents_to_store(
                st.session_state.vector_store, all_chunks
            )

        progress_bar.progress(90, text="Building RAG chain…")

        # 4. Build the RAG chain
        st.session_state.rag_chain = build_rag_chain(
            vector_store=st.session_state.vector_store,
            api_key=api_key,
        )

        progress_bar.progress(100, text="Done!")
        st.sidebar.success(
            f"✅ Indexed {len(all_chunks)} chunks from {len(files)} file(s)."
        )

    except Exception as e:
        progress_bar.empty()
        st.sidebar.error(f"Failed to build index: {e}")


# ---------------------------------------------------------------------------
# Load an existing saved FAISS index
# ---------------------------------------------------------------------------
def load_existing_index(api_key: str):
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key first.")
        return

    try:
        with st.sidebar.status("Loading saved index…"):
            if st.session_state.embeddings is None:
                st.session_state.embeddings = load_embeddings_cached()

            vs = load_vector_store(st.session_state.embeddings)
            if vs is None:
                st.sidebar.error("No saved index found on disk.")
                return

            st.session_state.vector_store = vs
            st.session_state.rag_chain = build_rag_chain(
                vector_store=vs,
                api_key=api_key,
            )

        st.sidebar.success("✅ Index loaded successfully!")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to load index: {e}")


# ---------------------------------------------------------------------------
# Clear the saved index and reset state
# ---------------------------------------------------------------------------
def clear_vector_store():
    import shutil
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        st.session_state.vector_store = None
        st.session_state.rag_chain = None
        st.session_state.processed_files = set()
        st.sidebar.success("Vector store cleared.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Could not clear store: {e}")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
def render_chat(api_key: str):
    st.title("💬 RAG Chatbot")
    st.caption("Ask questions about your uploaded documents")

    # ── Status banner ─────────────────────────────────────────────────────────
    if st.session_state.rag_chain is None:
        st.info(
            "👈 **Get started:** Upload a PDF in the sidebar, enter your Gemini API key, "
            "and click **Process Documents**.",
            icon="ℹ️",
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            with st.chat_message(role):
                st.markdown(content)

                # Show sources for assistant messages if available
                if role == "assistant" and message.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        st.markdown(message["sources"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder="Ask a question about your documents…",
        disabled=(st.session_state.rag_chain is None),
    )

    if user_input:
        handle_user_message(user_input)


# ---------------------------------------------------------------------------
# Handle a submitted user message
# ---------------------------------------------------------------------------
def handle_user_message(user_input: str):
    # Guard: chain must be ready
    if st.session_state.rag_chain is None:
        st.error("Please process a document before asking questions.")
        return

    # Append user turn to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("⏳ *Thinking…*")

        try:
            result = query_rag_chain(st.session_state.rag_chain, user_input)
            answer = result["answer"]
            sources = format_source_documents(result["source_documents"])

            thinking_placeholder.markdown(answer)

            # Show sources in an expander below the answer
            with st.expander("📚 Sources", expanded=False):
                st.markdown(sources)

            # Save assistant turn (including sources) to history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"❌ Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg}
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    api_key = render_sidebar()
    render_chat(api_key)


if __name__ == "__main__":
    main()
