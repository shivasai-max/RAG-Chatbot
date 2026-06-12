# 🤖 RAG Chatbot — LangChain + Gemini + FAISS

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask natural-language questions about them. Built with:

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (via LangChain) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS |
| PDF loading | LangChain + PyPDF |

---

## Project Structure

```
rag-chatbot/
├── app.py                  ← Streamlit app entry point
├── requirements.txt
├── .env.example            ← Copy to .env and add your API key
├── data/
│   └── sample.pdf          ← Drop test PDFs here
├── vectorstore/            ← Auto-created; stores the FAISS index
└── utils/
    ├── __init__.py
    ├── pdf_loader.py       ← PDF extraction
    ├── text_splitter.py    ← Chunking logic
    ├── embeddings.py       ← HuggingFace embedding model
    ├── vector_store.py     ← FAISS create / save / load
    └── rag_chain.py        ← LangChain RAG chain + Gemini
```

---

## Prerequisites

- Python 3.10 – 3.13
- A Google Gemini API key (free tier available)

---

## Step-by-Step Setup

### 1 — Clone or download the project

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You'll see `(venv)` at the start of your terminal prompt when it's active.

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ First install takes a few minutes. The HuggingFace model (~80 MB) is downloaded automatically on first run.

### 4 — Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key (starts with `AIza...`)

### 5 — Configure your API key

**Option A — via `.env` file (recommended):**
```bash
cp .env.example .env
# Open .env in any text editor and replace the placeholder:
# GEMINI_API_KEY=AIzaSy...your_actual_key...
```

**Option B — enter it directly in the Streamlit sidebar** (no `.env` needed).

### 6 — Run the app

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501` automatically.

---

## How to Use

1. **Enter your Gemini API key** in the sidebar (if not set via `.env`)
2. **Upload a PDF** using the file uploader in the sidebar
3. Click **⚡ Process Documents** — this extracts text, creates embeddings, and builds the FAISS index
4. **Ask questions** in the chat box at the bottom
5. The chatbot will answer based on your document content and show the source pages

### Tips

- You can upload **multiple PDFs** — they all get indexed together
- Click **💾 Load Index** to reload a previously built index without re-embedding
- Click **🗑️ Clear Index** to start fresh with new documents
- Use **🧹 Clear Chat History** to reset the conversation

---

## How RAG Works (Technical Overview)

```
User Question
    │
    ▼
FAISS Retriever ──── finds top-4 matching chunks from indexed PDFs
    │
    ▼
Prompt Template ──── combines retrieved context + chat history + question
    │
    ▼
Gemini LLM ───────── generates a grounded answer from the context
    │
    ▼
Answer + Source Pages displayed in chat UI
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional vectors
- Runs on CPU, ~80 MB download
- Cached locally after first use

**Chunking strategy:** `RecursiveCharacterTextSplitter`
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY not found` | Add key to `.env` or enter it in the sidebar |
| `No text extracted from PDF` | PDF may be scanned (image-only). Try a text-based PDF |
| `faiss-cpu not found` | Run `pip install faiss-cpu` |
| Slow first run | Normal — HuggingFace model downloading (~80 MB) |
| `torch` install fails on Python 3.13 | Use Python 3.11 or 3.12 for best compatibility |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |

---

## License

MIT — free to use, modify, and distribute.
