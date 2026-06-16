# 🤖 Hybrid RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built using:

* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Ollama (Local LLM)
* Groq API (Cloud LLM)

Upload PDF documents and ask natural-language questions about them.

---

# Features

* 📄 Upload multiple PDFs
* ✂️ Intelligent text chunking
* 🧠 Sentence Transformer embeddings
* 💾 FAISS vector database
* 🔍 Semantic retrieval
* 🤖 Dual LLM support:

  * Ollama Local
  * Groq API
* 💬 Chat interface using Streamlit
* 📚 Source page references
* ⚡ Local-first architecture

---

# Tech Stack

| Component    | Technology            |
| ------------ | --------------------- |
| Frontend     | Streamlit             |
| Framework    | LangChain             |
| Embeddings   | sentence-transformers |
| Vector Store | FAISS                 |
| Local LLM    | Ollama + Llama3       |
| Cloud LLM    | Groq API              |
| PDF Loader   | PyPDFLoader           |

---

# Project Structure

```bash
rag-chatbot/
│
├── app.py
├── requirements.txt
├── vectorstore/
│
├── utils/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_chain.py
│
└── data/
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup (Local Model)

## Install Ollama

Download:

https://ollama.com/download/windows

---

## Pull Llama3 Model

```bash
ollama pull llama3
```

---

## Start Ollama

```bash
ollama run llama3
```

Keep the terminal running in background.

---

# Groq API Setup

Create API Key:

https://console.groq.com/keys

Use the API key inside the Streamlit sidebar.

---

# Run Application

```bash
streamlit run app.py
```

---

# How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Ollama OR Groq
    ↓
Answer Generation
```

---

# Usage

1. Select:

   * Ollama
   * OR Groq

2. Upload PDF files

3. Click:

   ```text
   Process Documents
   ```

4. Ask questions in chat

5. View source pages used for answers

---

# Supported Modes

| Mode   | Description           |
| ------ | --------------------- |
| Ollama | Fully local inference |
| Groq   | Fast cloud inference  |

---

# Troubleshooting

| Problem               | Solution                |
| --------------------- | ----------------------- |
| Ollama not responding | Run `ollama run llama3` |
| Slow responses        | Use Groq mode           |
| No PDF text extracted | Use text-based PDFs     |
| FAISS error           | Reinstall `faiss-cpu`   |

---

# License

MIT License
