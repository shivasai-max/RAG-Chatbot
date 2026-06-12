"""
embeddings.py
-------------
Creates and caches HuggingFace sentence-transformer embeddings.

We use the 'all-MiniLM-L6-v2' model because it:
- Is lightweight (80 MB) and fast on CPU
- Produces 384-dimensional vectors with strong semantic quality
- Is freely available on HuggingFace Hub without an API key

The model is cached locally after the first download so subsequent runs
are instant. Streamlit's @st.cache_resource ensures the model is loaded
only once per server session, not on every user interaction.
"""

from langchain_huggingface import HuggingFaceEmbeddings


# Model identifier on HuggingFace Hub
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Device — use "cuda" if you have a GPU, "cpu" otherwise
EMBEDDING_DEVICE = "cpu"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace sentence-transformer embedding model.

    The model is downloaded from HuggingFace Hub on first call and cached
    locally in ~/.cache/huggingface/. Subsequent calls reuse the cache.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,   # Cosine similarity works better with L2-normalized vectors
            "batch_size": 32,               # Process 32 chunks at a time for efficiency
        },
    )
    return embeddings
