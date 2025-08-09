# rag_chatbot/global_objects.py
"""
Global singleton loaders for LLM and embedding model.
Ensures they are loaded only once in memory and reused across CLI + API.
"""

import os
from .vector_store import get_embedding_model, load_local_vector_store, get_retriever
from .LLM_interface import load_llm

# Singleton instances
_embedding_model = None
_llm = None


def get_embedder():
    """Return a singleton embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("🔍 Loading embedding model...")
        _embedding_model = get_embedding_model()
    return _embedding_model


def get_llm(use_gpu=False):
    """Return a singleton LLM instance."""
    global _llm
    if _llm is None:
        print("🤖 Loading LLM...")
        _llm = load_llm(use_gpu=use_gpu)
    return _llm


def load_vectorstore(cache_dir: str, top_k: int = 3):
    """
    Load a FAISS vector store and return a retriever.
    Raises FileNotFoundError if the index doesn't exist.
    """
    embedder = get_embedder()
    index_file = os.path.join(cache_dir, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No FAISS index found at {cache_dir}")
    vectorstore = load_local_vector_store(cache_dir, embedder)
    return get_retriever(vectorstore, top_k=top_k)
