# rag_chatbot/global_objects.py
"""
Singleton loaders for embedder and LLM instances.
Supports:
    - Hugging Face models (GPU/CPU switch)
    - Ollama models (local LLM server)
Caches models keyed by (backend, model_id, use_gpu) for performance.
"""

from typing import Optional
import os
import torch
import subprocess

from .LLM_interface import load_llm as load_hf_llm
from .vector_store import get_embedding_model, load_local_vector_store, get_retriever

# Global caches
_embedding_model = None
_llm_cache = {}  # key: (backend, model_id, use_gpu) → LLM instance


def get_embedder():
    """
    Load and cache the embedding model (singleton).
    """
    global _embedding_model
    if _embedding_model is None:
        print("🔍 Loading embedding model...")
        _embedding_model = get_embedding_model()
    return _embedding_model


def get_llm(
    backend: str = "hf",
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_gpu: bool = torch.cuda.is_available()
):
    """
    Return a cached LLM instance for the given (backend, model_id, use_gpu).
    If not cached, load and cache it.

    Args:
        backend: "hf" → Hugging Face models | "ollama" → Ollama local models
        model_id: Hugging Face model name or Ollama model name
        use_gpu: Applies only to Hugging Face models
    """
    key = (backend.lower(), model_id, bool(use_gpu))
    if key not in _llm_cache:
        print(f"🤖 Loading LLM: backend={backend}, model_id={model_id}, use_gpu={use_gpu}")

        if backend.lower() == "hf":
            # ✅ Load Hugging Face model
            _llm_cache[key] = load_hf_llm(model_id=model_id, use_gpu=use_gpu)

        elif backend.lower() == "ollama":
            # ✅ Load Ollama model
            try:
                from langchain_community.llms import Ollama
            except ImportError:
                raise ImportError(
                    "Ollama backend requested but `langchain-community` is not installed.\n"
                    "Install it with: pip install langchain-community"
                )
            _llm_cache[key] = Ollama(model=model_id)

        else:
            raise ValueError(f"❌ Unknown LLM backend: {backend}")

    else:
        print(f"♻️ Reusing cached LLM: backend={backend}, model_id={model_id}, use_gpu={use_gpu}")

    return _llm_cache[key]


def unload_llm(
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    use_gpu: Optional[bool] = None
):
    """
    Unload LLM instances from cache.
    If no filters provided, clears ALL models.
    For Ollama models, also runs 'ollama stop' to free GPU memory.
    Returns list of removed cache keys.
    """
    removed = []
    keys = list(_llm_cache.keys())

    for key in keys:
        k_backend, k_model_id, k_use_gpu = key

        if backend and k_backend != backend.lower():
            continue
        if model_id and k_model_id != model_id:
            continue
        if use_gpu is not None and k_use_gpu != bool(use_gpu):
            continue

        # Special handling for Ollama backend
        if k_backend == "ollama":
            try:
                subprocess.run(
                    ["ollama", "stop", k_model_id],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"🛑 Ollama model stopped and GPU memory freed: {k_model_id}")
            except Exception as e:
                print(f"⚠️ Failed to stop Ollama model {k_model_id}: {e}")

        _llm_cache.pop(key, None)
        removed.append(key)

    # Free CUDA memory if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return removed


def load_vectorstore(cache_dir: str, top_k: int = 3):
    """
    Load a FAISS vector store and return a retriever.
    Raises FileNotFoundError if index is missing.
    """
    embedder = get_embedder()
    index_file = os.path.join(cache_dir, "index.faiss")

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"❌ No FAISS index found at {cache_dir}")

    vectorstore = load_local_vector_store(cache_dir, embedder)
    return get_retriever(vectorstore, top_k=top_k)
