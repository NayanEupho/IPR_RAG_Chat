import os
import time
import traceback
import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_chatbot.global_objects import get_embedder, get_llm
from rag_chatbot.vector_store import create_vector_store, get_retriever
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.RAG_chain import build_rag_chain, answer_query
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG Chatbot API",
    description="OpenAI-compatible API for querying PDFs using Hugging Face or Ollama models",
    version="1.0.0"
)

# Allow Open WebUI and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
llm = None
retriever = None
_current_load_info = {}

def get_cache_folder(pdfs: list[str], base_folder: str = "vector_cache") -> str:
    import hashlib
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])

@app.get("/")
def root():
    """Root endpoint - OpenAI compatibility check."""
    return {"message": "RAG Chatbot API - OpenAI Compatible"}

@app.get("/v1/models")
def list_models():
    """Lists available models for Open WebUI dropdown."""
    return {
        "object": "list",
        "data": [
            {
                "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "object": "model",
                "owned_by": "huggingface",
                "permissions": [],
                "created": int(time.time())
            },
            {
                "id": "phi3:mini",
                "object": "model", 
                "owned_by": "ollama",
                "permissions": [],
                "created": int(time.time())
            }
        ]
    }

@app.get("/models")
def list_models_alt():
    """Alternative models endpoint for compatibility."""
    return list_models()

@app.post("/load_pdfs")
def load_pdfs_endpoint(payload: dict):
    """Loads PDFs into the vector store and initializes the model."""
    global llm, retriever, _current_load_info

    try:
        pdfs = payload.get("pdfs")
        if not pdfs:
            raise HTTPException(status_code=400, detail="PDF paths required")

        use_gpu = bool(payload.get("use_gpu", False))
        top_k = int(payload.get("top_k", 3))
        model_backend = payload.get("model_backend", "hf")
        model_id = payload.get("model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        logger.info("🧠 Loading or building vector store...")
        embedder = get_embedder()
        cache_dir = get_cache_folder(pdfs)
        index_file = os.path.join(cache_dir, "index.faiss")

        if os.path.exists(index_file):
            logger.info(f"✅ Using cached FAISS index at {cache_dir}")
            vectorstore = FAISS.load_local(
                cache_dir, embeddings=embedder, allow_dangerous_deserialization=True
            )
        else:
            logger.info("📥 Loading PDFs and preparing chunks...")
            pages = load_multiple_pdfs(pdfs)
            chunks = split_text_into_chunks(pages)
            logger.info("⚙️ Creating new index...")
            vectorstore = create_vector_store(chunks, embedder)
            vectorstore.save_local(cache_dir)
            logger.info(f"💾 Vector index saved to: {cache_dir}")

        retriever = get_retriever(vectorstore, top_k=top_k)
        llm = get_llm(backend=model_backend, model_id=model_id, use_gpu=use_gpu)

        _current_load_info = {
            "pdfs": pdfs,
            "use_gpu": use_gpu,
            "top_k": top_k,
            "model_backend": model_backend,
            "model_id": model_id,
            "cache_dir": cache_dir
        }

        return {
            "status": "PDFs loaded successfully",
            "load_info": _current_load_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load PDFs or model: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load PDFs or model: {e}")

@app.post("/v1/chat/completions")
def chat_completions(payload: dict, request: Request):
    """OpenAI-compatible chat completions endpoint for Open WebUI."""
    global llm, retriever

    logger.info("📥 Incoming /v1/chat/completions payload")

    # Check if model is loaded
    if llm is None or retriever is None:
        # Return OpenAI-compatible error response instead of HTTP error
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "local-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": "❌ **Model not loaded**. Please load PDFs first using the `/load_pdfs` endpoint.\n\nTo load PDFs, send a POST request to `/load_pdfs` with:\n```json\n{\n  \"pdfs\": [\"path/to/your/file.pdf\"],\n  \"model_backend\": \"hf\",\n  \"model_id\": \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\"\n}\n```"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 50,
                "total_tokens": 50
            }
        }

    try:
        user_msg = None

        # Extract user message from OpenAI format
        if "messages" in payload and isinstance(payload["messages"], list):
            for m in reversed(payload["messages"]):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    user_msg = m.get("content")
                    break

        # Fallback to prompt field
        if not user_msg and "prompt" in payload:
            user_msg = payload["prompt"]

        if not user_msg or not str(user_msg).strip():
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "local-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant", 
                            "content": "Please provide a question about your loaded PDFs."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10}
            }

        # Get answer from RAG chain
        chain = build_rag_chain(llm, retriever)
        result = answer_query(chain, user_msg)
        answer_text = result.get("answer", "No answer generated.")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", _current_load_info.get("model_id", "local-model")),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_msg.split()) if user_msg else 0,
                "completion_tokens": len(answer_text.split()) if answer_text else 0,
                "total_tokens": len(user_msg.split()) + len(answer_text.split()) if user_msg and answer_text else 0
            }
        }

    except Exception as e:
        logger.error("Error during chat_completions: %s\n%s", e, traceback.format_exc())
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "local-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": f"Error processing your request: {str(e)}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20}
        }

@app.get("/status")
def get_status():
    """Check if the model and retriever are loaded."""
    global llm, retriever, _current_load_info
    
    is_ready = llm is not None and retriever is not None
    
    return {
        "ready": is_ready,
        "model_loaded": llm is not None,
        "retriever_loaded": retriever is not None,
        "load_info": _current_load_info if is_ready else None,
        "message": "Ready to chat!" if is_ready else "Please load PDFs first using /load_pdfs endpoint"
    }

@app.post("/unload_model")
def unload_model(payload: Optional[dict] = None):
    """Unload model & retriever from memory and clear CUDA cache if available."""
    global llm, retriever, _current_load_info
    llm = None
    retriever = None
    _current_load_info = {}

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {"status": "Model and retriever unloaded from memory"}

# Health check endpoint for Open WebUI
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# OpenAI compatibility endpoint
@app.get("/v1")
def openai_v1():
    """OpenAI v1 API base endpoint."""
    return {"message": "OpenAI v1 API compatible endpoint"}