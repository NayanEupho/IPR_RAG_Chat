import os
import time
import traceback
import logging
from typing import Optional, List
import hashlib
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import base64

# Download required NLTK data on startup
import nltk
from nltk.data import find

def ensure_nltk_data():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    try:
        # Check if all resources are available
        all_present = True
        for resource_path, _ in resources:
            try:
                find(resource_path)
            except LookupError:
                all_present = False
                break

        if all_present:
            print("✅ NLTK data available")
            return

        print("❌ NLTK data not available, starting NLTK data download...")
        
        # Attempt to download missing resources
        for _, resource_name in resources:
            nltk.download(resource_name, quiet=True)
        
        # Verify again
        for resource_path, _ in resources:
            find(resource_path)
        
        print("✅ NLTK data downloaded successfully")

    except Exception as e:
        print(f"⚠️ Warning: Failed to download NLTK data: {e}")

# Run check
ensure_nltk_data()


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
uploaded_files_dir = Path("uploaded_pdfs")
uploaded_files_dir.mkdir(exist_ok=True)

def get_cache_folder(pdfs: list[str], base_folder: str = "vector_cache") -> str:
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])

def initialize_rag_system(pdfs: List[str], model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Initialize the RAG system with uploaded PDFs and the selected model."""
    global llm, retriever, _current_load_info
    
    try:
        # Determine model backend from model_id
        if "ollama" in model_id.lower() or "phi3" in model_id.lower():
            model_backend = "ollama"
            use_gpu = False  # Ollama handles GPU internally
        else:
            model_backend = "hf"
            use_gpu = torch.cuda.is_available()
        
        logger.info(f"🧠 Initializing RAG system with {len(pdfs)} PDFs and model: {model_id}")
        
        embedder = get_embedder()
        cache_dir = get_cache_folder(pdfs)
        index_file = os.path.join(cache_dir, "index.faiss")

        if os.path.exists(index_file):
            logger.info(f"✅ Using cached FAISS index at {cache_dir}")
            vectorstore = FAISS.load_local(
                cache_dir, embeddings=embedder, allow_dangerous_deserialization=True
            )
        else:
            logger.info("🔥 Loading PDFs and preparing chunks...")
            pages = load_multiple_pdfs(pdfs)
            chunks = split_text_into_chunks(pages)
            logger.info("⚙️ Creating new index...")
            vectorstore = create_vector_store(chunks, embedder)
            vectorstore.save_local(cache_dir)
            logger.info(f"💾 Vector index saved to: {cache_dir}")

        retriever = get_retriever(vectorstore, top_k=3)
        llm = get_llm(backend=model_backend, model_id=model_id, use_gpu=use_gpu)

        _current_load_info = {
            "pdfs": pdfs,
            "use_gpu": use_gpu,
            "top_k": 3,
            "model_backend": model_backend,
            "model_id": model_id,
            "cache_dir": cache_dir
        }
        
        logger.info("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False

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

# NEW: File upload endpoint for Open WebUI
@app.post("/v1/files")
async def upload_files(files: List[UploadFile] = File(...), purpose: str = Form("assistants")):
    """Handle file uploads from Open WebUI - OpenAI Files API compatible."""
    try:
        uploaded_files = []
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Got: {file.filename}"
                )
            
            # Save uploaded file
            file_path = uploaded_files_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append({
                "id": f"file-{hashlib.md5(file.filename.encode()).hexdigest()[:12]}",
                "object": "file",
                "bytes": len(content),
                "created_at": int(time.time()),
                "filename": file.filename,
                "purpose": purpose,
                "status": "processed",
                "path": str(file_path)
            })
        
        logger.info(f"📁 Uploaded {len(uploaded_files)} PDF file(s)")
        
        return {
            "object": "list",
            "data": uploaded_files
        }
            
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: List uploaded files
@app.get("/v1/files")
def list_files():
    """List all uploaded PDF files - OpenAI Files API compatible."""
    try:
        files = []
        if uploaded_files_dir.exists():
            for file_path in uploaded_files_dir.glob("*.pdf"):
                stat = file_path.stat()
                files.append({
                    "id": f"file-{hashlib.md5(file_path.name.encode()).hexdigest()[:12]}",
                    "object": "file",
                    "bytes": stat.st_size,
                    "created_at": int(stat.st_ctime),
                    "filename": file_path.name,
                    "purpose": "assistants",
                    "status": "processed",
                    "path": str(file_path)
                })
        
        return {
            "object": "list",
            "data": files
        }
    except Exception as e:
        logger.error(f"❌ Failed to list files: {e}")
        return {"object": "list", "data": []}

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
            logger.info("🔥 Loading PDFs and preparing chunks...")
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

    logger.info("🔥 Incoming /v1/chat/completions payload")

    # Check if model is loaded, if not, try to auto-initialize with uploaded PDFs
    if llm is None or retriever is None:
        logger.info("🔄 Model not loaded, checking for uploaded PDFs...")
        
        # Get model from payload
        requested_model = payload.get("model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Check for uploaded PDFs
        pdf_files = []
        if uploaded_files_dir.exists():
            pdf_files = [str(f) for f in uploaded_files_dir.glob("*.pdf")]
        
        if pdf_files:
            logger.info(f"📚 Found {len(pdf_files)} uploaded PDFs, initializing RAG system...")
            success = initialize_rag_system(pdf_files, requested_model)
            
            if not success:
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant", 
                                "content": "❌ **Failed to initialize RAG system**. Please try uploading your PDFs again or check the server logs for errors."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20}
                }
        else:
            # No PDFs found
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant", 
                            "content": "📁 **No PDFs uploaded yet**. Please upload PDF files using the file upload feature in Open WebUI, then ask your question again.\n\n*Tip: Look for the attachment/paperclip icon in the chat interface to upload files.*"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 30, "total_tokens": 30}
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
    
    # Count uploaded PDFs
    pdf_count = 0
    if uploaded_files_dir.exists():
        pdf_count = len(list(uploaded_files_dir.glob("*.pdf")))
    
    return {
        "ready": is_ready,
        "model_loaded": llm is not None,
        "retriever_loaded": retriever is not None,
        "uploaded_pdfs_count": pdf_count,
        "load_info": _current_load_info if is_ready else None,
        "message": "Ready to chat!" if is_ready else f"Please upload PDFs first. Found {pdf_count} uploaded PDFs."
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

# NEW: Clear uploaded files
@app.post("/clear_uploads")
def clear_uploads():
    """Clear all uploaded PDF files."""
    global llm, retriever, _current_load_info
    
    try:
        # Unload model first
        llm = None
        retriever = None
        _current_load_info = {}
        
        # Clear uploaded files
        if uploaded_files_dir.exists():
            for file_path in uploaded_files_dir.glob("*.pdf"):
                file_path.unlink()
        
        logger.info("🗑️ Cleared all uploaded files and unloaded models")
        return {"status": "All uploaded files cleared and models unloaded"}
        
    except Exception as e:
        logger.error(f"❌ Failed to clear uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# ---------- NEW: Serve a default favicon to avoid 404s ----------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Minimal transparent 1x1 PNG
    favicon_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    return Response(
        content=base64.b64decode(favicon_base64),
        media_type="image/png"
    )