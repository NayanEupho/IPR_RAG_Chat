##Start the API server
"""
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

## Load PDfs by deafult (happens with CPU)
"""
curl -X POST "http://localhost:8000/load_pdfs" ^
-H "Content-Type: application/json" ^
-d "{ \"pdfs\": [\"sample_pdfs/sample1.pdf\"] }"
"""

## Load PDFs with CPU
"""
curl -X POST "http://localhost:8000/load_pdfs" ^
-H "Content-Type: application/json" ^
-d "{ \"pdfs\": [\"sample_pdfs/sample1.pdf\"], \"use_gpu\": false }"
"""

## Load PDFs with GPU
"""
curl -X POST "http://localhost:8000/load_pdfs" ^
-H "Content-Type: application/json" ^
-d "{ \"pdfs\": [\"sample_pdfs/sample1.pdf\"], \"use_gpu\": true }"
"""

## Ask a question
"""
curl -X POST "http://localhost:8000/v1/chat/completions" ^
-H "Content-Type: application/json" ^
-d "{ \"model\": \"gpt-4\", \"messages\": [ { \"role\": \"user\", \"content\": \"What is this PDF about?\" } ] }"
"""

## Send a question & show retrieved chunks in API logs:
"""
curl -X POST "http://localhost:8000/v1/chat/completions" ^
-H "Content-Type: application/json" ^
-d "{ \"model\": \"gpt-4\", \"messages\": [ { \"role\": \"user\", \"content\": \"What is this PDF about?\" } ], \"show_chunks\": true }"
"""

## Override per query
"""
    Pass "use_gpu": true in /v1/chat/completions → that single request uses GPU.

    Pass "use_gpu": false in /v1/chat/completions → that single request uses CPU.

No breaking:
    If you omit "use_gpu" in query request, it behaves exactly like before.
"""

## Example per-query GPU override
"""
POST /v1/chat/completions
{
  "model": "local-llama",
  "messages": [
    { "role": "user", "content": "What is this PDF about?" }
  ],
  "temperature": 0.0,
  "show_chunks": true,
  "use_gpu": true
}

✅ Even if /load_pdfs was called with "use_gpu": false, this query will run on GPU.
"""

## Example per-query GPU override
"""
POST /v1/chat/completions
{
  "model": "local-llama",
  "messages": [
    { "role": "user", "content": "What is this PDF about?" }
  ],
  "temperature": 0.0,
  "show_chunks": true,
  "use_gpu": false
}

✅ Even if /load_pdfs was called earlier with "use_gpu": true, this query will force the model to run on CPU for this request only, without unloading the GPU version for other queries.
"""
import os
import time  # <-- added for timestamp
import gc    # <-- added for garbage collection
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- added for CORS
from pydantic import BaseModel
from typing import List, Optional
from rag_chatbot.vector_store import create_vector_store, get_retriever
from rag_chatbot.global_objects import get_embedder, get_llm
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.RAG_chain import build_rag_chain, answer_query

from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Chatbot API", version="1.0")

# Add CORS middleware to allow requests from any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your OpenWebUI domain if you want stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
llm = None
retriever = None

class LoadPDFsRequest(BaseModel):
    pdfs: List[str]
    use_gpu: bool = False
    top_k: int = 3

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    show_chunks: bool = False
    use_gpu: Optional[bool] = None  # Per-query override

@app.post("/load_pdfs")
def load_pdfs(request: LoadPDFsRequest):
    global llm, retriever
    embedder = get_embedder()
    cache_dir = get_cache_folder(request.pdfs)
    index_file = os.path.join(cache_dir, "index.faiss")

    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(
            cache_dir, embeddings=embedder, allow_dangerous_deserialization=True
        )
    else:
        pages = load_multiple_pdfs(request.pdfs)
        chunks = split_text_into_chunks(pages)
        vectorstore = create_vector_store(chunks, embedder)
        vectorstore.save_local(cache_dir)

    retriever = get_retriever(vectorstore, top_k=request.top_k)
    llm = get_llm(use_gpu=request.use_gpu)

    return {
        "status": "PDFs loaded successfully",
        "pdfs": request.pdfs,
        "use_gpu": request.use_gpu,
        "top_k": request.top_k
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    global llm, retriever

    if retriever is None or llm is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please call /load_pdfs first.")

    local_llm = llm
    if request.use_gpu is not None:
        local_llm = get_llm(use_gpu=request.use_gpu)

    query = next((m.content for m in request.messages if m.role == "user"), None)
    if not query:
        raise HTTPException(status_code=400, detail="No user message found.")

    chain = build_rag_chain(local_llm, retriever)
    result = answer_query(chain, query)

    # Explicit cleanup of chain
    del chain
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop"
            }
        ],
        "sources": result["sources"]
    }


@app.post("/unload_model")
def unload_model(unload: bool = True):
    global llm, retriever
    if unload:
        if llm is not None:
            del llm
        if retriever is not None:
            del retriever

        llm = None
        retriever = None

        gc.collect()  # Force Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status": "Model and retriever unloaded from memory"}
    else:
        return {"status": "Unload flag not set, nothing done"}

def get_cache_folder(pdfs: List[str], base_folder: str = "vector_cache") -> str:
    import hashlib
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])
