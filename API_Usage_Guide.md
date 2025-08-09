
# RAG Chatbot API - Usage Guide

This document explains how to use the `api_server.py` RAG Chatbot API for querying PDF documents with Hugging Face or Ollama models.  
It includes **features**, **endpoints**, and **examples**.

---

## 🚀 Overview

The API allows you to:
- Load one or more PDF documents into a FAISS vector store.
- Select a language model backend (**Hugging Face** or **Ollama**).
- Query the PDFs in **OpenAI API-compatible format** (works with Open WebUI).
- Unload models to free GPU/CPU memory.

---

## 📌 Features

- **Multiple Backends**:
  - Hugging Face models (`hf`).
  - Ollama local models (`ollama`).
- **Persistent Caching**:
  - Automatically caches FAISS vector indexes for faster reloads.
- **Configurable Retrieval**:
  - Choose `top_k` relevant chunks.
- **OpenAI API Compatibility**:
  - Works with Open WebUI and similar tools.
- **GPU Support** (Hugging Face backend).
- **Chunk Debugging**:
  - Optional retrieval chunk logging.

---

## ⚙️ API Endpoints

### 1. `GET /v1/models`
**Description:** Returns available models for selection.

**Example Request:**
```bash
curl -X GET http://localhost:8000/v1/models
```

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "object": "model",
      "owned_by": "huggingface",
      "permissions": []
    },
    {
      "id": "phi3:mini",
      "object": "model",
      "owned_by": "ollama",
      "permissions": []
    }
  ]
}
```

---

### 2. `POST /load_pdfs`
**Description:** Loads PDFs into a FAISS vector store and initializes the chosen model.

**Example Request:**
```bash
curl -X POST http://localhost:8000/load_pdfs   -H "Content-Type: application/json"   -d '{
    "pdfs": ["sample_pdfs/sample1.pdf"],
    "use_gpu": true,
    "top_k": 3,
    "model_backend": "hf",
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  }'
```

**Example Response:**
```json
{
  "status": "PDFs loaded successfully",
  "load_info": {
    "pdfs": ["sample_pdfs/sample1.pdf"],
    "use_gpu": true,
    "top_k": 3,
    "model_backend": "hf",
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "cache_dir": "vector_cache/abcd1234"
  }
}
```

---

### 3. `POST /v1/chat/completions`
**Description:** Queries the loaded PDFs and returns a model-generated answer.

**Example Request:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is this document about?"}
    ],
    "show_chunks": true
  }'
```

**Example Response:**
```json
{
  "id": "chatcmpl-1693456",
  "object": "chat.completion",
  "created": 1693456789,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This PDF discusses..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "sources": [
    {"page": 1, "source": "sample_pdfs/sample1.pdf"}
  ]
}
```

---

### 4. `POST /unload_model`
**Description:** Unloads the model and retriever from memory.

**Example Request:**
```bash
curl -X POST http://localhost:8000/unload_model
```

**Example Response:**
```json
{
  "status": "Model and retriever unloaded from memory"
}
```

---

## 🔍 Parameters

| Parameter        | Type    | Description |
|------------------|---------|-------------|
| `pdfs`           | list    | Paths to one or more PDFs. |
| `use_gpu`        | bool    | Use GPU (HF backend only). |
| `top_k`          | int     | Number of top chunks to retrieve. |
| `model_backend`  | string  | `"hf"` for Hugging Face, `"ollama"` for Ollama. |
| `model_id`       | string  | Model name (Hugging Face or Ollama). |
| `show_chunks`    | bool    | Log top retrieved chunks. |

---

## 🖥 Running the API Server

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

3. Access in browser:
```
http://localhost:8000/docs
```

---

## 🧩 Integration with Open WebUI

- Point Open WebUI to `http://localhost:8000`.
- The `/v1/chat/completions` endpoint is OpenAI-compatible.
- Models listed from `/v1/models` will appear in the dropdown.

---

## 📌 Notes
- Ensure your PDF paths are **absolute or relative to the API working directory**.
- If using Ollama, make sure Ollama is installed and running:
```bash
ollama run phi3:mini
```
- Cached FAISS indexes are stored in `vector_cache/`.
