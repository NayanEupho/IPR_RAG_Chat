# 🧠 RAG PDF Chatbot with CLI & GUI

This is a Retrieval-Augmented Generation (RAG) chatbot that lets you query local PDF documents using natural language. It supports CLI and a Streamlit-based GUI.

## 🚀 Features
- Upload and chat with PDF documents
- Uses FAISS for local vector storage
- TinyLlama for local LLM inference
- Supports CPU/GPU via CLI flag
- Caches embeddings for fast reuse

## 🧪 CLI Usage

```bash
# Activate venv
source venv/Scripts/activate   # Windows
# or
source venv/bin/activate       # macOS/Linux

# Query via CLI
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks
