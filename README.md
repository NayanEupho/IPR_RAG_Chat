# 🧠 RAG PDF Chatbot with CLI & GUI

This is a Retrieval-Augmented Generation (RAG) chatbot that lets you query local PDF documents using natural language on CLI.

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

## NOTE on Open WebUI connection settings:
> API URL: http://host.docker.internal:8000
> Key:  your-dummy-key (any dummy key, basically just type whatever you want)

## NOTE for Ollama Phi3:mini -:
> If using Ollama model: phi3:mini then make sure to execute this model file in Ollama at the set-up:
"""ollama create phi3:mini -f Modelfile"""
> Then re-start Ollama with "ollama stop" and the "ollama serve"

## Note for Open WebUI using TinyLlama/TinyLlama-1.1B-Chat-v1.0 -:
> If what to use TinyLlama/TinyLlama-1.1B-Chat-v1.0 the 1st start the server using "uvicorn api_server:app --reload --host 0.0.0.0 --port 8000"
> Next got to http://127.0.0.1:8000/docs on browser and then go to "/load_pdfs" and execute this JSON by clicking on "Try it out" and executing this in "request body": 
"""{
    "pdfs": ["sample_pdfs/sample1.pdf"],
    "use_gpu": true,
    "top_k": 3,
    "model_backend": "hf",
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  }
  """
> Now go to Open WebUI and start chating.