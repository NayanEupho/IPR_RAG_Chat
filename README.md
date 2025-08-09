# 🧠 RAG PDF Chatbot with CLI & GUI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![LLM](https://img.shields.io/badge/Model-TinyLlama-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)
![Platform](https://img.shields.io/badge/Platform-CLI%20%26%20GUI-purple)

> A **Retrieval-Augmented Generation (RAG)** chatbot that lets you query local PDF documents using natural language — via CLI or GUI.

---

## 🚀 Features
- 📄 Upload and chat with PDF documents
- 🗂 Uses **FAISS** for local vector storage
- 🤖 **TinyLlama** for local LLM inference
- ⚙️ Supports **CPU/GPU** via CLI flag
- ⚡ Caches embeddings for faster reuse

---

## ⚡ Quick Start (Diagram)

```mermaid
graph TD;
    A[📄 PDF Files] --> B[🔍 FAISS Embedding Store];
    B --> C[🧠 LLM (TinyLlama)];
    C --> D[💬 CLI / 🌐 Open WebUI];
```

---

## 🧪 CLI Usage

### 1️⃣ Activate Virtual Environment
```bash
# Windows
source venv/Scripts/activate

# macOS / Linux
source venv/bin/activate
```

### 2️⃣ Query via CLI
```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks
```

---

## 📝 Important Notes

### 📌 Open WebUI Connection Settings
> **API URL:** `http://host.docker.internal:8000`  
> **Key:**  your-dummy-key (type anything you want here)

---

### 📌 Ollama (Phi3:mini)
> If using Ollama model: **`phi3:mini`** then make sure to execute this model file in Ollama during setup:  
> ```bash
> ollama create phi3:mini -f Modelfile
> ```  
> Then restart Ollama:  
> ```bash
> ollama stop
> ollama serve
> ```

---

### 📌 Open WebUI using TinyLlama / TinyLlama-1.1B-Chat-v1.0
> If you want to use **TinyLlama/TinyLlama-1.1B-Chat-v1.0**, first start the server:  
> ```bash
> uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
> ```  
> Next, go to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** in your browser, navigate to `/load_pdfs`, click **"Try it out"**, and execute this JSON in the request body:  
> ```json
> {
>     "pdfs": ["sample_pdfs/sample1.pdf"],
>     "use_gpu": true,
>     "top_k": 3,
>     "model_backend": "hf",
>     "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
> }
> ```  
> Now go to **Open WebUI** and start chatting.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 💡 Tip
For better performance, store embeddings locally and reuse them instead of regenerating each time.
