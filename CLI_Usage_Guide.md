# 📘 RAG Chatbot CLI Usage Guide

This guide explains how to use the **CLI-based RAG chatbot** (`CLI_app.py`) with both **Hugging Face models** (e.g., TinyLlama) and **Ollama models** (e.g., phi3:mini).  
You can choose between **GPU** and **CPU** execution without breaking existing functionality.

---

## 1️⃣ Activating the Environment

Before running the CLI app, activate your Python virtual environment:

```bash
venv\Scripts\activate
```

Check if the environment is active:

```bash
where python
```

---

## 2️⃣ CLI Arguments

`CLI_app.py` supports the following arguments:

| Argument         | Type      | Default | Description |
|------------------|-----------|---------|-------------|
| `--pdfs`         | list[str] | **Required** | One or more PDF paths to load. |
| `--top_k`        | int       | 3       | Number of top relevant chunks to retrieve per query. |
| `--use_gpu`      | flag      | False   | Use GPU (only for Hugging Face backend). |
| `--show_chunks`  | flag      | False   | Show metadata for retrieved chunks in the terminal. |
| `--model_backend`| str       | hf      | LLM backend to use (`hf` for Hugging Face, `ollama` for Ollama). |
| `--model_id`     | str       | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Model name (HF model ID or Ollama model name). |

---

## 3️⃣ Example Commands

### **A) Hugging Face TinyLlama (GPU)**

```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks --model_backend hf --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### **B) Hugging Face TinyLlama (CPU)**

```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --show_chunks --model_backend hf --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### **C) Ollama phi3:mini (CPU)**

```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --show_chunks --model_backend ollama --model_id phi3:mini
```

---

## 4️⃣ Cached FAISS Index

The CLI app caches vector indexes in the `vector_cache/` folder.  
When loading the same PDF again without changes, it **reuses the cached FAISS index** instead of reprocessing.

- Cache folder is generated using a **hash** of the PDF contents.
- This greatly speeds up repeated runs.

---

## 5️⃣ Showing Retrieved Chunks

Use `--show_chunks` to display metadata for retrieved chunks:

```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --show_chunks --model_backend hf --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Sample Output:

```
🔍 Top 3 chunks retrieved:
  1. Page 1, Type: text, Source: sample1.pdf
  2. Page 2, Type: table, Source: sample1.pdf
  3. Page 3, Type: text, Source: sample1.pdf
```

---

## 6️⃣ Interactive Session

Once the CLI starts, you can ask questions:

```
💬 Enter your questions (type 'exit' to quit)

>>> What is this PDF about?
🧠 Answer: This is an e-receipt for an online driving license application...
⏱️  Took: 3.57 seconds

>>> exit
👋 Exiting interactive session.
```

---

## 7️⃣ Tips & Notes

- `--use_gpu` works **only for Hugging Face models**.
- `ollama` backend always runs on **CPU** unless your Ollama setup supports GPU acceleration.
- The PDF loader handles **multiple PDFs** at once:  
  ```bash
  python CLI_app.py --pdfs sample_pdfs/doc1.pdf sample_pdfs/doc2.pdf --model_backend hf
  ```
- If you change PDFs, the cache will be rebuilt automatically.

---

✅ You are now ready to use the **CLI-based RAG Chatbot** with both Hugging Face and Ollama models!
