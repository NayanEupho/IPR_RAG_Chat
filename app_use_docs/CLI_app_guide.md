# 🖥️ CLI Usage Guide: `CLI_app.py` - RAG Chatbot (PDF-based)

This guide walks you through running the `CLI_app.py` interactive chatbot that uses Retrieval-Augmented Generation (RAG) over PDF files.

---

## 📦 Prerequisites

Make sure you've installed all required packages:

```bash
pip install -r requirements.txt
```

> ⚠️ If you're using a GPU, make sure PyTorch with CUDA is installed:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 What It Does

- Accepts one or more PDF files
- Builds or loads a FAISS vector store index
- Uses a local LLM (e.g. TinyLlama) for answering questions
- Runs in an interactive command-line loop
- Caches the model and index for faster future runs

---

## 🔧 CLI Usage

```bash
python CLI_app.py --pdfs <path(s)> [options]
```

### ✅ Required Flags

| Flag       | Description                                      | Example                                 |
|------------|--------------------------------------------------|-----------------------------------------|
| `--pdfs`   | Path(s) to one or more PDFs                      | `--pdfs sample_pdfs/sample1.pdf`        |

### ⚙️ Optional Flags

| Flag              | Description                                               | Default |
|-------------------|-----------------------------------------------------------|---------|
| `--top_k`         | Number of top matching chunks to retrieve                | 3       |
| `--use_gpu`       | Use GPU if available                                      | False   |
| `--force_rebuild` | Force rebuild of vector index, even if cache exists       | False   |

---

## 💬 Example

```bash
python CLI_app.py --pdfs sample_pdfs/sample1.pdf --top_k 4 --use_gpu
```

---

## 💡 Features

- ✅ Loads model only once and keeps in memory
- ✅ Uses vector index cache if available
- ✅ Answers queries in real-time
- ✅ Allows multiple questions until user types `exit`
- ✅ Shows time taken for each query
- ✅ Automatically uses GPU if `--use_gpu` is specified and available

---

## 🧠 Example Interaction

```text
🧠 Loading or building vector store...
✅ Using cached FAISS index at vector_cache\f1aa2f14
🚀 Loading LLM...
🚀 Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
🖥️  Device: CUDA
✅ Model is using GPU: NVIDIA GeForce RTX 4050 Laptop GPU

💬 Enter your questions (type 'exit' to quit)

>>> What is the applicant's name?

🧠 Answer: Bharavi Modi
⏱️  Took: 1.32 seconds

>>> exit
👋 Exiting interactive session.
```

---

## 🗂️ Output Caching

- Vector indexes are saved in `vector_cache/` based on the hash of PDF contents.
- If content doesn't change, future runs will load instantly.

---

## 📌 Notes

- This CLI app is great for local workflows, development, or terminal-based use.
- If you'd like to add a GUI or API server later, you can reuse the logic in this app.

---

Happy Querying! 📄🤖✨
