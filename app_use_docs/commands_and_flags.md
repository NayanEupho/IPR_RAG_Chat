
# 📘 Usage Guide: `main_2.py` - RAG Chatbot CLI

This guide explains how to use the `main_2.py` command-line interface to query PDFs using your RAG-based chatbot.

---

## ✅ Available Flags

| Flag                | Type        | Required | Description |
|---------------------|-------------|----------|-------------|
| `--pdfs`            | List[str]   | ✅ Yes   | One or more paths to PDF files to query. |
| `--query`           | str         | ✅ Yes   | The question or instruction to ask the chatbot. |
| `--top_k`           | int         | ❌ No    | Number of top relevant document chunks to retrieve (Default: 3). |
| `--save_source`     | str         | ❌ No    | Path to save the source traceability as a JSON file. |
| `--use_gpu`         | flag        | ❌ No    | Enables GPU acceleration for LLM inference (if available). |   

---

## 🚀 Example Commands

### 1️⃣ Basic Query on a Single PDF

```
python main_2.py --pdfs sample_pdfs/sample1.pdf --query "What is the conclusion of this PDF?"
```

- Loads `sample1.pdf`
- Retrieves top 3 relevant text chunks
- Uses TinyLlama to generate the answer

---

### 2️⃣ Query on Multiple PDFs

```
python main_2.py --pdfs sample_pdfs/a.pdf sample_pdfs/b.pdf --query "Summarize the main findings."
```

- Combines chunks from both `a.pdf` and `b.pdf`

---

### 3️⃣ Custom Number of Chunks

```
python main_2.py --pdfs sample_pdfs/sample1.pdf --query "List all important dates." --top_k 5
```

- Retrieves top 5 relevant chunks for more context

---

### 4️⃣ Save Source Info as JSON

```
python main_2.py --pdfs sample_pdfs/sample1.pdf --query "What fees are mentioned?" --save_source source_info.json
```

- Saves source document data used in the answer to `source_info.json`

Example output:

```json
{
  "query": "What fees are mentioned?",
  "used_documents": [
    {
      "index": 1,
      "source": "sample_pdfs/sample1.pdf",
      "page": 2,
      "chunk_text": "..."
    }
  ]
}
```

---

### 5️⃣ Use GPU for Inference

```
python main_2.py --pdfs sample_pdfs/sample1.pdf --query "Summarize the document." --use_gpu
```

- Runs the LLM on your GPU instead of CPU (if compatible)
- Improves performance and reduces response time

---

### 6️⃣ Full Featured Command

```
python main_2.py --pdfs sample_pdfs/sample1.pdf --query "Summarize the payment information." --top_k 4 --save_source summary_trace.json --use_gpu
```

- Retrieves 4 chunks
- Answers the question
- Saves a JSON traceability file
- Uses GPU for faster inference

---

## 💬 Flag Summary

| Flag            | Purpose                                                |
|-----------------|--------------------------------------------------------|
| `--pdfs`        | Specifies PDF(s) to process                            |
| `--query`       | The user question                                      |
| `--top_k`       | Number of top relevant chunks to pass to the model     |
| `--save_source` | Output file for saving source chunks as JSON trace     |
---