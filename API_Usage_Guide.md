# 📘 RAG Chatbot API Usage Guide

This guide explains how to use your **FastAPI-powered RAG chatbot API** with your **local model** (via LangChain), including:
- Starting the API server
- Loading PDFs into the vector store
- Asking questions
- Using GPU or CPU
- Overriding device per query
- Viewing retrieved chunks (debug mode)

---

## 1️⃣ Start the API Server

From your project root, run:

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Flags explained:**
- `--reload` → Auto-restart the server when you edit files
- `--host 0.0.0.0` → Listen on all network interfaces (for local or remote access)
- `--port 8000` → API will run on port 8000

Once running, the API is available at:
```
http://localhost:8000
```

---

## 2️⃣ Open the Interactive API Docs

FastAPI automatically generates a Swagger UI:
```
http://localhost:8000/docs
```

From here, you can:
- Test endpoints without using curl or Postman
- See request/response formats
- Execute API calls directly in the browser

---

## 3️⃣ Load PDFs into the Vector Store

Before asking questions, you must **load your PDFs**.

In `/docs`, find the **`POST /load_pdfs`** endpoint → click **Try it out**.

Example JSON for **GPU**:
```json
{
  "pdfs": ["sample_pdfs/sample1.pdf"],
  "use_gpu": true,
  "top_k": 3
}
```

Example JSON for **CPU**:
```json
{
  "pdfs": ["sample_pdfs/sample1.pdf"],
  "use_gpu": false,
  "top_k": 3
}
```

**Fields:**
- `pdfs` → List of one or more PDF file paths
- `use_gpu` → `true` for GPU, `false` for CPU
- `top_k` → Number of top chunks to retrieve per query

**Example response:**
```json
{
  "status": "PDFs loaded successfully",
  "pdfs": ["sample_pdfs/sample1.pdf"],
  "use_gpu": false,
  "top_k": 3
}
```

---

## 4️⃣ Ask a Question (Default Behavior)

In `/docs`, find the **`POST /v1/chat/completions`** endpoint → click **Try it out**.

Example JSON:
```json
{
  "model": "local-llama",
  "messages": [
    { "role": "user", "content": "What is this PDF about?" }
  ],
  "temperature": 0.0,
  "show_chunks": true
}
```

**Fields:**
- `model` → Any string (OpenAI-compatible format requirement). Doesn’t affect which model is used.
- `messages` → List of messages in conversation format. Your **question** goes in the `user` message.
- `temperature` → Controls creativity (0.0 = more factual, 1.0 = more creative)
- `show_chunks` → If `true`, prints chunk metadata in the **server logs** for debugging.

---

## 5️⃣ Per-Query Device Override

Even after loading PDFs with a certain device setting, you can override it **for one query only**.

### 🔹 Force GPU for a single query
```json
{
  "model": "local-llama",
  "messages": [
    { "role": "user", "content": "Summarize the document." }
  ],
  "temperature": 0.0,
  "show_chunks": false,
  "use_gpu": true
}
```

### 🔹 Force CPU for a single query
```json
{
  "model": "local-llama",
  "messages": [
    { "role": "user", "content": "Summarize the document." }
  ],
  "temperature": 0.0,
  "show_chunks": false,
  "use_gpu": false
}
```

If `use_gpu` is **not provided**, the query uses the **default device** from `/load_pdfs`.

---

## 6️⃣ Using curl Instead of `/docs`

You can call the API from the terminal using `curl`.

**Load PDFs (CPU):**
```bash
curl -X POST "http://localhost:8000/load_pdfs" \
-H "Content-Type: application/json" \
-d "{ \"pdfs\": [\"sample_pdfs/sample1.pdf\"], \"use_gpu\": false }"
```

**Ask a Question with GPU override:**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d "{ \"model\": \"local-llama\", \"messages\": [ { \"role\": \"user\", \"content\": \"What is this PDF about?\" } ], \"temperature\": 0.0, \"show_chunks\": true, \"use_gpu\": true }"
```

---

## 7️⃣ Unloading the Model

To free GPU/CPU memory without stopping the API, call POST /unload_model:
**Unload request**

{
  "unload": true
}

Response:

{
  "status": "Model and retriever unloaded from memory"
}

After unloading, if you query, you’ll get:

{"detail": "No model loaded. Please call /load_pdfs first."}



## Notes & Tips
- The API uses **cached FAISS indexes** for faster reloads — if the PDF content hasn’t changed, it reuses the existing vector store.
- `show_chunks` only prints to the **server console**, not in the JSON response (keeps responses clean).
- You can reload the same or different PDFs anytime without restarting the server.
- `use_gpu` in `/v1/chat/completions` lets you **temporarily** switch devices without reloading PDFs.
- /unload_model clears model from memory without stopping the API.
