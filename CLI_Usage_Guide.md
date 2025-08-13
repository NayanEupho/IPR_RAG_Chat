# 📘 RAG Chatbot CLI - Complete Usage Guide

A comprehensive guide for using the **CLI-based RAG chatbot** (`CLI_app.py`) with advanced features including **multi-model support**, **intelligent caching**, and **interactive querying**.

---

## 🚀 Overview

The CLI app provides a **terminal-based interface** for querying PDF documents using RAG (Retrieval-Augmented Generation). It supports multiple backends, GPU acceleration, and offers an interactive chat experience.

### ✨ **Key Features**
- 📄 **Multi-PDF Support** - Load and query multiple documents simultaneously
- 🧠 **Dual Backend Support** - Hugging Face Transformers & Ollama
- ⚡ **GPU Acceleration** - CUDA support for faster inference (HF models)
- 🔄 **Smart Caching** - FAISS vector store caching for instant reloads
- 💬 **Interactive Chat** - Real-time Q&A with your documents
- 🔍 **Debug Mode** - View retrieved chunks and metadata
- 🛑 **Graceful Cleanup** - Automatic model unloading and resource management

---

## 🔧 Installation & Setup

### **1️⃣ Environment Activation**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
where python  # Windows
which python  # macOS/Linux
```

### **2️⃣ Dependencies Check**
Ensure all required packages are installed:
```bash
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **3️⃣ NLTK Data Setup** (Required)
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

---

## ⚙️ Command Line Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--pdfs` | list[str] | - | ✅ | **Path(s) to PDF file(s)** - supports multiple files |
| `--model_backend` | str | `hf` | ❌ | **Backend**: `hf` (Hugging Face) or `ollama` |
| `--model_id` | str | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ❌ | **Model identifier** - HF model name or Ollama model |
| `--top_k` | int | `3` | ❌ | **Retrieval count** - number of relevant chunks |
| `--use_gpu` | flag | `False` | ❌ | **GPU acceleration** - HF backend only |
| `--show_chunks` | flag | `False` | ❌ | **Debug mode** - display retrieved chunks |

---

## 🧪 Usage Examples

### **🤖 Hugging Face Models**

#### **TinyLlama with GPU** (Recommended)
```bash
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend hf \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --use_gpu \
  --show_chunks \
  --top_k 5
```

#### **TinyLlama with CPU** (Slower but works everywhere)
```bash
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend hf \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --show_chunks
```

#### **Alternative HF Models**
```bash
# Microsoft DialoGPT
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend hf \
  --model_id microsoft/DialoGPT-medium \
  --use_gpu

# Llama 2 7B (requires more VRAM)
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend hf \
  --model_id meta-llama/Llama-2-7b-chat-hf \
  --use_gpu
```

### **🦙 Ollama Models**

#### **Phi3 Mini** (Recommended Ollama model)
```bash
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend ollama \
  --model_id phi3:mini \
  --show_chunks
```

#### **Other Ollama Models**
```bash
# Llama 3.1
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend ollama \
  --model_id llama3.1:8b

# Code Llama
python CLI_app.py \
  --pdfs sample_pdfs/document.pdf \
  --model_backend ollama \
  --model_id codellama:7b
```

### **📚 Multiple PDF Processing**
```bash
python CLI_app.py \
  --pdfs document1.pdf document2.pdf research_paper.pdf \
  --model_backend hf \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --use_gpu \
  --top_k 7
```

---

## 💾 Smart Caching System

### **How It Works**
- **Hash-based Caching**: Creates unique cache folders based on PDF content
- **Instant Reloads**: Subsequent runs with same PDFs load instantly
- **Content-Aware**: Automatically rebuilds cache when PDFs change
- **Storage Location**: `vector_cache/` directory

### **Cache Structure**
```
vector_cache/
├── abc12345/          # Hash of PDF contents
│   ├── index.faiss    # Vector embeddings
│   └── index.pkl      # Metadata
└── def67890/          # Another document set
    ├── index.faiss
    └── index.pkl
```

### **Cache Management**
```bash
# View cache folders
ls vector_cache/

# Clear all caches
rm -rf vector_cache/

# Check cache size
du -sh vector_cache/
```

---

## 🔍 Debug Mode Features

### **Enable Debug Mode**
```bash
python CLI_app.py --pdfs document.pdf --show_chunks
```

### **Debug Output Example**
```
🔍 Top 3 chunks retrieved:
  1. Page 1, Type: text, Source: document.pdf
     Score: 0.892, Characters: 1,247
     Content: "This section discusses the implementation of..."

  2. Page 3, Type: table, Source: document.pdf  
     Score: 0.856, Characters: 892
     Content: "Table 2: Performance metrics showing..."

  3. Page 2, Type: text, Source: document.pdf
     Score: 0.823, Characters: 1,156
     Content: "The methodology follows a systematic approach..."
```

### **Debug Information**
- **Page Number**: Source page in PDF
- **Content Type**: text, table, figure, etc.
- **Relevance Score**: Similarity score (0-1)
- **Character Count**: Length of retrieved chunk
- **Content Preview**: First few words of the chunk

---

## 💬 Interactive Chat Experience

### **Starting a Session**
```
🧠 Loading or building vector store...
✅ Using cached FAISS index at vector_cache/abc12345
💬 Enter your questions (type 'exit' to quit)

>>>
```

### **Sample Conversation**
```
>>> What is the main topic of this document?
🧠 Answer: This document focuses on implementing retrieval-augmented 
generation for document querying, covering both technical architecture 
and practical applications.
⏱️  Took: 2.34 seconds

>>> Can you explain the methodology mentioned?
🧠 Answer: The methodology involves three key phases: document 
preprocessing using PDF extraction, vector embedding creation with 
FAISS indexing, and query-response generation using transformer models.
⏱️  Took: 1.87 seconds

>>> exit
🛑 Ollama model stopped: phi3:mini
👋 Exiting interactive session.
```

### **Session Features**
- **Real-time Responses**: Get answers as you type
- **Context Awareness**: Maintains conversation context
- **Performance Metrics**: Shows response time for each query
- **Graceful Exit**: Automatic cleanup when closing

---

## 🎛️ Advanced Configuration

### **Performance Tuning**
```bash
# High accuracy (more chunks)
python CLI_app.py --pdfs document.pdf --top_k 10

# Fast responses (fewer chunks)  
python CLI_app.py --pdfs document.pdf --top_k 1

# GPU optimization for large models
python CLI_app.py --pdfs document.pdf --use_gpu --model_id microsoft/DialoGPT-large
```

### **Memory Management**
```bash
# Monitor GPU memory
nvidia-smi

# Clear GPU cache (run in Python)
python -c "import torch; torch.cuda.empty_cache()"

# Check system memory
free -h  # Linux
wmic OS get TotalVisibleMemorySize /value  # Windows
```

---

## 🛠️ Model Backend Comparison

| Feature | Hugging Face (`hf`) | Ollama (`ollama`) |
|---------|-------------------|------------------|
| **GPU Support** | ✅ CUDA acceleration | ⚠️ Depends on setup |
| **Model Variety** | 🌟 Huge selection | 🔸 Curated models |
| **Setup Complexity** | 🟡 Moderate | 🟢 Simple |
| **Memory Usage** | 🔴 Higher | 🟢 Optimized |
| **Offline Usage** | ✅ Full offline | ✅ Full offline |
| **Custom Models** | ✅ Any HF model | 🔸 Ollama format only |

### **Hugging Face Benefits**
- Access to thousands of pre-trained models
- Direct GPU acceleration support
- Fine-tuned models for specific tasks
- Latest research models

### **Ollama Benefits**  
- Optimized inference engine
- Lower memory footprint
- Easy model management
- Built-in quantization

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### **NLTK Data Missing**
```
Error: Resource punkt_tab not found
```
**Solution:**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

#### **GPU Out of Memory**
```
Error: CUDA out of memory
```
**Solutions:**
```bash
# Use CPU instead
python CLI_app.py --pdfs document.pdf  # Remove --use_gpu

# Use smaller model
python CLI_app.py --pdfs document.pdf --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Reduce batch size (modify code)
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### **Ollama Model Not Found**
```
Error: model 'phi3:mini' not found
```
**Solutions:**
```bash
# Install the model
ollama pull phi3:mini

# List available models  
ollama list

# Check Ollama is running
ollama serve
```

#### **PDF Loading Errors**
```
Error: Unable to load PDF
```
**Solutions:**
```bash
# Check file path
ls -la sample_pdfs/document.pdf

# Verify PDF is not corrupted
file document.pdf

# Try different PDF
python CLI_app.py --pdfs sample_pdfs/sample1.pdf
```

### **Performance Issues**

#### **Slow Processing**
```bash
# Enable GPU (if available)
python CLI_app.py --pdfs document.pdf --use_gpu

# Reduce top_k for faster responses
python CLI_app.py --pdfs document.pdf --top_k 1

# Use cached results (second run is faster)
```

#### **High Memory Usage**
```bash
# Monitor memory usage
htop  # Linux
taskmgr  # Windows

# Use Ollama for lower memory
python CLI_app.py --pdfs document.pdf --model_backend ollama --model_id phi3:mini
```

---

## 📊 Performance Benchmarks

### **Typical Response Times** (TinyLlama on RTX 3080)
| Operation | Time | Description |
|-----------|------|-------------|
| First PDF load | 15-30s | PDF processing + embedding |
| Cached load | 2-5s | Using existing embeddings |
| Simple query | 1-3s | Basic factual questions |
| Complex query | 3-8s | Multi-step reasoning |
| GPU inference | 1-2s | Model response generation |
| CPU inference | 5-15s | Model response generation |

### **Memory Requirements**
| Model | GPU VRAM | System RAM | Notes |
|-------|----------|------------|-------|
| TinyLlama | 2-4 GB | 4-8 GB | Recommended minimum |
| DialoGPT-medium | 4-6 GB | 6-10 GB | Better quality |
| Llama-2-7b | 8-16 GB | 16+ GB | High-end setup |
| Ollama phi3:mini | 1-2 GB | 4-6 GB | Most efficient |

---

## 🔧 Customization & Extensions

### **Environment Variables**
```bash
export TRANSFORMERS_CACHE=/path/to/cache  # Model cache location
export CUDA_VISIBLE_DEVICES=0            # GPU selection
export TOKENIZERS_PARALLELISM=false      # Disable warnings
```

### **Script Modifications**
```python
# Increase context length (modify CLI_app.py)
def get_llm_with_context(model_id, max_length=2048):
    # Custom model loading with extended context
    pass

# Custom embedding model (modify global_objects.py)
def get_custom_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')
```

---

## 🎯 Best Practices

### **📁 File Organization**
```
project/
├── CLI_app.py
├── sample_pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── research/
│       └── paper.pdf
├── vector_cache/          # Auto-generated
└── uploaded_pdfs/         # For API usage
```

### **🚀 Performance Optimization**
1. **Use GPU** when available (`--use_gpu`)
2. **Cache embeddings** by reusing same PDFs
3. **Adjust top_k** based on document complexity
4. **Choose appropriate models** for your hardware
5. **Monitor resources** to avoid memory issues

### **💡 Usage Tips**
- Start with **TinyLlama + GPU** for best balance
- Use **debug mode** to understand retrieval quality
- **Preprocess large PDFs** during off-hours
- **Group related documents** for better context
- **Clean up caches** periodically to save disk space

---

## 🔄 Integration with API Server

### **Shared Components**
Both CLI and API server share:
- PDF loading and processing pipeline
- Vector store creation and caching
- Model loading and inference
- RAG chain construction

### **Switching Between Modes**
```bash
# CLI mode (interactive)
python CLI_app.py --pdfs document.pdf --model_backend hf

# API mode (server)
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Both use same cache folder
ls vector_cache/
```

### **Cache Compatibility**
Vector caches created by CLI can be used by the API server and vice versa, enabling seamless switching between interfaces.

---

## 📚 Example Use Cases

### **📖 Research Papers**
```bash
python CLI_app.py \
  --pdfs research/paper1.pdf research/paper2.pdf \
  --model_backend hf \
  --model_id microsoft/DialoGPT-medium \
  --use_gpu \
  --top_k 5
```

### **📄 Legal Documents**
```bash
python CLI_app.py \
  --pdfs contracts/agreement.pdf \
  --model_backend ollama \
  --model_id phi3:mini \
  --top_k 8 \
  --show_chunks
```

### **📊 Technical Manuals**
```bash
python CLI_app.py \
  --pdfs manuals/user_guide.pdf manuals/api_docs.pdf \
  --model_backend hf \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --use_gpu \
  --top_k 3
```

---

## 🎉 Getting Started Checklist

- [ ] **Activate virtual environment**
- [ ] **Install dependencies** (`pip install -r requirements.txt`)
- [ ] **Download NLTK data** (punkt_tab, punkt, stopwords)
- [ ] **Prepare PDF files** in accessible directory
- [ ] **Choose model backend** (hf vs ollama)
- [ ] **Test basic command** with sample PDF
- [ ] **Enable GPU** if available
- [ ] **Try debug mode** to understand retrieval
- [ ] **Experiment with parameters** (top_k, model_id)

---

✅ **You're now ready to master the CLI-based RAG chatbot!** 

Start with the basic TinyLlama example and gradually explore advanced features like multi-PDF processing, different models, and performance optimization.