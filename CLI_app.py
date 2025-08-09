# CLI_app.py
# venv\Scripts\activate to activate pre-existing env
# where python to check if the env got activated
# Example commands:
# HuggingFace TinyLlama (GPU):
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks --model_backend hf --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Ollama phi3:mini:
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --show_chunks --model_backend ollama --model_id phi3:mini

import os
import argparse
import time
import warnings
import logging
import subprocess
import transformers

from langchain_community.vectorstores import FAISS
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import create_vector_store, get_retriever
from rag_chatbot.global_objects import get_embedder, get_llm, unload_llm
from rag_chatbot.RAG_chain import build_rag_chain, answer_query


def get_cache_folder(pdfs: list[str], base_folder: str = "vector_cache") -> str:
    import hashlib
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])


def interactive_loop(chain, backend: str, model_id: str, use_gpu: bool, show_chunks: bool = False):
    print("\n💬 Enter your questions (type 'exit' to quit)\n")
    while True:
        query = input(">>> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting interactive session.")

            # Stop Ollama model if in use
            if backend.lower() == "ollama":
                try:
                    subprocess.run(
                        ["ollama", "stop", model_id],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"🛑 Ollama model stopped: {model_id}")
                except Exception as e:
                    print(f"⚠️ Failed to stop Ollama model {model_id}: {e}")

            # Unload from LLM cache
            unload_llm(backend=backend, model_id=model_id, use_gpu=use_gpu)
            break

        start_time = time.time()
        result = answer_query(chain, query, show_chunks=show_chunks)
        print(f"\n🧠 Answer: {result['answer']}")
        print(f"⏱️  Took: {round(time.time() - start_time, 2)} seconds\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI RAG chatbot.")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Path(s) to one or more PDF files.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top relevant chunks to retrieve.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available (HF backend only).")
    parser.add_argument("--show_chunks", action="store_true", help="Show metadata for retrieved chunks.")
    parser.add_argument("--model_backend", choices=["hf", "ollama"], default="hf", help="LLM backend to use.")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model ID (HF model name or Ollama model name, e.g., 'phi3:mini').")
    args = parser.parse_args()

    # Silence warnings and logs
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()

    print("🧠 Loading or building vector store...")
    embedder = get_embedder()
    cache_dir = get_cache_folder(args.pdfs)
    index_file = os.path.join(cache_dir, "index.faiss")

    if os.path.exists(index_file):
        print(f"✅ Using cached FAISS index at {cache_dir}")
        vectorstore = FAISS.load_local(cache_dir, embeddings=embedder, allow_dangerous_deserialization=True)
    else:
        print("📥 Loading PDFs and preparing chunks...")
        pages = load_multiple_pdfs(args.pdfs)
        chunks = split_text_into_chunks(pages)
        print("⚙️  Creating new index...")
        vectorstore = create_vector_store(chunks, embedder)
        vectorstore.save_local(cache_dir)
        print(f"💾 Vector index saved to: {cache_dir}")

    retriever = get_retriever(vectorstore, top_k=args.top_k)

    # Load LLM
    llm = get_llm(
        backend=args.model_backend,
        model_id=args.model_id,
        use_gpu=args.use_gpu
    )

    chain = build_rag_chain(llm, retriever)
    interactive_loop(chain, backend=args.model_backend, model_id=args.model_id, use_gpu=args.use_gpu, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
