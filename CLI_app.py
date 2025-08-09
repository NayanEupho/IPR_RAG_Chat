# venv\Scripts\activate to activate pre-existing env
# where python to check if the env got activated
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu

# CLI_app.py
import os
import argparse
import time
import warnings
import logging
import transformers

from langchain_community.vectorstores import FAISS

from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import create_vector_store, get_retriever
from rag_chatbot.global_objects import get_embedder, get_llm
from rag_chatbot.RAG_chain import build_rag_chain, answer_query


def get_cache_folder(pdfs: list[str], base_folder: str = "vector_cache") -> str:
    """Generates a unique cache folder path based on PDF contents."""
    import hashlib
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])


def interactive_loop(chain, show_chunks: bool = False):
    """Interactive CLI chat loop using shared answer_query logic."""
    print("\n💬 Enter your questions (type 'exit' to quit)\n")
    while True:
        query = input(">>> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting interactive session.")
            break

        start_time = time.time()
        result = answer_query(chain, query, show_chunks=show_chunks)
        print(f"\n🧠 Answer: {result['answer']}")
        print(f"⏱️  Took: {round(time.time() - start_time, 2)} seconds\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI RAG chatbot.")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Path(s) to one or more PDF files.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top relevant chunks to retrieve.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--show_chunks", action="store_true", help="Show metadata for retrieved chunks.")
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
    llm = get_llm(use_gpu=args.use_gpu)

    chain = build_rag_chain(llm, retriever)
    interactive_loop(chain, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
