# venv\Scripts\activate to activate pre-existing env
# where python to check if the env got activated
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu --show_chunks
# python CLI_app.py --pdfs sample_pdfs/sample1.pdf --use_gpu

import os
import argparse
import time
import warnings
import logging
import transformers

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import get_embedding_model, create_vector_store, get_retriever
from rag_chatbot.LLM_interface import load_llm


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


def clean_response(raw: str, prompt: str) -> str:
    """Cleans the raw LLM output by removing the prompt and repeated context."""
    cleaned = raw.replace(prompt, "").strip()
    for tag in ["Answer:", "Context:", prompt]:
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1].strip()
    return cleaned


def print_chunk_summary(docs: list[Document]):
    print(f"🔍 Top {len(docs)} chunks retrieved:")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(
            f"  {i}. Page {meta.get('page')}, "
            f"Type: {meta.get('type', 'text')}, "
            f"Section: {meta.get('section', 'N/A')}, "
            f"Source: {meta.get('source')}"
        )


def interactive_loop(llm, retriever, show_chunks: bool = False):
    print("\n💬 Enter your questions (type 'exit' to quit)\n")
    while True:
        query = input(">>> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting interactive session.")
            break

        start_time = time.time()
        relevant_docs = retriever.invoke(query)

        if show_chunks:
            print_chunk_summary(relevant_docs)

        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        prompt = (
            f"Answer the question based on the context below:\n\n"
            f"{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        response = llm.invoke(prompt)

        if isinstance(response, list):
            response = response[0].get("generated_text", "")
        elif isinstance(response, dict) and "text" in response:
            response = response["text"]

        cleaned = clean_response(response, prompt)
        print(f"\n🧠 Answer: {cleaned}")
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
    embedder = get_embedding_model()
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

    print("🚀 Loading LLM...")
    llm = load_llm(use_gpu=args.use_gpu)

    interactive_loop(llm, retriever, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
