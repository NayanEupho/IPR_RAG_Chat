import argparse
import json
import os
import hashlib
import time
import warnings
import logging
import transformers

from langchain_community.vectorstores import FAISS
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import get_embedding_model, create_vector_store, get_retriever
from rag_chatbot.LLM_interface import load_llm


def collect_sources(query: str, docs) -> dict:
    return {
        "query": query,
        "used_documents": [
            {
                "index": i + 1,
                "source": doc.metadata.get("source", "N/A"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_type": doc.metadata.get("type", "text"),
                "section": doc.metadata.get("section", "Unknown"),
                "chunk_text": doc.page_content
            }
            for i, doc in enumerate(docs)
        ]
    }


def get_cache_folder(pdfs: list[str], base_folder: str = "vector_cache") -> str:
    os.makedirs(base_folder, exist_ok=True)
    combined_hash = hashlib.md5()
    for path in sorted(pdfs):
        combined_hash.update(path.encode())
        if os.path.exists(path):
            with open(path, "rb") as f:
                combined_hash.update(f.read())
    return os.path.join(base_folder, combined_hash.hexdigest()[:8])


def clean_response(raw: str, prompt: str) -> str:
    """Cleans the LLM response to strip out repeated prompt or extra Answer: sections."""
    cleaned = raw.replace(prompt, "").strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1].strip()
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Run a RAG-based PDF QA query from the command line.")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Path(s) to one or more PDF files.")
    parser.add_argument("--query", required=True, help="User query/question to ask the chatbot.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top relevant chunks to retrieve (default: 3).")
    parser.add_argument("--save_source", help="Path to save source metadata (traceability) as JSON.")
    parser.add_argument("--use_gpu", action="store_true", help="Force use of GPU if available (default: auto-detect).")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild of vector index even if cache exists.")
    args = parser.parse_args()

    # 🧹 Quiet logs
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()

    start_time = time.time()

    print("🧠 Loading or building vector store...")
    embedder = get_embedding_model()
    cache_dir = get_cache_folder(args.pdfs)
    index_file = os.path.join(cache_dir, "index.faiss")

    if os.path.exists(index_file) and not args.force_rebuild:
        print(f"✅ Found cached vector index: {cache_dir}")
        vectorstore = FAISS.load_local(
            cache_dir,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
    else:
        print("📥 Loading PDFs...")
        pages = load_multiple_pdfs(args.pdfs)

        print("🔪 Splitting text into chunks...")
        chunks = split_text_into_chunks(pages)

        print("⚙️  Creating new index...")
        vectorstore = create_vector_store(chunks, embedder)
        vectorstore.save_local(cache_dir)
        print(f"💾 Vector index saved to: {cache_dir}")

    retriever = get_retriever(vectorstore, top_k=args.top_k)

    print("🔍 Retrieving relevant chunks...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relevant_docs = retriever.invoke(args.query)

    print(f"🔍 Top {args.top_k} chunks retrieved ({len(relevant_docs)} total):")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. Page {doc.metadata.get('page')} | Type: {doc.metadata.get('type', 'text')} | Source: {doc.metadata.get('source')}")

    if args.save_source:
        trace_data = collect_sources(args.query, relevant_docs)
        try:
            with open(args.save_source, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Source info saved to: {args.save_source}")
        except Exception as e:
            print(f"❌ Failed to save source file: {e}")

    print("\n🤖 Generating answer...")
    try:
        llm = load_llm(use_gpu=args.use_gpu)
    except RuntimeError as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        f"Answer the question based on the context below:\n\n"
        f"{context}\n\n"
        f"Question: {args.query}\nAnswer:"
    )

    response = llm.invoke(prompt)

    if isinstance(response, list):
        response = response[0].get("generated_text", "")
    elif isinstance(response, dict) and "text" in response:
        response = response["text"]

    cleaned = clean_response(response, prompt)

    print(f"\nQuestion: {args.query}")
    print(f"Answer: {cleaned}")
    print(f"\n⏱️  Total runtime: {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    main()
