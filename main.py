# main.py
import argparse
from rich import print
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import get_embedding_model, create_vector_store, get_retriever
from rag_chatbot.LLM_interface import load_llm
from rag_chatbot.RAG_chain import build_rag_chain, answer_query

console = Console()

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot - Ask questions about your PDFs")
    parser.add_argument('--pdfs', nargs='+', required=True, help="List of PDF file paths")
    parser.add_argument('--query', required=True, help="The question to ask based on the PDFs")
    args = parser.parse_args()

    pdf_paths = [Path(p) for p in args.pdfs]

    # ✅ Step 1: Load and clean PDF text
    console.print("\n[bold yellow]🔍 Loading PDFs...[/bold yellow]")
    pages = load_multiple_pdfs([str(p) for p in pdf_paths])
    console.print(f"Loaded {len(pages)} pages.\n")

    # ✅ Step 2: Split into chunks
    console.print("[bold yellow]🧩 Splitting into chunks...[/bold yellow]")
    chunks = split_text_into_chunks(pages)
    console.print(f"Generated {len(chunks)} chunks.\n")

    # ✅ Step 3: Embed and build vector store
    console.print("[bold yellow]🔗 Creating vector store...[/bold yellow]")
    embedding_model = get_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store)

    # ✅ Step 4: Load LLM
    console.print("[bold yellow]🤖 Loading language model...[/bold yellow]")
    llm = load_llm()

    # ✅ Step 5: Build RAG chain and answer
    console.print("[bold yellow]💬 Answering your question...[/bold yellow]")
    chain = build_rag_chain(llm, retriever)
    result = answer_query(chain, args.query)

    # ✅ Step 6: Print results
    console.print(Panel(result["answer"], title="📘 [bold green]Answer", style="green"))

    if result["sources"]:
        console.print("\n[bold cyan]📚 Source Chunks Used:[/bold cyan]")
        for i, src in enumerate(result["sources"], start=1):
            console.print(
                Panel(
                    f"[bold]File:[/bold] {src['source']} | [bold]Page:[/bold] {src['page']} | [bold]Chunk:[/bold] {src['chunk']}\n\n{src['text']}",
                    title=f"🔹 Chunk {i}",
                    style="blue"
                )
            )


if __name__ == "__main__":
    main()
