# rag_chatbot/RAG_chain.py
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.prompts import PromptTemplate

def get_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the provided document excerpts to answer the user's question accurately.
Ensure your response is concise, factual, and grounded in the context. Do not make up information.

Context:
{context}

Question:
{question}

Answer:"""
    )


def clean_response(raw: str, prompt: str) -> str:
    cleaned = raw.replace(prompt, "").strip()
    for tag in ["Answer:", "Context:", prompt]:
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1].strip()
    return cleaned


def build_rag_chain(llm, retriever):
    """
    Return a simple chain object (a tuple) containing llm and retriever.
    The 'llm' must support .invoke(prompt) -> str.
    The 'retriever' must support .invoke(query) -> List[Document].
    """
    return {"llm": llm, "retriever": retriever}


def answer_query(chain, query: str, show_chunks: bool = False) -> Dict:
    """
    Retrieves top chunks using the retriever, builds a prompt, invokes the LLM,
    cleans the response, and returns answer + sources metadata.
    """
    llm = chain["llm"]
    retriever = chain["retriever"]

    # Retrieve docs
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        relevant_docs = retriever.invoke(query)

    if show_chunks:
        _print_chunk_summary(relevant_docs)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt_template = get_prompt()
    prompt = prompt_template.format(context=context, question=query)

    # call LLM
    response = llm.invoke(prompt)
    # ensure response is a string
    if isinstance(response, list) and len(response) > 0:
        # try to extract
        candidate = response[0]
        response = candidate.get("generated_text") or candidate.get("text") or str(candidate)
    elif isinstance(response, dict):
        response = response.get("text") or response.get("generated_text") or str(response)

    cleaned = clean_response(str(response), prompt)
    return {
        "answer": cleaned,
        "sources": extract_source_metadata(relevant_docs)
    }


def extract_source_metadata(docs: List[Document]) -> List[Dict]:
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append({
            "text": doc.page_content.strip(),
            "source": meta.get("source", "N/A"),
            "page": meta.get("page", "N/A"),
            "chunk": meta.get("chunk", "N/A"),
            "type": meta.get("type", meta.get("chunk_type", "text"))
        })
    return sources


def _print_chunk_summary(docs: List[Document]):
    print(f"🔍 Top {len(docs)} chunks retrieved:")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        print(
            f"  {i}. Page {meta.get('page', 'N/A')}, "
            f"Type: {meta.get('type', meta.get('chunk_type', 'text'))}, "
            f"Source: {meta.get('source', 'N/A')}"
        )
