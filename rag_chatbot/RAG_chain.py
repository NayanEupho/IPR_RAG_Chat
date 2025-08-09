from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.llms.base import BaseLLM
from typing import List, Dict


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
    """Remove prompt/context text from the model's answer."""
    cleaned = raw.replace(prompt, "").strip()
    for tag in ["Answer:", "Context:", prompt]:
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1].strip()
    return cleaned


def build_rag_chain(
    llm: BaseLLM,
    retriever: VectorStoreRetriever
) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt()}
    )


def answer_query(chain: RetrievalQA, query: str, show_chunks: bool = False) -> Dict:
    result = chain.invoke(query)

    if show_chunks:
        print_chunk_summary(result.get("source_documents", []))

    # Get the prompt string we used
    context = "\n\n".join(doc.page_content for doc in result.get("source_documents", []))
    prompt = get_prompt().format(context=context, question=query)

    # Clean the LLM output
    cleaned_answer = clean_response(result["result"], prompt)

    return {
        "answer": cleaned_answer,
        "sources": extract_source_metadata(result.get("source_documents", []))
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
            "type": meta.get("chunk_type", "text")
        })
    return sources


def print_chunk_summary(docs: List[Document]):
    print(f"🔍 Top {len(docs)} chunks retrieved:")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        print(
            f"  {i}. Page {meta.get('page', 'N/A')}, "
            f"Type: {meta.get('chunk_type', 'text')}, "
            f"Source: {meta.get('source', 'N/A')}"
        )
