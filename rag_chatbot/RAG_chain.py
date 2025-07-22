from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.llms.base import BaseLLM
from typing import List, Dict


def get_prompt() -> PromptTemplate:
    """
    Creates a custom prompt template for grounded answers.
    """
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


def build_rag_chain(
    llm: BaseLLM,
    retriever: VectorStoreRetriever
) -> RetrievalQA:
    """
    Builds the RetrievalQA chain using the provided LLM and retriever.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt()}
    )


def answer_query(chain: RetrievalQA, query: str) -> Dict:
    """
    Runs a query through the RAG chain and returns the result with sources.
    """
    result = chain.invoke(query)
    return {
        "answer": result["result"],
        "sources": extract_source_metadata(result.get("source_documents", []))
    }


def extract_source_metadata(docs: List[Document]) -> List[Dict]:
    """
    Extracts metadata from source documents (chunks), including type info if available.
    """
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append({
            "text": doc.page_content.strip(),
            "source": meta.get("source", "N/A"),
            "page": meta.get("page", "N/A"),
            "chunk": meta.get("chunk", "N/A"),
            "type": meta.get("chunk_type", "text")  # "table" if marked during preprocessing
        })
    return sources
