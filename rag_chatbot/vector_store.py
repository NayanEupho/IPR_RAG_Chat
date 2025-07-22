import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings
) -> FAISS:
    # Count table vs text chunks
    table_chunks = sum(1 for doc in documents if doc.metadata.get("type") == "table")
    text_chunks = sum(1 for doc in documents if doc.metadata.get("type") == "text")
    unknown_chunks = len(documents) - (table_chunks + text_chunks)

    print(f"\n📊 Indexing {text_chunks} text chunks, {table_chunks} table chunks"
          + (f", {unknown_chunks} unknown type chunks" if unknown_chunks else "") + "...")

    return FAISS.from_documents(documents, embedding_model)


def save_local_vector_store(vectorstore: FAISS, save_path: str):
    """Saves the FAISS index to a local directory."""
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(folder_path=save_path)


def load_local_vector_store(load_path: str, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """Loads a FAISS index from a local directory."""
    return FAISS.load_local(folder_path=load_path, embeddings=embedding_model)


def get_retriever(
    vector_store: FAISS,
    top_k: int = 3
) -> VectorStoreRetriever:
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
