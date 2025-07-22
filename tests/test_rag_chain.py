import unittest
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from rag_chatbot.vector_store import get_embedding_model, create_vector_store, get_retriever
from rag_chatbot.LLM_interface import load_llm
from rag_chatbot.RAG_chain import build_rag_chain, answer_query
from tests.test_utils import get_selected_pdfs

class TestRAGChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pdfs = get_selected_pdfs()

    def test_full_rag_pipeline(self):
        pages = load_multiple_pdfs(self.selected_pdfs)
        chunks = split_text_into_chunks(pages)
        embedder = get_embedding_model()
        store = create_vector_store(chunks, embedder)
        retriever = get_retriever(store)
        llm = load_llm()
        chain = build_rag_chain(llm, retriever)
        result = answer_query(chain, "What is this document about?")
        self.assertIn("answer", result)

if __name__ == "__main__":
    unittest.main()
