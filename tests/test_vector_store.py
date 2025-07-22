import unittest
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.vector_store import get_embedding_model, create_vector_store, get_retriever
from langchain.schema import Document
from tests.test_utils import get_selected_pdfs

class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pdfs = get_selected_pdfs()

    def test_embedding_and_retrieval(self):
        pages = load_multiple_pdfs(self.selected_pdfs)

        docs = [
            Document(
                page_content=content,
                metadata={"source": filename, "page": page_num, "chunk": 0}
            )
            for content, filename, page_num in pages
        ]

        embedder = get_embedding_model()
        store = create_vector_store(docs, embedder)
        retriever = get_retriever(store)
        results = retriever.invoke("test")
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
