import unittest
from rag_chatbot.pdf_loader import load_multiple_pdfs
from rag_chatbot.text_splitter import split_text_into_chunks
from tests.test_utils import get_selected_pdfs

class TestTextSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pdfs = get_selected_pdfs()

    def test_chunking(self):
        pages = load_multiple_pdfs(self.selected_pdfs)
        chunks = split_text_into_chunks(pages)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(hasattr(chunks[0], 'page_content'))
        self.assertIn('source', chunks[0].metadata)

if __name__ == "__main__":
    unittest.main()
