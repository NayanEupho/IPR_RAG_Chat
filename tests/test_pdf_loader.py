import unittest
from rag_chatbot.pdf_loader import load_multiple_pdfs
from tests.test_utils import get_selected_pdfs

class TestPDFLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pdfs = get_selected_pdfs()

    def test_load_selected_pdfs(self):
        pages = load_multiple_pdfs(self.selected_pdfs)
        self.assertGreater(len(pages), 0)
        self.assertIsInstance(pages[0], tuple)
        self.assertEqual(len(pages[0]), 3)

if __name__ == "__main__":
    unittest.main()
