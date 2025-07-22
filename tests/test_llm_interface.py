import unittest
from rag_chatbot.LLM_interface import load_llm
from tests.test_utils import get_selected_pdfs

class TestLLMInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected_pdfs = get_selected_pdfs()

    def test_llm_loads(self):
        llm = load_llm()
        self.assertIsNotNone(llm)
        self.assertTrue(callable(llm))

if __name__ == "__main__":
    unittest.main()
