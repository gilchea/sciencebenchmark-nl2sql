import unittest
import os
import sys

# Thêm src vào path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever.semantic import SemanticRetriever

class TestSemanticRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a mock example pool and retriever."""
        cls.example_pool = [
            {"question": "show me all the organizations", "query": "SELECT name FROM organization", "sql": {}},
            {"question": "how many projects are there", "query": "SELECT count(*) FROM project", "sql": {}},
            {"question": "list projects with budget over 1 million", "query": "SELECT * FROM project WHERE budget > 1000000", "sql": {}},
            {"question": "what are the project titles", "query": "SELECT title FROM project", "sql": {}},
        ]

        # Dùng CPU để test cho nhanh và độc lập
        cls.retriever = SemanticRetriever(device='cpu')
        cls.retriever.build_index(cls.example_pool)

    def test_retriever_initialization(self):
        """Test if the retriever and index are created."""
        self.assertIsNotNone(self.retriever)
        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.index.ntotal, 4)

    def test_retrieve_k1(self):
        """Test retrieving 1 example."""
        query = "what is the total budget" # Gần với "budget over 1 million"
        results = self.retriever.retrieve(query, k=1)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIn('question', results[0])
        self.assertIn('query', results[0]) # Kiểm tra key mới
        self.assertEqual(results[0]['question'], "list projects with budget over 1 million")

    def test_retrieve_k2(self):
        """Test retrieving 2 examples."""
        query = "count all projects" # Gần với "how many projects" và "project titles"
        results = self.retriever.retrieve(query, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['question'], "how many projects are there")

print("File test/test_retriever.py đã được ghi.")
