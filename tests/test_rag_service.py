import unittest
from unittest.mock import patch

from rag.service import RAGService
from rag.models import Chunk, Document

class FakeEmbedder:
    
    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        
        return [
            [1.0, 0.0]
            for _ in texts
        ]
        
    def embed_text(
        self,
        text: str
    ) -> list[float]:
        
        if text == "Unrelated question":
                return [0.0, 1.0]
        
        return [1.0, 0.0]
    
class RAGServiceTest(unittest.TestCase):
    
    def test_requires_initialization_before_building_prompt(self):
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder()
        )
        
        with self.assertRaisesRegex(
            RuntimeError,
            "must be initialized"
        ):
            service.build_prompt(
                "What is ORION-27?"
            )
            
    @patch("rag.service.chunk_documents")
    @patch("rag.service.load_documents")
    def test_initialize_loads_and_indexes_chunks(
        self,
        mock_load_documents,
        mock_chunk_documents
    ):
        
        documents = [
            Document(
                text="Test document",
                source="test.txt"
            )
        ]
        chunks = [
            Chunk(
                text="Test document",
                source="test.txt",
                page=None,
                chunk_index=0
            )
        ]
        
        mock_load_documents.return_value = documents
        mock_chunk_documents.return_value = chunks
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder(),
            chunk_size=100,
            overlap=20
        )
        
        service.initialize()
        
        self.assertTrue(
            service.initialized
        )
        
        self.assertEqual(
            service.vector_store.chunks,
            chunks
        )
        
        self.assertEqual(
            len(service.vector_store.vectors),
            1
        )
        
        mock_load_documents.assert_called_once_with(
            "documents"
        )
        
        mock_chunk_documents.assert_called_once_with(
            documents=documents,
            chunk_size=100,
            overlap=20
        )
        
    def test_build_prompt_includes_retrieved_context(self):
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder(),
            top_k=1,
            min_score=0.50
        )
        
        service.vector_store.add_chunks(
            [
                Chunk(
                    text=(
                        "ORION-27 is the internal codename "
                        "for the example RAG knowledge base."
                    ),
                    source="rag_test_notes.txt",
                    page=None,
                    chunk_index=0
                )
            ]
        )
        
        service.initialized = True
        
        prompt = service.build_prompt(
            "What is ORION-27?"
        )
        
        self.assertIsNotNone(
            prompt
        )
        
        self.assertIn(
            "What is ORION-27?",
            prompt
        )
        
        self.assertIn(
            "ORION-27 is the internal codename",
            prompt
        )
        
        self.assertIn(
            "rag_test_notes.txt, chunk 0",
            prompt
        )
        
    def test_build_prompt_returns_none_for_irrelevant_question(self):
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder(),
            top_k=1,
            min_score=0.50
        )
        
        service.vector_store.add_chunks(
            [
                Chunk(
                    text=(
                        "ORION-27 is the internal codename "
                        "for the example RAG knowledge base."
                    ),
                    source="rag_test_notes.txt",
                    page=None,
                    chunk_index=0
                )
            ]
        )
        
        service.initialized = True

        prompt = service.build_prompt(
            "Unrelated question"
        )
        
        self.assertIsNone(
            prompt
        )
            
if __name__ == "__main__":
    unittest.main()