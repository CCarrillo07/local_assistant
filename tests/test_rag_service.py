import unittest
from unittest.mock import Mock, patch

from rag.service import RAGService
from rag.models import Chunk, Document, SearchResult
from rag.vector_store_base import VectorStore

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

    @patch("rag.service.build_rag_prompt")
    def test_build_prompt_reranks_retrieval_candidates(
        self,
        mock_build_rag_prompt
    ):
        """Rerank a wider candidate set before building the prompt."""

        question = "What is ORION-27?"

        first_candidate = SearchResult(
            chunk=Chunk(
                text="Unrelated Aurora information.",
                source="aurora.txt",
                page=None,
                chunk_index=0
            ),
            score=0.80
        )

        relevant_candidate = SearchResult(
            chunk=Chunk(
                text=(
                    "ORION-27 is the internal codename "
                    "for the example RAG knowledge base."
                ),
                source="rag_test_notes.txt",
                page=None,
                chunk_index=1
            ),
            score=0.70
        )

        candidates = [
            first_candidate,
            relevant_candidate
        ]
        reranked_results = [
            relevant_candidate
        ]

        vector_store = Mock(
            spec=VectorStore
        )
        vector_store.search.return_value = candidates

        reranker = Mock()
        reranker.rerank.return_value = reranked_results
        mock_build_rag_prompt.return_value = "RAG prompt"

        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder(),
            vector_store=vector_store,
            reranker=reranker,
            candidate_k=5,
            top_k=1,
            min_score=0.42
        )
        service.initialized = True

        prompt = service.build_prompt(
            question
        )

        self.assertEqual(
            prompt,
            "RAG prompt"
        )

        vector_store.search.assert_called_once_with(
            query=question,
            top_k=5,
            min_score=0.42
        )

        reranker.rerank.assert_called_once_with(
            question=question,
            results=candidates,
            top_k=1
        )

        mock_build_rag_prompt.assert_called_once_with(
            question=question,
            results=reranked_results
        )
    
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
        
    @patch("rag.service.load_documents")
    def test_initialize_reuses_restored_index(
        self,
        mock_load_documents
    ):
        vector_store = Mock(
            spec=VectorStore
        )
        
        vector_store.restore.return_value = True
        vector_store.size = 9
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder(),
            vector_store=vector_store
        )
        
        with patch.object(
            service,
            "_calculate_index_fingerprint",
            return_value="test-fingerprint"
        ): 
            service.initialize()
            
        self.assertTrue(
            service.initialized
        )
        
        vector_store.restore.assert_called_once_with(
            "test-fingerprint"
        )
        
        vector_store.rebuild.assert_not_called()
        mock_load_documents.assert_not_called()
            
if __name__ == "__main__":
    unittest.main()
