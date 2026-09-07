import unittest

from rag.models import Chunk
from rag.vector_store import InMemoryVectorStore

class FakeEmbedder:

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        vectors = {
            "RAG document": [1.0, 0.0],
            "Cooking document": [0.0, 1.0]
        }

        return [
            vectors[text]
            for text in texts
        ]
    
    def embed_text(
        self,
        text: str
    ) -> list[float]:

        vectors = {
            "RAG question" : [1.0, 0.0],
            "Unrelated question": [1.0, 1.0]
        }

        return vectors[text]

class VectorStoreTest(unittest.TestCase):
    
    def setUp(self):

        self.vector_store = InMemoryVectorStore(
            embedder=FakeEmbedder()
        )

        self.chunks = [
            Chunk(
                text="RAG document",
                source="rag.txt",
                page=None,
                chunk_index=0
            ),
            Chunk(
                text="Cooking document",
                source="cooking.txt",
                page=None,
                chunk_index=0
            )
        ]

        self.vector_store.add_chunks(
            self.chunks
        )

    def test_returns_most_similar_chunk(self):

        results = self.vector_store.search(
            query="RAG question",
            top_k=1
        )

        self.assertEqual(
            len(results),
            1
        )

        self.assertEqual(
            results[0].chunk.source,
            "rag.txt"
        )

        self.assertAlmostEqual(
            results[0].score,
            1.0
        )

    def test_filters_results_below_threshold(self):

        results = self.vector_store.search(
            query="Unrelated question",
            top_k=2,
            min_score=8.0
        )

        self.assertEqual(
            results,
            []
        )

if __name__ == "__main__":
    unittest.main()