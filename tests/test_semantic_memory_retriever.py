import unittest

from memory.long_term_base import MemoryRecord
from memory.semantic_retriever import SemanticMemoryRetriever


class FakeEmbedder:

    def __init__(self):
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    def embed_text(
        self,
        text: str
    ) -> list[float]:
        self.query_calls.append(
            text
        )
        return [1.0, 0.0]

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        self.document_calls.append(
            texts
        )

        embeddings = {
            "My puppy is named Happy": [0.0, 1.0],
            "My favorite food is enchiladas": [0.9, 0.1],
            "I live in Mérida": [0.7, 0.7]
        }

        return [
            embeddings[text]
            for text in texts
        ]


class SemanticMemoryRetrieverTest(unittest.TestCase):

    def setUp(self):
        self.memories = [
            MemoryRecord(
                memory_id=1,
                content="My puppy is named Happy"
            ),
            MemoryRecord(
                memory_id=2,
                content="My favorite food is enchiladas"
            ),
            MemoryRecord(
                memory_id=3,
                content="I live in Mérida"
            )
        ]

    def test_returns_most_relevant_memories_in_score_order(self):
        """Rank memories semantically and respect the result limit."""

        embedder = FakeEmbedder()
        retriever = SemanticMemoryRetriever(
            embedder=embedder,
            min_score=0.5
        )

        results = retriever.retrieve(
            query="What food do I like?",
            memories=self.memories,
            limit=2
        )

        self.assertEqual(
            results,
            [
                self.memories[1],
                self.memories[2]
            ]
        )
        self.assertEqual(
            embedder.query_calls,
            ["What food do I like?"]
        )
        self.assertEqual(
            embedder.document_calls,
            [[
                memory.content
                for memory in self.memories
            ]]
        )

    def test_filters_memories_below_minimum_score(self):
        """Exclude weak semantic matches before building context."""

        retriever = SemanticMemoryRetriever(
            embedder=FakeEmbedder(),
            min_score=0.8
        )

        results = retriever.retrieve(
            query="What food do I like?",
            memories=self.memories,
            limit=3
        )

        self.assertEqual(
            results,
            [
                self.memories[1]
            ]
        )

    def test_empty_memory_list_skips_embedding(self):
        """Avoid an embedding request when no memories are stored."""

        embedder = FakeEmbedder()
        retriever = SemanticMemoryRetriever(
            embedder=embedder,
            min_score=0.5
        )

        results = retriever.retrieve(
            query="What food do I like?",
            memories=[],
            limit=2
        )

        self.assertEqual(
            results,
            []
        )
        self.assertEqual(
            embedder.query_calls,
            []
        )
        self.assertEqual(
            embedder.document_calls,
            []
        )


if __name__ == "__main__":
    unittest.main()
