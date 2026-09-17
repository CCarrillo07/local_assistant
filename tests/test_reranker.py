import unittest

from rag.flashrank_reranker import FlashRankReranker
from rag.models import Chunk, SearchResult


class FakeRanker:
    """Return deterministic scores without loading a real model."""

    def __init__(self):
        self.request = None

    def rerank(self, request):
        self.request = request

        return [
            {
                "id": 1,
                "text": request.passages[1]["text"],
                "score": 0.90
            },
            {
                "id": 0,
                "text": request.passages[0]["text"],
                "score": 0.20
            }
        ]


class FlashRankRerankerTest(unittest.TestCase):

    def test_reranks_filters_and_limits_results(self):
        fake_ranker = FakeRanker()

        reranker = FlashRankReranker(
            ranker=fake_ranker,
            min_score=0.50
        )

        irrelevant_result = SearchResult(
            chunk=Chunk(
                text="Unrelated information.",
                source="unrelated.txt",
                page=None,
                chunk_index=0
            ),
            score=0.80
        )

        relevant_result = SearchResult(
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

        results = reranker.rerank(
            question="What is ORION-27?",
            results=[
                irrelevant_result,
                relevant_result
            ],
            top_k=2
        )

        self.assertEqual(
            fake_ranker.request.query,
            "What is ORION-27?"
        )

        self.assertEqual(
            fake_ranker.request.passages,
            [
                {
                    "id": 0,
                    "text": "Unrelated information."
                },
                {
                    "id": 1,
                    "text": (
                        "ORION-27 is the internal codename "
                        "for the example RAG knowledge base."
                    )
                }
            ]
        )

        self.assertEqual(
            len(results),
            1
        )

        self.assertEqual(
            results[0].chunk,
            relevant_result.chunk
        )

        self.assertEqual(
            results[0].score,
            0.90
        )


if __name__ == "__main__":
    unittest.main()
