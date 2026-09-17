import unittest
from unittest.mock import patch

from config import RAGConfig
from rag.reranker_factory import create_reranker


class RerankerFactoryTest(unittest.TestCase):

    def test_returns_none_when_reranker_is_disabled(self):
        config = RAGConfig(
            reranker_enabled=False
        )

        reranker = create_reranker(
            config
        )

        self.assertIsNone(
            reranker
        )

    @patch(
        "rag.reranker_factory.FlashRankReranker"
    )
    def test_creates_configured_flashrank_reranker(
        self,
        mock_reranker_class
    ):
        config = RAGConfig(
            reranker_enabled=True,
            reranker_backend="flashrank",
            reranker_model=(
                "ms-marco-MiniLM-L-12-v2"
            ),
            reranker_cache_dir="storage/test-reranker",
            reranker_max_length=128,
            reranker_min_score=0.50,
            candidate_k=5
        )

        reranker = create_reranker(
            config
        )

        self.assertIs(
            reranker,
            mock_reranker_class.return_value
        )

        mock_reranker_class.assert_called_once_with(
            model_name=(
                "ms-marco-MiniLM-L-12-v2"
            ),
            cache_dir="storage/test-reranker",
            max_length=128,
            min_score=0.50
        )


if __name__ == "__main__":
    unittest.main()
