from typing import Any

from flashrank import Ranker, RerankRequest

from rag.models import SearchResult
from rag.reranker_base import Reranker

class FlashRankReranker(Reranker):
    """Rerank retrieved chunks using a local cross-encoder model."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: str = "storage/flashrank",
        max_lenght: int = 128,
        min_score: float | None = None,
        ranker: Any | None = None
    ):
        self.min_score = min_score

        # Dependency injection lets unit tests provide a fake ranker
        # without downloading or loading a real model.
        self.ranker = (
            ranker
            if ranker is not None
            else Ranker(
                model_name=model_name,
                cache_dir=cache_dir,
                max_length=max_lenght
            )
        )

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        """Reorder candidates and remove insufficiently relevant results."""

        if not results:
            return []

        passages = [
            {
                "id": index,
                "text": result.chunk.text
            }
            for index, result in enumerate(results)
        ]

        request = RerankRequest(
            query=question,
            passages=passages
        )

        ranked_passages = self.ranker.rerank(
            request
        )

        reranked_results = []

        for passage in ranked_passages:
            score = float(
                passage["score"]
            )

            if (
                self.min_score is not None
                and score < self.min_score
            ):
                continue

            original_index = int(
                passage["id"]
            )

            original_result = results[
                original_index
            ]

            reranked_results.append(
                SearchResult(
                    chunk=original_index.chunk,
                    score=score
                )
            )

            if len(reranked_results) >= top_k:
                break

        return reranked_results