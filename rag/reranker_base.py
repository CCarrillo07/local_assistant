from abc import ABC, abstractmethod

from rag.models import SearchResult

class Reranker(ABC):
    """Common contract for result-reranking implementations."""

    @abstractmethod
    def rerank(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        """Reorder and filter retrieved search results."""
        raise NotImplementedError