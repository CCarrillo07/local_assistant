from abc import ABC, abstractmethod

from rag.models import Chunk, SearchResult

class VectorStore(ABC):
    """Common contract for all vector_store implementations ."""
    
    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk]
    ) -> None:
        """Add document chunks and their embeddings."""
        raise NotImplementedError
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None
    ) -> list[SearchResult]:
        """Return the document chunks most relevant to a query."""
        raise NotImplementedError