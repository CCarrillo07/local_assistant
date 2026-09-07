import numpy as np

from logger import get_logger
from rag.embedder import OllamaEmbedder
from rag.models import Chunk, SearchResult

logger = get_logger(__name__)

class InMemoryVectorStore:

    def __init__(
        self,
        embedder: OllamaEmbedder
    ):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[np.ndarray] = []

    def add_chunks(
        self,
        chunks: list[Chunk]
    ) -> None:

        if not chunks: 
            logger.info("No chunks were provided")
            return
        
        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_texts(texts)

        self.chunks.extend(chunks)

        self.vectors.extend(
            np.array(
                embedding,
                dtype=np.float32
            )
            for embedding in embeddings
        )

        logger.info(
            "Added %s chunks to the vector store",
            len(chunks)
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None
    ) -> list[SearchResult]:
        
        if not self.chunks:
            logger.info("Vector store is empty")
            return[]

        query_vector = np.array(
            self.embedder.embed_text(query),
            dtype=np.float32
        )

        results = []

        for chunk, vector in zip(
            self.chunks,
            self.vectors
        ):
            
            score = self._cosine_similarity(
                query_vector,
                vector
            )

            result = SearchResult(
                chunk=chunk,
                score=score
            )

            results.append(result)

        results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        if min_score is not None:
            results = [
                result for result in results if result.score >= min_score
            ]

        return results[:top_k]

    @staticmethod
    def _cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray
    ) -> float:

        denominator = (
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        )

        if denominator == 0:
            return 0.0

        similarity = np.dot(
            vector_a,
            vector_b
        ) / denominator

        return float(similarity)

