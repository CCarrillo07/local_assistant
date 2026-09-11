import json
from dataclasses import asdict
from pathlib import Path

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
        
    def save(
        self,
        index_path: str | Path
    ) -> None:
        """Save chunks and vectors to a compressed local index."""
        
        if len(self.chunks) != len(self.vectors):
            raise RuntimeError(
                "The number of chunks and vectors must match"
            )
            
        path = Path(index_path)
        
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        
        chunk_data = [
            asdict(chunk)
            for chunk in self.chunks
        ]
        
        if self.vectors:
            vectors = np.stack(
                self.vectors
            )
        else:
            vectors = np.empty(
                (0,0),
                dtype=np.float32
            )
            
        with path.open("wb") as index_file:
            np.savez_compressed(
                index_file,
                chunks_json=json.dumps(chunk_data),
                vectors=vectors
            )
            
        logger.info(
            "Saved %s chunks to index: %s",
            len(self.chunks),
            path
        )
        
    def load(
        self,
        index_path: str | Path
    ) -> None:
        """Load chunks and vectors from a compressed local index."""
        
        path = Path(index_path)
        
        with np.load(
            path,
            allow_pickle=False
        ) as index:
            chunk_data = json.loads(
                str(index["chunks_json"].item())
            )
            
            vectors = np.array(
                index["vectors"],
                dtype=np.float32
            )
            
        if len(chunk_data) != len(vectors):
            raise ValueError(
                "The stored chunks and vectors do not match"
            )
            
        self.chunks = [
            Chunk(**data)
            for data in chunk_data
        ]
        
        self.vectors = [
            vector.copy()
            for vector in vectors
        ]
        
        logger.info(
            "Loaded %s chunks from index: %s",
            len(self.chunks),
            path    
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

