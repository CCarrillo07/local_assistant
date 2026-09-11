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

        # These lists are parallel: chunks[0] describes vectors[0], and so on.
        # Keeping them aligned is essential for returning the correct source
        # after a similarity search.
        self.chunks: list[Chunk] = []
        self.vectors: list[np.ndarray] = []

    def add_chunks(
        self,
        chunks: list[Chunk]
    ) -> None:

        if not chunks:
            logger.info("No chunks were provided")
            return

        # Embedding models receive plain text, not complete Chunk objects.
        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_texts(texts)

        # Store the metadata objects and their embeddings in the same order.
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
        index_path: str | Path,
        metadata: dict[str, object] | None = None
    ) -> None:
        """Save chunks and vectors to a compressed local index."""
        
        if len(self.chunks) != len(self.vectors):
            raise RuntimeError(
                "The number of chunks and vectors must match"
            )

        path = Path(index_path)

        # Create the storage directory automatically when it does not exist.
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Dataclass objects cannot be written directly as JSON. asdict()
        # converts each Chunk into a regular dictionary containing text,
        # source, page, and chunk_index.
        chunk_data = [
            asdict(chunk)
            for chunk in self.chunks
        ]

        # Convert the list of individual embedding arrays into one matrix.
        # Each row in this matrix corresponds to one item in chunk_data.
        if self.vectors:
            vectors = np.stack(
                self.vectors
            )
        else:
            vectors = np.empty(
                (0, 0),
                dtype=np.float32
            )

        # The NPZ file stores both parts of the index together:
        # - chunks_json: searchable-text metadata encoded as JSON
        # - vectors: the numeric embedding matrix used for similarity search
        with path.open("wb") as index_file:
            np.savez_compressed(
                index_file,
                chunks_json=json.dumps(chunk_data),
                vectors=vectors,
                metadata_json=json.dumps(metadata or {})
            )

        logger.info(
            "Saved %s chunks to index: %s",
            len(self.chunks),
            path
        )
        
    def load(
        self,
        index_path: str | Path
    ) -> dict[str, object]:
        """Load chunks and vectors from a compressed local index."""

        path = Path(index_path)

        # Pickle loading is disabled because pickle data can execute code when
        # opened. This index needs only standard NumPy arrays and a JSON string,
        # so allowing pickle would add risk without providing any benefit.
        with np.load(
            path,
            allow_pickle=False
        ) as index:
            
            if "metadata_json" in index.files:
                metadata = json.loads(
                    str(index["metadata_json"].item())
                )
            else:
                metadata = {}

            # NumPy returns the stored JSON value as a zero-dimensional array.
            # item() extracts its single value, str() makes it a Python string,
            # and json.loads() converts that string back into dictionaries.
            chunk_data = json.loads(
                str(index["chunks_json"].item())
            )

            # Restore the embedding matrix and enforce the same float32 type
            # used when embeddings are first added to the vector store.
            vectors = np.array(
                index["vectors"],
                dtype=np.float32
            )

        # Every saved chunk must still have exactly one saved vector.
        if len(chunk_data) != len(vectors):
            raise ValueError(
                "The stored chunks and vectors do not match"
            )

        # ** expands a dictionary into named arguments. For example,
        # Chunk(**data) becomes Chunk(text=..., source=..., page=...,
        # chunk_index=...), rebuilding the original dataclass object.
        self.chunks = [
            Chunk(**data)
            for data in chunk_data
        ]

        # Iterating through the matrix returns a view of each row. copy()
        # gives the vector store independent arrays it can safely retain.
        self.vectors = [
            vector.copy()
            for vector in vectors
        ]

        logger.info(
            "Loaded %s chunks from index: %s",
            len(self.chunks),
            path
        )
        
        return metadata
        
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None
    ) -> list[SearchResult]:
        
        if not self.chunks:
            logger.info("Vector store is empty")
            return []

        # Convert the user's question into the same vector space as the
        # document chunks so their semantic similarity can be compared.
        query_vector = np.array(
            self.embedder.embed_text(query),
            dtype=np.float32
        )

        results = []

        # Compare the question with every stored vector while keeping each
        # score attached to the correct Chunk metadata.
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

        # Highest cosine-similarity scores represent the closest matches.
        results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        # Remove weak matches before limiting the number of returned results.
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
