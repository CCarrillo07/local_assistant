from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from logger import get_logger
from rag.embedder import OllamaEmbedder
from rag.models import Chunk, SearchResult
from rag.vector_store_base import VectorStore

logger = get_logger(__name__)

class QdrantVectorStore(VectorStore):
    """Store and search document embeddings using local Qdrant."""
    
    def __init__(
        self,
        embedder: OllamaEmbedder,
        path: str | Path,
        collection_name: str
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        
        storage_path = Path(path)
        
        storage_path.mkdir(
            parents=True,
            exist_ok=True
        )
        
        # Local mode persists Qdrant data without required Docker
        # or a separately running Qdrant server
        self.client = QdrantClient(
            path=str(storage_path)
        )
        
    @property
    def size(self) -> int:
        """Return the number of chunks stored in the collection."""
        
        if not self.client.collection_exists(
            self.collection_name
        ):
            return 0
        
        return self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count
        
    def add_chunks(
        self,
        chunks: list[Chunk]
    ) -> None:
        """Embed and add chunks without replacing the collection."""
        
        embeddings = self._embed_chunks(chunks)
        
        if not embeddings:
            return
        
        if not self.client.collection_exists(
            self.collection_name
        ):
            self._create_collection(
                vector_size=len(embeddings[0])
            )
            
        self._upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            fingerprint=None
        )
        
    def restore(
        self,
        fingerprint: str
    ) -> bool:
        """Reuse the collection when every point has the current fingerprint."""
        
        if not self.client.collection_exists(
            self.collection_name
        ):
            return False
        
        total_points = self.size
        
        if total_points == 0:
            return False
        
        fingerprint_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="_index_fingerprint",
                    match=models.MatchValue(
                        value=fingerprint
                    )
                )
            ]
        )
        
        matching_points = self.client.count(
            collection_name=self.collection_name,
            count_filter=fingerprint_filter,
            exact=True
        ).count
        
        if matching_points != total_points:
            logger.info(
                "Qdrant collection is outdated; rebuilding"
            )
            return False
        
        return True
    
    def rebuild(
        self,
        chunks: list[Chunk],
        fingerprint: str
    ) -> None:
        """Replace the Qdrant collection with the supplied chunks."""
        
        # Generate embeddings before removing the existing collection.
        # If Ollama fails, the previous collection remains available.
        embeddings = self._embed_chunks(chunks)
                
        if self.client.collection_exists(
            self.collection_name
        ):
            self.client.delete_collection(
                collection_name=self.collection_name
            )
        
        if not embeddings:
            logger.info(
                "No chunks were provided for the Qdran collection"
            )
            return
                
        self._create_collection(
            vector_size=len(embeddings[0])
        )
        
        self._upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            fingerprint=fingerprint
        )
        
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None
    ) -> list[SearchResult]:
        """Return the chunks most similar to the supplied question."""
        
        if not self.client.collection_exists(
            self.collection_name
        ):
            logger.info("Qdrant collection does not exist")
            return []
        
        query_vector = self.embedder.embed_text(
            query
        )
        
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True
        ).points
        
        results = []
        
        for point in points:
            payload = point.payload or {}
            
            chunk = Chunk(
                text=str(payload["text"]),
                source=str(payload["source"]),
                page=payload.get("page"),
                chunk_index=int(
                    payload["chunk_index"]
                )
            )
            
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(point.score)
                )
            )
            
        return results
        
    def close(self) -> None:
        """Release the local Qdrant storage files."""
        
        self.client.close()
        
    def _embed_chunks(
        self,
        chunks: list[Chunk]
    ) -> list[list[float]]:
        """Generate one embedding for every supplied chunk."""
        
        if not chunks:
            return []
        
        embeddings = self.embedder.embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )
        
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "The number of chunks and embeddings "
                "must match"
            )
            
        return embeddings
    
    def _create_collection(
        self,
        vector_size: int
    ) -> None:
        """Create a cosine-similarity Qdrant collection."""
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
    def _upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        fingerprint: str | None
    ) -> None:
        """Insert or replace chunks using stable identifiers."""
        
        points = []
        
        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            # The same chunk produces the same UUID, preventing 
            # accidental duplicate records during repetead writes.
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    (
                        f"{chunk.source}|{chunk.page}|"
                        f"{chunk.chunk_index}|{chunk.text}"
                    )
                )
            )
            
            payload = {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index
            }
            
            if fingerprint is not None:
                payload["_index_fingerprint"] = fingerprint
                
                
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
            
        logger.info(
            "Stored %s chunks in Qdrant collection: %s",
            len(points),
            self.collection_name
        )