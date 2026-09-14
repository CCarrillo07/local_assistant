from config import RAGConfig
from logger import get_logger
from rag.embedder import OllamaEmbedder
from rag.vector_store import InMemoryVectorStore
from rag.vector_store_base import VectorStore

logger = get_logger(__name__)

def create_vector_store(
    config: RAGConfig,
    embedder: OllamaEmbedder
) -> VectorStore:
    """Create the vector_store backend selected in configuration."""
    
    logger.info(
        "Using vector-store backend: %s",
        config.vector_store_backend
    )
    
    if config.vector_store_backend == "npz":
        return InMemoryVectorStore(
            embedder=embedder,
            index_path=config.index_path
        )
    
    if config.vector_store_backend == "qdrant":
        
        # Import Qdrant only when that backend is selected.
        # This keeps the NPZ implementation independent.
        from rag.qdrant_vector_store import(
            QdrantVectorStore
        )
        
        return QdrantVectorStore(
            embedder=embedder,
            path=config.qdrant_path,
            collection_name=config.qdrant_collection
        )
        
    # RAGConfig normally prevents this path, but the factory
    # still protects itself from unsupported input.
    raise ValueError(
        "Invalid vector-store backend: "
        f"{config.vector_store_backend}"
    )
    