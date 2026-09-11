import hashlib
import json
from pathlib import Path

from logger import get_logger
from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.prompt import build_rag_prompt
from rag.vector_store import InMemoryVectorStore
from rag.vector_store import VectorStore

logger = get_logger(__name__)

class RAGService:
    
    def __init__(
        self,
        document_directory: str | Path,
        embedder: OllamaEmbedder,
        index_path: str | Path | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 100,
        overlap: int = 20,
        top_k: int = 2,
        min_score: float = 0.42
    ):
        
        self.document_directory = document_directory
        
        self.vector_store = (
            vector_store 
            if vector_store is not None
            else InMemoryVectorStore(
                embedder=embedder,
                index_path=index_path
            )
        )
        
        
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_score = min_score
        
        self.vector_store = InMemoryVectorStore(embedder=embedder)
        
        self.initialized = False

    def _calculate_index_fingerprint(self) -> str:
        """Create an identifier for the documents and RAG configuration."""
        
        document_directory = Path(
            self.document_directory
        )
        
        fingerprint_data = {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": getattr(
                self.embedder,
                "model",
                self.embedder.__class__.__name__
            )
        }
        
        hasher = hashlib.sha256()
        
        hasher.update(
            json.dumps(
                fingerprint_data,
                sort_keys=True
            ).encode("utf-8")
        )
        
        supported_files = sorted(
            (
                path
                for path in document_directory.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower() in {".pdf", ".txt"}
                )
            ),
            key=lambda path: path.as_posix().lower()
        )
        
        for path in supported_files:
            
            relative_path = path.relative_to(
                document_directory
            )
            
            hasher.update(
                relative_path.as_posix().encode("utf-8")
            )
            
            hasher.update(
                path.read_bytes()
            )
            
        return hasher.hexdigest()
    
    def initialize(self) -> None:
       
        fingerprint = self._calculate_index_fingerprint()
        
        # The vector-store implementation decides how its persisted index
        # is restored. RAGService does not need to know whether it uses
        # as NPZ file, Qdrant, or another storage system.
        if self.vector_store.restore(fingerprint):
            self.initialized = True

            logger.info(
                "RAG service initialized from "
                "persisted index with %s chunks",
                self.vector_store.size
            )
                                  
        documents = load_documents(
            self.document_directory
        )
        
        chunks = chunk_documents(
            documents=documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        self.vector_store.rebuild(
            chunks=chunks,
            fingerprint=fingerprint
        )
        
        self.initialized = True

        logger.info(
            "RAG service initialized with %s chunks",
            len(chunks)
        )
       
    def build_prompt(
        self,
        question: str
    ) -> str | None:
        
        if not self.initialized:
            raise RuntimeError(
                "RAG service must be initialized "
                "before processing questions"
            )
            
        results = self.vector_store.search(
            query=question,
            top_k=self.top_k,
            min_score=self.min_score
        )
        
        if not results:
            return None
        
        return build_rag_prompt(
            question=question,
            results=results
        )