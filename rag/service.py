import hashlib
import json
from pathlib import Path

from logger import get_logger
from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.prompt import build_rag_prompt
from rag.vector_store import InMemoryVectorStore

logger = get_logger(__name__)

class RAGService:
    
    def __init__(
        self,
        document_directory: str | Path,
        embedder: OllamaEmbedder,
        index_path: str | Path | None = None,
        chunk_size: int = 100,
        overlap: int = 20,
        top_k: int = 2,
        min_score: float = 0.42
    ):
        
        self.document_directory = document_directory
        
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else None
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
       
        fingerprint = None
       
        if self.index_path is not None:
           
           fingerprint = (
               self._calculate_index_fingerprint()
           )
           
           if self.index_path.exists():
               
               metadata = self.vector_store.load(
                   self.index_path
               )
               
               if (
                   metadata.get("fingerprint") == fingerprint
               ):
                   self.initialized = True

                   logger.info(
                       "RAG service initialized from "
                       "persisted index with %s chunks",
                       len(self.vector_store.chunks)
                   )
                   
                   return
               
               logger.info(
                   "Persisted RAG index is outdated; "
                   "rebuilding"
               )
               
        documents = load_documents(
            self.document_directory
        )
        
        chunks = chunk_documents(
            documents=documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        self.vector_store = InMemoryVectorStore(
            embedder=self.embedder
        )
        
        self.vector_store.add_chunks(
            chunks
        )
        
        if self.index_path is not None:
            self.vector_store.save(
                self.index_path,
                metadata={
                    "fingerprint": fingerprint
                }
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