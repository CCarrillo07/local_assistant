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
        chunk_size: int = 100,
        overlap: int = 20,
        top_k: int = 2,
        min_score: float = 0.42
    ):
        
        self.document_directory = document_directory
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_score = min_score
        
        self.vector_store = InMemoryVectorStore(embedder=embedder)
        
        self.initialized = False

    def initialize(self) -> None:
        
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