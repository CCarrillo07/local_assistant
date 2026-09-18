from dataclasses import dataclass, field

MODELS = {
    "fast": "qwen3.5:2b-q4_K_M",
    "main": "qwen3.5:2b-q8_0"
}

@dataclass(frozen=True)
class RAGConfig:
    available: bool = True
    mode: str = "manual"
    allow_user_control: bool = True
    document_directory: str = "documents"
    vector_store_backend: str = "qdrant"
    index_path: str = "storage/rag_index.npz"
    qdrant_path: str = "storage/qdrant"
    qdrant_collection: str = "rag_documents"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    chunk_size: int = 100
    overlap: int = 20
    top_k: int = 2
    min_score: float = 0.42
    candidate_k: int = 5
    reranker_enabled: bool = True
    reranker_backend: str = "flashrank"
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_cache_dir: str = "storage/flashrank"
    reranker_max_length: int = 128
    reranker_min_score: float | None = 0.10

    def __post_init__(self) -> None:

        valid_modes = {
            "manual",
            "auto",
            "required"
        }

        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid RAG mode: {self.mode}"
            )

        user_control_expected = (
            self.available
            and self.mode == "manual"
        )

        if self.allow_user_control != user_control_expected:
            raise ValueError(
                "allow_user_control must be True only "
                "when RAG is available and mode is manual"
            )
        
        valid_vector_store_backends = {
            "npz",
            "qdrant"
        }
        
        if (
            self.vector_store_backend
            not in valid_vector_store_backends
        ):
            raise ValueError(
                "Invalid vector-store backend: "
                f"{self.vector_store_backend}"
            )

        if self.candidate_k < self.top_k:
            raise ValueError(
                "candidate_k must be greater than "
                "or equal to top_k"
            )

        valid_reranker_backends = {
            "flashrank"
        }

        if (
            self.reranker_backend
            not in valid_reranker_backends
        ):
            raise ValueError(
                "Invalid reranker backend: "
                f"{self.reranker_backend}"
            )

        if self.reranker_max_length <= 0:
            raise ValueError(
                "reranker_max_length must be positive"
            )

        if (
            self.reranker_min_score is not None
            and not 0.0 <= self.reranker_min_score <= 1.0
        ):
            raise ValueError(
                "reranker_min_score must be between "
                "0.0 and 1.0"
            )

@dataclass(frozen=True)
class MemoryConfig:
    short_term_enabled: bool = True
    short_term_max_turns: int = 10

    def __post_init__(self) -> None:
        if self.short_term_max_turns <= 0:
            raise ValueError(
                "short_term_max_turns must be positive"
            )

@dataclass(frozen=True)
class AppConfig:
    ollama_host: str = "http://localhost:11434"
    default_model: str = MODELS["main"]
    memory: MemoryConfig = field(
        default_factory=MemoryConfig
    )
    rag: RAGConfig = field(
        default_factory=RAGConfig
    )

CONFIG = AppConfig()

def resolve_model(name: str) -> str:
    return MODELS.get(name, name)
