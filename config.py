from dataclasses import dataclass, field

MODELS = {
    "fast": "qwen3.5:2b-q4_K_M",
    "main": "qwen3.5:2b-q8_0"
}

@dataclass(frozen=True)
class RAGConfig:
    document_directory: str = "documents"
    embedding_model: str = "nomic-embed-text:latest"
    chunk_size: int = 100
    overlap: int = 20
    top_k: int = 2
    min_score: float = 0.42
    available: bool = True
    mode: str = "manual"
    allow_user_control: bool = True

@dataclass(frozen=True)
class AppConfig:
    ollama_host: str = "http://localhost:11434"
    default_model: str = MODELS["main"]
    rag: RAGConfig = field(
        default_factory=RAGConfig
    )
    
CONFIG = AppConfig()

def resolve_model(name: str) -> str:
    return MODELS.get(name, name)
