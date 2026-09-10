from dataclasses import dataclass, field

MODELS = {
    "fast": "qwen3.5:2b-q4_K_M",
    "main": "qwen3.5:2b-q8_0"
}

@dataclass(frozen=True)
class RAGConfig:
    available: bool = True
    mode: str = "auto"
    allow_user_control: bool = False
    document_directory: str = "documents"
    embedding_model: str = "nomic-embed-text:latest"
    chunk_size: int = 100
    overlap: int = 20
    top_k: int = 2
    min_score: float = 0.42

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
