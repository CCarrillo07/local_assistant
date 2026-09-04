from dataclasses import dataclass

MODELS = {
    "fast": "qwen3.5:2b-q4_K_M",
    "main": "qwen3.5:2b-q8_0"
}

@dataclass(frozen=True)
class AppConfig:
    ollama_host: str = "http://localhost:11434"
    default_model: str = MODELS["main"]
    
CONFIG = AppConfig()

def resolve_model(name: str) -> str:
    return MODELS.get(name,name)
