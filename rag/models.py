from dataclasses import dataclass

@dataclass
class Document:
    text: str
    source: str
    page: int | None = None

@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_index: int

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

@dataclass
class RAGContext:
    """Grounded prompt and the results support it."""

    prompt: str
    results: list[SearchResult]

