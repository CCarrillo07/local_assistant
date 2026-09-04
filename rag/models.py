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