from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from rag.models import Chunk, Document

# Included in the index fingerprint so changing the chunking
# algorithm causes the persisted vector index to be rebuilt.
CHUNKING_STRATEGY = "recursive_character_v1"

def _count_words(text: str) -> int:
    """Return the number of words in a piece of text."""
    
    return len(text.split())

def _create_text_splitter(
    chunk_size: int,
    overlap: int
) -> RecursiveCharacterTextSplitter:
    """Create and configure the splitter used by the RAG pipeline."""
    
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )
        
    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )
        
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )
        
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        
        # Keep chunk_size and overlap measured in words,
        # matching the behavior of our previous chunker.
        length_function=_count_words,
        
        # Try natural boundaries first. Smaller boundaries are
        # used only when a section is too large.
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        
        # Keep punctuation with the sentence it belongs to.
        keep_separator="end",
        strip_whitespace=True
    )
    
def chunk_document(
    document: Document,
    chunk_size: int = 200,
    overlap: int = 40
) -> list[Chunk]:
    """Split one document while preserving its metadata."""
    
    splitter = _create_text_splitter(
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    split_texts = splitter.split_text(
        document.text
    )
    
    chunks = []
    
    for chunk_index, text in enumerate(split_texts):
        
        chunks.append(
            Chunk(
                text=text,
                source=document.source,
                page=document.page,
                chunk_index=chunk_index
            )
        )
        
    return chunks

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 200,
    overlap: int = 40
) -> list[Chunk]:
    """Split multiple documents into one collection of chunks."""
    
    chunks = []
    
    for document in documents:
        
        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        chunks.extend(document_chunks)
        
    return chunks
