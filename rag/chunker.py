from rag.models import Chunk, Document

def chunk_document(
    document: Document,
    chunk_size: int = 200,
    overlap: int = 40
) -> list[Chunk]:

    words = document.text.split()

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunk = Chunk(
            text=chunk_text,
            source=document.source,
            page=document.page,
            chunk_index=chunk_index
        )

        chunks.append(chunk)

        start += chunk_size - overlap

        chunk_index += 1

    return chunks

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 200,
    overlap: int = 40
) -> list[Chunk]:

    chunks = []

    for document in documents: 

        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunks.extend(document_chunks)

    return chunks