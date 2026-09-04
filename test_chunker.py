from rag.chunker import chunk_documents
from rag.loader import load_documents

documents = load_documents(
    "documents"
)

chunks = chunk_documents(
    documents,
    chunk_size=100,
    overlap=20
)

print(f"\nDocuments: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for chunk in chunks:

    print("=" * 60)

    print(f"Source: {chunk.source}")
    print(f"Page: {chunk.page}")
    print(f"Chunk: {chunk.chunk_index}")

    print()

    print(chunk.text)