from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.vector_store import InMemoryVectorStore

documents = load_documents("documents")

chunks = chunk_documents(documents,chunk_size=100,overlap=20)

embedder = OllamaEmbedder()

vector_store = InMemoryVectorStore(
    embedder=embedder
)

vector_store.add_chunks(chunks)

query = "What is ORION-27?"

results = vector_store.search(
    query=query,
    top_k=3
)

print(f"\nQuery: {query}")
print(f"Results: {len(results)}")

for position, result in enumerate(
    results,
    start=1
):
    chunk = result.chunk

    print("=" * 60)
    print(f"Position: {position}")
    print(f"Score: {result.score:.4f}")
    print(f"Source: {chunk.source}")
    print(f"Page: {chunk.page}")
    print(f"Chunk: {chunk.chunk_index}")
    print()
    print(chunk.text)
