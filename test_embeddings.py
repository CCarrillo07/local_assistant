import numpy as np

from rag.embedder import OllamaEmbedder

embedder = OllamaEmbedder()

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    similarity = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return float(similarity)

embedder = OllamaEmbedder()

texts = [
    "RAG retrieves information from external documents.",
    "Retrieval augmented generation searches a knowledge base.",
    "The ESP32 can communicate with the assistant over Wi-Fi."
]

vectors = embedder.embed_texts(texts)

vector_a = vectors[0]
vector_b = vectors[1]
vector_c = vectors[2]

print(f"\nEmbeddings: {len(vectors)}")
print(f"\nDimensions: {len(vector_a)}")

print("RAG vs RAG:", cosine_similarity(vector_a,vector_b))
print("RAG vs ESP32:", cosine_similarity(vector_b,vector_c))
