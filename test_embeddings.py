import numpy as np

from rag.embedder import OllamaEmbedder

embedder = OllamaEmbedder()


text_a = ("RAG retrieves information from external documents.")
text_b = ("Retrieval augmented generation searches a knowledge base.")
text_c = ("The ESP32 can communicate with the assistant over Wi-Fi.")

vector_a = embedder.embed_text(text_a)
vector_b = embedder.embed_text(text_b)
vector_c = embedder.embed_text(text_c)

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("RAG vs RAG:", cosine_similarity(vector_a, vector_b))

print("RAG vs ESP32:", cosine_similarity(vector_b, vector_c))