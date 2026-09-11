from config import CONFIG
from llm.base import Message
from llm.ollama_provider import OllamaProvider
from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.prompt import build_rag_prompt
from rag.vector_store import InMemoryVectorStore

documents = load_documents("documents")

chunks = chunk_documents(documents,chunk_size=100,overlap=20)

embedder = OllamaEmbedder()

vector_store = InMemoryVectorStore(embedder=embedder)

vector_store.add_chunks(chunks)

question = "What is ORION-27?"

results = vector_store.search(
    query=question,
    top_k=3
)

rag_prompt = build_rag_prompt(
    question=question,
    results=results
)

provider = OllamaProvider(
    model=CONFIG.default_model,
    host=CONFIG.ollama_host
)

messages: list[Message] = [
    {
        "role": "system",
        "content": (
            "You are a document question-answering assistant. "
            "Follow the instructions in the user's prompt and "
            "do not invent information."
        )
    },
    {
        "role": "user",
        "content": rag_prompt
    }
]

print(f"\nQuestion: {question}")
print("\nAnswer: ", end="")


for text in provider.stream_chat(messages):
    print(text,end="",flush=True)

print()