from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.vector_store import InMemoryVectorStore

test_questions = [
    {
        "question": "What is ORION-27",
        "expected_source": "rag_test_notes.txt"
    },
    {
        "question": "Which model is used as the development model?",
        "expected_source": "rag_test_document.pdf"
    },
    {
        "question": "What should happen before tool calling is needed?",
        "expected_source": "rag_test_notes.txt"
    },
    {
        "question": "What is the capital of Japan?",
        "expected_source": None
    },
    {
        "question": "How do I bake a chocolate cake?",
        "expected_source": None
    }
]

documents = load_documents(
    "documents"
)

chunks = chunk_documents(
    documents,
    chunk_size=100,
    overlap=20
)

embedder = OllamaEmbedder()

vector_store = InMemoryVectorStore(embedder=embedder)

vector_store.add_chunks(chunks)

for test_case in test_questions:

    question = test_case["question"]
    expected_source = test_case["expected_source"]

    results = vector_store.search(
        query=question,
        top_k=2
    )

    first_result = results[0]
    second_result = results[1]

    score_difference = (first_result.score - second_result.score)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Expected source: {expected_source}")
    print(
        f"Top source: "
        f"{first_result.chunk.source}"
    )
    print(
        f"Top score: "
        f"{first_result.score:.4f}"
    )
    print(
        f"Second score: "
        f"{second_result.score:.4f}"
    )
    print(
        f"Score difference: "
        f"{score_difference:.4f}"
    )