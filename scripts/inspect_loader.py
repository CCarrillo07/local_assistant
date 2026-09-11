from rag.loader import load_documents

documents = load_documents(
    "documents"
)

for document in documents:

    print("=" * 60)

    print(f"Source: {document.source}")

    print(f"Page: {document.page}")

    print()

    print(document.text[:500])