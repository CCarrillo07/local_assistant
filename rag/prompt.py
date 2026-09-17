from rag.models import SearchResult

def build_rag_prompt(
    question:str,
    results: list[SearchResult]
) -> str:
    
    if not results:
        return f"""
    No sufficiently relevant document context was retrieved.
    
    Do not answer the question using general knowledge.
    Reply with exactly:

    The answer could not be found in the documents.
    """.strip()

    context_sections = []

    for position, result in enumerate(
        results,
        start=1
    ):

        context_section = (
            f"[Context passage {position}]\n"
            f"{result.chunk.text}"
        )

        context_sections.append(
            context_section
        )

    context = "\n\n".join(
        context_sections
    )

    return f"""
Answer the question using only the provided context.

Rules:
1. Preserve the facts exactly as stated in the context.
2. Do not add assumptions or unsupported descriptions.
3. Do not include citations, filenames, page numbers, or source labels in your answer.
Never create hyperlinks, URLs, directory paths, or other source locations.
4. The application will display verified sources separately.
5. If the answer is not present, say:
"The answer could not be found in the documents."

Question:
{question}

Context:
{context}
""".strip()