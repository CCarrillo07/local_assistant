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

        chunk = result.chunk

        source_location = chunk.source

        if chunk.page is not None:
            source_location += f" , page {chunk.page}"

        context_section = (
            f"[Source {position}: "
            f"{source_location}, "
            f"chunk {chunk.chunk_index}]\n"
            f"{chunk.text}"
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
3. Cite the exact source filename supporting the answer.
4. Include a page number only when the source metadata
explicitly contains a page value.
5. Never interpret a chunk number or source position as page number.
6. If the answer is not present, say:
"The answer could not be found in the documents."

Question:
{question}

Context:
{context}
""".strip()