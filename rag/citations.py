from rag.models import SearchResult

def format_sources(
    results: list[SearchResult]
) -> str:
    """Create a unique, consistently formatted source list."""

    if not results:
        return ""

    source_lines = []
    seen_locations = set()

    for result in results:

        chunk = result.chunk
        location = chunk.source

        if chunk.page is not None:
            location += f", page {chunk.page}"

        if location in seen_locations:
            continue

        seen_locations.add(location)

        source_lines.append(
            f"- {location}"
        )

    return (
        "Sources:\n"
        + "\n".join(source_lines)
    )