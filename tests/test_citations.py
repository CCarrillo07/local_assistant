import unittest

from rag.citations import format_sources
from rag.models import Chunk, SearchResult


class CitationFormatterTest(unittest.TestCase):

    def test_formats_unique_sources_in_retrieval_order(self):
        """Build a stable source list without duplicate locations."""

        results = [
            SearchResult(
                chunk=Chunk(
                    text="First PDF passage",
                    source="employee_guide.pdf",
                    page=3,
                    chunk_index=0
                ),
                score=0.98
            ),
            SearchResult(
                chunk=Chunk(
                    text="Another passage from the same page",
                    source="employee_guide.pdf",
                    page=3,
                    chunk_index=1
                ),
                score=0.93
            ),
            SearchResult(
                chunk=Chunk(
                    text="A passage from a text file",
                    source="project_notes.txt",
                    page=None,
                    chunk_index=0
                ),
                score=0.89
            )
        ]

        self.assertEqual(
            format_sources(results),
            (
                "Sources:\n"
                "- employee_guide.pdf, page 3\n"
                "- project_notes.txt"
            )
        )

    def test_returns_empty_text_without_results(self):
        """Do not print an empty Sources heading."""

        self.assertEqual(
            format_sources([]),
            ""
        )


if __name__ == "__main__":
    unittest.main()
