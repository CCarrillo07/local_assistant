import unittest

from rag.models import Chunk, SearchResult
from rag.prompt import build_rag_prompt


class RAGPromptTests(unittest.TestCase):

    def test_includes_all_retrieved_chunks(self):

        results = [
            SearchResult(
                chunk=Chunk(
                    text="First retrieved fact.",
                    source="document.pdf",
                    page=2,
                    chunk_index=0
                ),
                score=0.90
            ),
            SearchResult(
                chunk=Chunk(
                    text="Second retrieved fact.",
                    source="notes.txt",
                    page=None,
                    chunk_index=1
                ),
                score=0.80
            )
        ]

        prompt = build_rag_prompt(
            question="What are the facts?",
            results=results
        )

        self.assertIn(
            "First retrieved fact.",
            prompt
        )

        self.assertIn(
            "Second retrieved fact.",
            prompt
        )

        self.assertIn(
            "[Context passage 1]",
            prompt
        )

        self.assertIn(
            "[Context passage 2]",
            prompt
        )

        self.assertIn(
            "Never create hyperlinks, URLs,",
            prompt
        )

    def test_does_not_expose_source_metadata_to_model(self):
        """Keep citation metadata outside the model prompt."""

        results = [
            SearchResult(
                chunk=Chunk(
                    text="A fact from a document.",
                    source="private_document.pdf",
                    page=7,
                    chunk_index=3
                ),
                score=0.90
            )
        ]

        prompt = build_rag_prompt(
            question="What is the fact?",
            results=results
        )

        self.assertNotIn(
            "private_document.pdf",
            prompt
        )
        self.assertNotIn(
            "page 7",
            prompt
        )
        self.assertNotIn(
            "chunk 3",
            prompt
        )

    def test_builds_abstention_prompt_without_results(self):

        prompt = build_rag_prompt(
            question="An unsupported question",
            results=[]
        )

        self.assertIn(
            "The answer could not be found "
            "in the documents.",
            prompt
        )

        self.assertIn(
            "Do not answer the question "
            "using general knowledge.",
            prompt
        )

    def test_tells_model_not_to_generate_citations(self):

        results = [
            SearchResult(
                chunk=Chunk(
                    text="A grounded fact.",
                    source="verified_source.txt",
                    page=None,
                    chunk_index=0
                ),
                score=0.95
            )
        ]

        prompt = build_rag_prompt(
            question="What is the fact?",
            results=results
        )

        self.assertIn(
            "Do not include citations, filenames, "
            "page numbers, or source labels",
            prompt
        )

        self.assertIn(
            "The application will display "
            "verified sources separately.",
            prompt
        )


if __name__ == "__main__":
    unittest.main()
