import unittest

from rag.chunker import chunk_document
from rag.models import Document


class ChunkerTest(unittest.TestCase):

    def test_preserves_metadata_and_assigns_indexes(self):
        document = Document(
            text=(
                "Alpha bravo charlie delta echo foxtrot golf hotel "
                "india juliet kilo lima mike november oscar papa."
            ),
            source="example.txt",
            page=3
        )

        chunks = chunk_document(
            document=document,
            chunk_size=5,
            overlap=1
        )

        self.assertGreater(
            len(chunks),
            1
        )

        for chunk in chunks:
            self.assertEqual(
                chunk.source,
                "example.txt"
            )

            self.assertEqual(
                chunk.page,
                3
            )

        self.assertEqual(
            [
                chunk.chunk_index
                for chunk in chunks
            ],
            list(range(len(chunks)))
        )

    def test_rejects_invalid_chunk_configuration(self):
        document = Document(
            text="Example document text.",
            source="example.txt",
            page=None
        )

        with self.assertRaisesRegex(
            ValueError,
            "overlap must be smaller than chunk_size"
        ):
            chunk_document(
                document=document,
                chunk_size=5,
                overlap=5
            )


if __name__ == "__main__":
    unittest.main()
