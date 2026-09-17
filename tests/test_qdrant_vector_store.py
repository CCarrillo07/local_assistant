import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.models import Chunk
from rag.qdrant_vector_store import QdrantVectorStore

class FakeEmbedder:
    
    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        
        return [
            self.embed_text(text)
            for text in texts
        ]
        
    def embed_text(
        self,
        text: str
    ) -> list[float]:
        
        if "ORION-27" in text:
            return [1.0, 0.0]
            
        return [0.0, 1.0]
    
class TrackingEmbedder(FakeEmbedder):
    
    def __init__(self):
        self.embedded_batches: list[list[str]] = []
    
    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        self.embedded_batches.append(
            list(texts)
        )
        
        return super().embed_texts(
            texts
        )


class QdrantVectorStoreTest(unittest.TestCase):
    
    def test_rebuilds_restores_and_searches_collection(
        self
    ):
        chunks = [
            Chunk(
                text="ORION-27 is the internal codename.",
                source="notes.txt",
                page=None,
                chunk_index=0
            ),
            Chunk(
                text="The assistant can run locally.",
                source="overview.txt",
                page=None,
                chunk_index=0
            )
        ]
        
        with TemporaryDirectory() as temporary_directory:
            
            storage_path = (
                Path(temporary_directory)
                / "qdrant"
            )
            
            store = QdrantVectorStore(
                embedder=FakeEmbedder(),
                path=storage_path,
                collection_name="test_documents"
            )
            
            try:
                store.rebuild(
                    chunks=chunks,
                    fingerprint="test-fingerprint"
                )
                
                self.assertEqual(
                    store.size,
                    2
                )
                
            finally:
                store.close()
                
            # Opening another instance simulates restarting
            # the assistant and reusing Qdrant data
            restored_store = QdrantVectorStore(
                embedder=FakeEmbedder(),
                path=storage_path,
                collection_name="test_documents"
            )
            
            try: 
                self.assertTrue(
                    restored_store.restore(
                        "test-fingerprint"
                    )
                )
                
                results = restored_store.search(
                    query="What is ORION-27?",
                    top_k=2,
                    min_score=None
                )
                
                self.assertEqual(
                    len(results),
                    2
                )
                
                self.assertEqual(
                    results[0].chunk.source,
                    "notes.txt"
                )
                
            finally:
                restored_store.close()
                
    def test_add_chunks_creates_collection(self):
        
        with TemporaryDirectory() as temporary_directory:
            
            store = QdrantVectorStore(
                embedder=FakeEmbedder(),
                path=Path(temporary_directory) / "qdrant",
                collection_name="test_documents"
            )
            
            try:
                store.add_chunks(
                    [
                        Chunk(
                            text="ORION-27 is a codename.",
                            source="notes.txt",
                            page=None,
                            chunk_index=0
                        )
                    ]
                )
                
                self.assertEqual(
                    store.size,
                    1
                )
                
            finally:
                store.close()

    def test_rebuild_replaces_existing_collection(self):
        old_chunks = [
            Chunk(
                text="ORION-27 is the internal codename.",
                source="old_notes.txt",
                page=None,
                chunk_index=0
            ),
            Chunk(
                text="The assistant can run locally.",
                source="old_overview.txt",
                page=None,
                chunk_index=0
            )
        ]

        replacement_chunk = Chunk(
            text="This is the replacement document.",
            source="replacement.txt",
            page=None,
            chunk_index=0
        )

        with TemporaryDirectory() as temporary_directory:
            store = QdrantVectorStore(
                embedder=FakeEmbedder(),
                path=(
                    Path(temporary_directory)
                    / "qdrant"
                ),
                collection_name="test_documents"
            )

            try:
                store.rebuild(
                    chunks=old_chunks,
                    fingerprint="old-fingerprint"
                )

                self.assertEqual(
                    store.size,
                    2
                )

                store.rebuild(
                    chunks=[replacement_chunk],
                    fingerprint="new-fingerprint"
                )

                self.assertEqual(
                    store.size,
                    1
                )

                self.assertTrue(
                    store.restore(
                        "new-fingerprint"
                    )
                )

                results = store.search(
                    query="replacement document",
                    top_k=10,
                    min_score=None
                )

                self.assertEqual(
                    len(results),
                    1
                )

                self.assertEqual(
                    results[0].chunk.source,
                    "replacement.txt"
                )

                old_sources = {
                    "old_notes.txt",
                    "old_overview.txt"
                }

                self.assertTrue(
                    all(
                        result.chunk.source not in old_sources
                        for result in results
                    )
                )

            finally:
                store.close()
    
    def test_synchronize_updates_only_changed_chunks(self):
        """Reuse unchanged vectors while reconciling the collection."""
        
        unchanged_chunk = Chunk(
            text="ORION-27 is the internal codename.",
            source="notes.txt",
            page=None,
            chunk_index=0
        )
        deleted_chunk = Chunk(
            text="This document will be deleted.",
            source="deleted.txt",
            page=None,
            chunk_index=0
        )
        new_chunk = Chunk(
            text="This document was added later.",
            source="new.txt",
            page=None,
            chunk_index=0
        )
        
        with TemporaryDirectory() as temporary_directory:
            embedder = TrackingEmbedder()
            store = QdrantVectorStore(
                embedder=embedder,
                path=(
                    Path(temporary_directory)
                    / "qdrant"
                ),
                collection_name="test_documents"
            )
            
            try:
                store.rebuild(
                    chunks=[
                        unchanged_chunk,
                        deleted_chunk
                    ],
                    fingerprint="old-fingerprint"
                )
                
                embedder.embedded_batches.clear()
                
                store.synchronize(
                    chunks=[
                        unchanged_chunk,
                        new_chunk
                    ],
                    fingerprint="new-fingerprint"
                )
                
                self.assertEqual(
                    embedder.embedded_batches,
                    [[new_chunk.text]]
                )
                self.assertEqual(
                    store.size,
                    2
                )
                self.assertTrue(
                    store.restore(
                        "new-fingerprint"
                    )
                )
                
                results = store.search(
                    query="ORION-27",
                    top_k=10,
                    min_score=None
                )
                stored_sources = {
                    result.chunk.source
                    for result in results
                }
                
                self.assertEqual(
                    stored_sources,
                    {
                        "notes.txt",
                        "new.txt"
                    }
                )
                
            finally:
                store.close()
                                 
                
if __name__ == "__main__":
    unittest.main()
