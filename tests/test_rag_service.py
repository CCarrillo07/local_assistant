import unittest

from rag.service import RAGService

class FakeEmbedder:
    
    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        
        return [
            [1.0, 0.0]
            for _ in texts
        ]
        
    def embed_text(
        self,
        text: str
    ) -> list[float]:
        
        return [1.0, 0.0]
    
class RAGServiceTest(unittest.TestCase):
    
    def test_requires_initialization_before_building_prompt(self):
        
        service = RAGService(
            document_directory="documents",
            embedder=FakeEmbedder()
        )
        
        with self.assertRaisesRegex(
            RuntimeError,
            "must be initialized"
        ):
            service.build_prompt(
                "What is ORION-27?"
            )
            
if __name__ == "__main__":
    unittest.main()