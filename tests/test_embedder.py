import unittest
from unittest.mock import patch

from rag.embedder import OllamaEmbedder

class OllamaEmbeddertest(unittest.TestCase):
    
    @patch("rag.embedder.ollama.embed")
    def test_uses_query_and_document_prefixes(
        self,
        mock_embed
    ):
        mock_embed.return_value = {
            "embeddings": [[0.1, 0.2]]
        }
        
        embedder = OllamaEmbedder(
            model="test-model",
            query_prefix="search_query: ",
            document_prefix="search_document: "
        )
        
        embedder.embed_text("What is ORION-27?")
        embedder.embed_texts(["ORION-27 is a codename."])
        
        query_input = (
            mock_embed.call_args_list[0].kwargs["input"]
        )
        
        document_input = (
            mock_embed.call_args_list[1].kwargs["input"]
        )
        
        self.assertEqual(
            query_input,
            ["search_query: What is ORION-27?"]
        )
        
        self.assertEqual(
            document_input,
            [
                "search_document: "
                "ORION-27 is a codename."
            ]
        )
        
if __name__ == "__main__":
    unittest.main()
