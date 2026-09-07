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
            question="What are the fact?",
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
            "document.pdf, page 2",
            prompt
        )
        
        self.assertIn(
            "notes.txt",
            prompt
        )
        
    def test_does_not_add_page_to_txt_source(self):
        
        results = [
            SearchResult(
                chunk=Chunk(
                    text="A fact from a text file.",
                    source="notes.txt",
                    page=None,
                    chunk_index=1
                ),
                score=0.90
            )
        ]
        
        prompt = build_rag_prompt(
            question="What is the fact?",
            results=results
        )
        
        self.assertIn(
            "notes.txt, chunk 3",
            prompt
        )
        
        self.assertNotIn(
            "notes.txt, page",
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
        
if __name__ == "__main__":
    unittest.main()