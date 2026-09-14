import ollama

from logger import get_logger

logger = get_logger(__name__)

class OllamaEmbedder:

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        query_prefix: str = "",
        document_prefix: str = ""
    ):

        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_text(
        self,
        text: str
    ) -> list[float]:
        
        prepared_text = (
            f"{self.query_prefix}{text}"
        )

        return self._embed_texts([prepared_text])[0]

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        
        prepared_texts = [
            f"{self.document_prefix}{text}"
            for text in texts
        ]

        return self._embed_texts(
            prepared_texts
        )
        
    def _embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """Send prepared text to Ollama."""
        
        if not texts:
            return []
        
        logger.info(
            "Embedding %s text(s) with model: %s",
            len(texts),
            self.model
        )
        
        response = ollama.embed(
            model=self.model,
            input=texts
        )
        
        embeddings = response["embeddings"]
        
        logger.info(
            "Generated %s embeddings with %s dimensions",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0
        )
            
        return embeddings