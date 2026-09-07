import ollama

from logger import get_logger

logger = get_logger(__name__)

class OllamaEmbedder:

    def __init__(
        self,
        model: str = "nomic-embed-text:latest"
    ):

        self.model = model

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

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