import ollama

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

        response = ollama.embed(
            model=self.model,
            input=text
        )

        return response["embeddings"][0] 