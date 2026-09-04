from collections.abc import Iterator
from ollama import Client
from llm.base import LLMProvider, Message
from logger import get_logger

logger = get_logger(__name__)

class OllamaProvider(LLMProvider):
    
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = Client(host=host)
        
    def stream_chat(
        self,
        messages: list[Message]
    ) -> Iterator[str]:

        logger.info(
            "Starting LLM request with model %s",
            self.model
        )

        try:

            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                think=False
            )
            
            for chunk in stream:
                
                text = chunk["message"]["content"]
            
                if text:
                    yield text

            logger.info("LLM request completed")
        
        except Exception:
            
            logger.exception(
                "LLM request failed"
            )

            raise
    
    def set_model(self, model: str) -> None:

        self.model = model

        logger.info(
            "Model changed to %s",
            model
        )