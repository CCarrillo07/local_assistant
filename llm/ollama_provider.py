from collections.abc import Iterator
from ollama import Client

from llm.base import LLMProvider, Message

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
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True

        )
        
        for chunk in stream:
            
            text = chunk["message"]["content"]
        
            if text:
                yield text
    
    def set_model(self, model: str) -> None:
        self.model = model