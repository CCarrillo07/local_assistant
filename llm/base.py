from abc import ABC, abstractmethod
from collections.abc import Iterator

Message = dict[str, str]

class LLMProvider(ABC):
    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
    ) -> Iterator[str]:
        pass
    
    @abstractmethod
    def set_model(self, model: str) -> None:
        pass