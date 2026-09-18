from abc import ABC, abstractmethod

from llm.base import Message

class ConversationMemory(ABC):
    """Common contract for conversation-memory implementations."""

    @abstractmethod
    def add_turn(
        self,
        user_message: str,
        assistant_message: str
    ) -> None:
        """Store one complete conversation turn."""
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Return messages available to the model."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored conversation turns."""
        raise NotImplementedError