from llm.base import Message
from memory.base import ConversationMemory

class NullMemory(ConversationMemory):
    """Discard conversation turns when memory is disabled"""

    def add_turn(
        self,
        user_message: str,
        assistant_message: str
    ) -> None:
        pass

    def get_messages(self) -> list[Message]:
        return []

    def clear(self) -> None:
        pass