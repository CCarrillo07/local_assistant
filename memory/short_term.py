from llm.base import Message
from memory.base import ConversationMemory

class SlidingWindowMemory(ConversationMemory):
    """Retain only the most recent complete conversation turns."""

    def __init__(
        self,
        max_turns: int = 10
    ):
        if max_turns <= 0:
            raise ValueError(
                "max_turns must be positive"
            )

        self.max_turns = max_turns
        self._messages: list[Message] = []

    def add_turn(
        self,
        user_message: str,
        assistant_message: str
    ) -> None:
        self._messages.extend(
            [
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ]
        )

        maximum_messages = (
            self.max_turns * 2
        )

        if len(self._messages) > maximum_messages:
            self._messages = self._messages[
                -maximum_messages:
            ]

    def get_messages(self) -> list[Message]:
        return [
            message.copy()
            for message in self._messages
        ]

    def clear(self) -> None:
        self._messages = []