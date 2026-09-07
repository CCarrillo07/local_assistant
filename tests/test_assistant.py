import unittest 
from collections.abc import Iterator

from assistant import Assistant
from llm.base import LLMProvider, Message

class FakeLLMProvider(LLMProvider):

    def __init__(self):
        self.messages_received = []

    def stream_chat(
        self,
        messages: list[Message]
    ) -> Iterator[str]:
        self.messages_received = messages

        yield "Test response"

    def set_model(
        self,
        model: str
    ) -> None:
        pass

class AssistantTest(unittest.TestCase):

    def test_sends_augmented_message_but_stores_original(self):

        provider = FakeLLMProvider()

        assistant = Assistant(
            llm=provider,
            system_prompt="Test system prompt"
        )

        response = "".join(
            assistant.send_message(
                user_message="Original question",
                model_message="Question with RAG context"
            )
        )

        self.assertEqual(
            response,
            "Test response"
        )

        self.assertEqual(
            provider.messages_received[-1]["content"],
            "Question with RAG context"
        )

        self.assertEqual(
            assistant.messages[-2]["content"],
            "Original question"
        )

        stored_contents = [
            message["content"]
            for message in assistant.messages
        ]

        self.assertNotIn(
            "Question with RAG context",
            stored_contents
        )

if __name__ == "__main__":
    unittest.main()