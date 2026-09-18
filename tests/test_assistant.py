import unittest
from collections.abc import Iterator

from assistant import Assistant
from llm.base import LLMProvider, Message
from memory.short_term import SlidingWindowMemory


class FakeLLMProvider(LLMProvider):

    def __init__(self):
        self.messages_received: list[Message] = []

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


class FailingLLMProvider(FakeLLMProvider):

    def stream_chat(
        self,
        messages: list[Message]
    ) -> Iterator[str]:
        self.messages_received = messages

        yield "Partial response"
        raise RuntimeError("LLM request failed")


class AssistantTest(unittest.TestCase):

    def test_sends_augmented_message_but_stores_original(self):
        """Keep temporary RAG context out of conversation memory."""

        provider = FakeLLMProvider()
        memory = SlidingWindowMemory(
            max_turns=2
        )
        assistant = Assistant(
            llm=provider,
            system_prompt="Test system prompt",
            memory=memory
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
            provider.messages_received,
            [
                {
                    "role": "system",
                    "content": "Test system prompt"
                },
                {
                    "role": "user",
                    "content": "Question with RAG context"
                }
            ]
        )
        self.assertEqual(
            memory.get_messages(),
            [
                {
                    "role": "user",
                    "content": "Original question"
                },
                {
                    "role": "assistant",
                    "content": "Test response"
                }
            ]
        )

    def test_sends_bounded_memory_before_current_model_message(self):
        """Build each request from system prompt, memory, and current input."""

        provider = FakeLLMProvider()
        memory = SlidingWindowMemory(
            max_turns=2
        )
        memory.add_turn(
            user_message="Earlier question",
            assistant_message="Earlier answer"
        )
        assistant = Assistant(
            llm=provider,
            system_prompt="Test system prompt",
            memory=memory
        )

        "".join(
            assistant.send_message(
                user_message="Current question",
                model_message="Current question with RAG context"
            )
        )

        self.assertEqual(
            provider.messages_received,
            [
                {
                    "role": "system",
                    "content": "Test system prompt"
                },
                {
                    "role": "user",
                    "content": "Earlier question"
                },
                {
                    "role": "assistant",
                    "content": "Earlier answer"
                },
                {
                    "role": "user",
                    "content": "Current question with RAG context"
                }
            ]
        )

    def test_does_not_store_partial_turn_when_llm_fails(self):
        """A failed streamed response must not leave partial memory."""

        provider = FailingLLMProvider()
        memory = SlidingWindowMemory(
            max_turns=2
        )
        assistant = Assistant(
            llm=provider,
            system_prompt="Test system prompt",
            memory=memory
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "LLM request failed"
        ):
            "".join(
                assistant.send_message(
                    user_message="Question that fails"
                )
            )

        self.assertEqual(
            memory.get_messages(),
            []
        )

    def test_reset_clears_conversation_memory(self):
        """Reset the assistant without changing memory configuration."""

        provider = FakeLLMProvider()
        memory = SlidingWindowMemory(
            max_turns=2
        )
        assistant = Assistant(
            llm=provider,
            system_prompt="Test system prompt",
            memory=memory
        )
        "".join(
            assistant.send_message(
                user_message="Remember this"
            )
        )

        assistant.reset()

        self.assertEqual(
            memory.get_messages(),
            []
        )
        self.assertEqual(
            memory.max_turns,
            2
        )


if __name__ == "__main__":
    unittest.main()
