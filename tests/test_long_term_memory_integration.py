import unittest
from unittest.mock import patch

import main
from config import AppConfig, MemoryConfig, RAGConfig
from memory.long_term_base import MemoryRecord


class MainLongTermMemoryContextTest(unittest.TestCase):

    @patch(
        "builtins.input",
        side_effect=[
            "What is my favorite food?",
            "/exit"
        ]
    )
    @patch("builtins.print")
    @patch(
        "main.build_long_term_memory_message",
        create=True
    )
    @patch(
        "main.create_long_term_memory"
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_injects_bounded_long_term_memory_into_normal_requests(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_build_memory_message,
        mock_print,
        mock_input
    ):
        """Load persistent facts and pass bounded context to Assistant."""

        test_config = AppConfig(
            default_model="test-model",
            memory=MemoryConfig(
                short_term_enabled=False,
                long_term_enabled=True,
                long_term_context_limit=2
            ),
            rag=RAGConfig(
                available=False,
                mode="manual",
                allow_user_control=False,
                vector_store_backend="npz",
                reranker_enabled=False
            )
        )
        memories = [
            MemoryRecord(
                memory_id=1,
                content="My puppy is named Happy"
            ),
            MemoryRecord(
                memory_id=2,
                content="My favorite food is enchiladas"
            )
        ]
        memory_message = {
            "role": "system",
            "content": "Bounded long-term memory"
        }

        store = mock_long_term_factory.return_value
        store.list_memories.return_value = memories
        mock_build_memory_message.return_value = (
            memory_message
        )

        assistant = mock_assistant_class.return_value
        assistant.send_message.return_value = iter(
            ["Enchiladas"]
        )

        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()

        store.list_memories.assert_called_once_with()
        mock_build_memory_message.assert_called_once_with(
            memories=memories,
            max_memories=2
        )
        assistant.send_message.assert_called_once_with(
            user_message="What is my favorite food?",
            model_message=None,
            context_messages=[
                memory_message
            ]
        )


if __name__ == "__main__":
    unittest.main()
