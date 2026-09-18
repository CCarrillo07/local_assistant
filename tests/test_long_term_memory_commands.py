import unittest
from unittest.mock import call, patch

import main
from config import AppConfig, MemoryConfig, RAGConfig
from memory.long_term_base import MemoryRecord


def create_test_config() -> AppConfig:
    return AppConfig(
        default_model="test-model",
        memory=MemoryConfig(
            short_term_enabled=False,
            long_term_enabled=True,
            long_term_backend="sqlite",
            long_term_database_path="storage/test_memory.db"
        ),
        rag=RAGConfig(
            available=False,
            mode="manual",
            allow_user_control=False,
            vector_store_backend="npz",
            reranker_enabled=False
        )
    )


class MainLongTermMemoryCommandTest(unittest.TestCase):

    @patch(
        "builtins.input",
        side_effect=["/exit"]
    )
    @patch("builtins.print")
    @patch(
        "main.create_long_term_memory",
        create=True
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_creates_store_and_displays_memory_commands(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_print,
        mock_input
    ):
        """Create persistent memory and advertise its commands."""

        test_config = create_test_config()

        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()

        mock_long_term_factory.assert_called_once_with(
            config=test_config.memory
        )
        mock_print.assert_any_call(
            "/remember <fact>"
        )
        mock_print.assert_any_call(
            "/memories"
        )
        mock_print.assert_any_call(
            "/forget <id>"
        )

    @patch(
        "builtins.input",
        side_effect=[
            "/remember My puppy is named Happy",
            "/memories",
            "/exit"
        ]
    )
    @patch("builtins.print")
    @patch(
        "main.create_long_term_memory",
        create=True
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_remembers_and_lists_user_approved_facts(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_print,
        mock_input
    ):
        """Memory commands must not be sent to the language model."""

        record = MemoryRecord(
            memory_id=7,
            content="My puppy is named Happy"
        )
        store = mock_long_term_factory.return_value
        store.remember.return_value = record
        store.list_memories.return_value = [
            record
        ]

        with patch.object(
            main,
            "CONFIG",
            create_test_config()
        ):
            main.main()

        store.remember.assert_called_once_with(
            "My puppy is named Happy"
        )
        store.list_memories.assert_called_once_with()
        mock_print.assert_any_call(
            "\nMemory saved [7]: "
            "My puppy is named Happy"
        )
        mock_print.assert_any_call(
            "\nLong-term memories:"
        )
        mock_print.assert_any_call(
            "[7] My puppy is named Happy"
        )
        mock_assistant_class.return_value.send_message.assert_not_called()

    @patch(
        "builtins.input",
        side_effect=[
            "/forget 7",
            "/forget 999",
            "/forget invalid",
            "/exit"
        ]
    )
    @patch("builtins.print")
    @patch(
        "main.create_long_term_memory",
        create=True
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_forgets_by_numeric_id_and_reports_result(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_print,
        mock_input
    ):
        """Delete only valid IDs and distinguish missing memories."""

        store = mock_long_term_factory.return_value
        store.forget.side_effect = [
            True,
            False
        ]

        with patch.object(
            main,
            "CONFIG",
            create_test_config()
        ):
            main.main()

        self.assertEqual(
            store.forget.call_args_list,
            [
                call(7),
                call(999)
            ]
        )
        mock_print.assert_any_call(
            "\nMemory 7 forgotten"
        )
        mock_print.assert_any_call(
            "\nMemory 999 was not found"
        )
        mock_print.assert_any_call(
            "\nUsage: /forget <id>"
        )
        mock_assistant_class.return_value.send_message.assert_not_called()

    @patch(
        "builtins.input",
        side_effect=[
            "/forget 7",
            "/forget 999",
            "/exit"
        ]
    )
    @patch("builtins.print")
    @patch(
        "main.create_long_term_memory",
        create=True
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_successful_forget_clears_short_term_memory(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_print,
        mock_input
    ):
        """Clear conversation context only after a successful deletion."""

        store = mock_long_term_factory.return_value
        store.forget.side_effect = [
            True,
            False
        ]
        assistant = mock_assistant_class.return_value

        with patch.object(
            main,
            "CONFIG",
            create_test_config()
        ):
            main.main()

        self.assertEqual(
            store.forget.call_args_list,
            [
                call(7),
                call(999)
            ]
        )
        assistant.reset.assert_called_once_with()

    @patch(
        "builtins.input",
        side_effect=[
            "/reset",
            "/exit"
        ]
    )
    @patch(
        "main.create_long_term_memory",
        create=True
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_reset_does_not_modify_long_term_memory(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_long_term_factory,
        mock_input
    ):
        """Reset clears conversation history, not persistent facts."""

        store = mock_long_term_factory.return_value
        assistant = mock_assistant_class.return_value

        with patch.object(
            main,
            "CONFIG",
            create_test_config()
        ):
            main.main()

        assistant.reset.assert_called_once_with()
        store.remember.assert_not_called()
        store.list_memories.assert_not_called()
        store.forget.assert_not_called()


if __name__ == "__main__":
    unittest.main()
