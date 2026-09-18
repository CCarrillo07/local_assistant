import unittest
from unittest.mock import patch

import main
from config import AppConfig, MemoryConfig, RAGConfig
from memory.factory import create_short_term_memory
from memory.null import NullMemory
from memory.short_term import SlidingWindowMemory


class MemoryConfigurationTest(unittest.TestCase):

    def test_creates_configured_sliding_window_memory(self):
        """Build bounded short-term memory from application configuration."""

        config = MemoryConfig(
            short_term_enabled=True,
            short_term_max_turns=4
        )

        memory = create_short_term_memory(
            config=config
        )

        self.assertIsInstance(
            memory,
            SlidingWindowMemory
        )
        self.assertEqual(
            memory.max_turns,
            4
        )

    def test_creates_null_memory_when_short_term_memory_is_disabled(self):
        """Disable memory without adding conditionals to Assistant."""

        config = MemoryConfig(
            short_term_enabled=False,
            short_term_max_turns=4
        )

        memory = create_short_term_memory(
            config=config
        )
        memory.add_turn(
            user_message="Do not remember this",
            assistant_message="This should be discarded"
        )

        self.assertIsInstance(
            memory,
            NullMemory
        )
        self.assertEqual(
            memory.get_messages(),
            []
        )

    def test_rejects_non_positive_short_term_turn_limit(self):
        """Reject invalid window sizes at the configuration boundary."""

        for invalid_limit in (0, -1):
            with self.subTest(
                short_term_max_turns=invalid_limit
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "short_term_max_turns must be positive"
                ):
                    MemoryConfig(
                        short_term_max_turns=invalid_limit
                    )


class MainMemoryWiringTest(unittest.TestCase):

    @patch(
        "builtins.input",
        side_effect=["/exit"]
    )
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    @patch("main.create_short_term_memory")
    def test_wires_configured_memory_into_assistant(
        self,
        mock_memory_factory,
        mock_provider_class,
        mock_assistant_class,
        mock_input
    ):
        """Create memory once and inject it into the Assistant."""

        test_config = AppConfig(
            default_model="test-model",
            memory=MemoryConfig(
                short_term_enabled=True,
                short_term_max_turns=3
            ),
            rag=RAGConfig(
                available=False,
                mode="manual",
                allow_user_control=False
            )
        )
        mock_provider = (
            mock_provider_class.return_value
        )
        mock_memory = (
            mock_memory_factory.return_value
        )

        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()

        mock_memory_factory.assert_called_once_with(
            config=test_config.memory
        )
        mock_assistant_class.assert_called_once_with(
            llm=mock_provider,
            system_prompt=main.SYSTEM_PROMPT,
            memory=mock_memory
        )


if __name__ == "__main__":
    unittest.main()
