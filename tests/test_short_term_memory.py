import unittest

from memory.short_term import SlidingWindowMemory


class SlidingWindowMemoryTest(unittest.TestCase):

    def test_retains_only_the_most_recent_complete_turns(self):
        """Evict complete old turns when the configured window is full."""

        memory = SlidingWindowMemory(
            max_turns=2
        )

        memory.add_turn(
            user_message="First question",
            assistant_message="First answer"
        )
        memory.add_turn(
            user_message="Second question",
            assistant_message="Second answer"
        )
        memory.add_turn(
            user_message="Third question",
            assistant_message="Third answer"
        )

        self.assertEqual(
            memory.get_messages(),
            [
                {
                    "role": "user",
                    "content": "Second question"
                },
                {
                    "role": "assistant",
                    "content": "Second answer"
                },
                {
                    "role": "user",
                    "content": "Third question"
                },
                {
                    "role": "assistant",
                    "content": "Third answer"
                }
            ]
        )

    def test_clear_removes_all_conversation_turns(self):
        """Reset short-term memory without affecting its configuration."""

        memory = SlidingWindowMemory(
            max_turns=2
        )
        memory.add_turn(
            user_message="Remember this",
            assistant_message="Remembered"
        )

        memory.clear()

        self.assertEqual(
            memory.get_messages(),
            []
        )
        self.assertEqual(
            memory.max_turns,
            2
        )

    def test_get_messages_returns_an_independent_copy(self):
        """Callers must not be able to mutate stored conversation history."""

        memory = SlidingWindowMemory(
            max_turns=1
        )
        memory.add_turn(
            user_message="Original question",
            assistant_message="Original answer"
        )

        returned_messages = memory.get_messages()
        returned_messages[0]["content"] = "Changed externally"
        returned_messages.append(
            {
                "role": "user",
                "content": "Injected message"
            }
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
                    "content": "Original answer"
                }
            ]
        )

    def test_rejects_non_positive_turn_limit(self):
        """A sliding window must retain at least one complete turn."""

        for invalid_limit in (0, -1):
            with self.subTest(
                max_turns=invalid_limit
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_turns must be positive"
                ):
                    SlidingWindowMemory(
                        max_turns=invalid_limit
                    )


if __name__ == "__main__":
    unittest.main()
