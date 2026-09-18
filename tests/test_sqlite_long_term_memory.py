import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from memory.long_term_base import MemoryRecord
from memory.sqlite_long_term import SQLiteLongTermMemoryStore


class SQLiteLongTermMemoryStoreTest(unittest.TestCase):

    def test_persists_memories_across_store_instances(self):
        """Saved facts must survive application restarts."""

        with TemporaryDirectory() as temporary_directory:
            database_path = (
                Path(temporary_directory)
                / "nested"
                / "long_term_memory.db"
            )

            first_store = SQLiteLongTermMemoryStore(
                database_path=database_path
            )
            saved_memory = first_store.remember(
                "My puppy is named Happy"
            )

            second_store = SQLiteLongTermMemoryStore(
                database_path=database_path
            )

            self.assertIsInstance(
                saved_memory,
                MemoryRecord
            )
            self.assertGreater(
                saved_memory.memory_id,
                0
            )
            self.assertEqual(
                saved_memory.content,
                "My puppy is named Happy"
            )
            self.assertEqual(
                second_store.list_memories(),
                [saved_memory]
            )

    def test_normalizes_and_deduplicates_identical_memories(self):
        """Repeated explicit saves must not create duplicate facts."""

        with TemporaryDirectory() as temporary_directory:
            store = SQLiteLongTermMemoryStore(
                database_path=(
                    Path(temporary_directory)
                    / "memory.db"
                )
            )

            first_memory = store.remember(
                "My favorite food is enchiladas"
            )
            repeated_memory = store.remember(
                "  My favorite food is enchiladas  "
            )

            self.assertEqual(
                repeated_memory,
                first_memory
            )
            self.assertEqual(
                store.list_memories(),
                [first_memory]
            )

    def test_forgets_only_the_requested_memory(self):
        """Delete one fact by ID while preserving the other facts."""

        with TemporaryDirectory() as temporary_directory:
            store = SQLiteLongTermMemoryStore(
                database_path=(
                    Path(temporary_directory)
                    / "memory.db"
                )
            )
            first_memory = store.remember(
                "My puppy is named Happy"
            )
            second_memory = store.remember(
                "My favorite food is enchiladas"
            )

            self.assertTrue(
                store.forget(
                    first_memory.memory_id
                )
            )
            self.assertFalse(
                store.forget(
                    first_memory.memory_id
                )
            )
            self.assertEqual(
                store.list_memories(),
                [second_memory]
            )

    def test_rejects_blank_memory_content(self):
        """Do not persist empty or whitespace-only memories."""

        with TemporaryDirectory() as temporary_directory:
            store = SQLiteLongTermMemoryStore(
                database_path=(
                    Path(temporary_directory)
                    / "memory.db"
                )
            )

            for invalid_content in ("", "   "):
                with self.subTest(
                    content=invalid_content
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Memory content cannot be blank"
                    ):
                        store.remember(
                            invalid_content
                        )


if __name__ == "__main__":
    unittest.main()
