import unittest
from unittest.mock import patch

from config import MemoryConfig
from memory import factory


class LongTermMemoryConfigTest(unittest.TestCase):

    def test_accepts_independent_long_term_memory_configuration(self):
        """Long-term memory must be configurable independently."""

        config = MemoryConfig(
            short_term_enabled=False,
            short_term_max_turns=3,
            long_term_enabled=True,
            long_term_backend="sqlite",
            long_term_database_path="storage/test_memory.db"
        )

        self.assertFalse(
            config.short_term_enabled
        )
        self.assertTrue(
            config.long_term_enabled
        )
        self.assertEqual(
            config.long_term_backend,
            "sqlite"
        )
        self.assertEqual(
            config.long_term_database_path,
            "storage/test_memory.db"
        )

    def test_rejects_unsupported_long_term_memory_backend(self):
        """Reject backends that the application cannot construct."""

        with self.assertRaisesRegex(
            ValueError,
            "Invalid long-term memory backend"
        ):
            MemoryConfig(
                long_term_backend="unsupported"
            )


class LongTermMemoryFactoryTest(unittest.TestCase):

    @patch(
        "memory.factory.SQLiteLongTermMemoryStore",
        create=True
    )
    def test_creates_configured_sqlite_store(
        self,
        mock_sqlite_store_class
    ):
        """Create SQLite storage at the configured database path."""

        config = MemoryConfig(
            long_term_enabled=True,
            long_term_backend="sqlite",
            long_term_database_path="storage/test_memory.db"
        )

        store = factory.create_long_term_memory(
            config=config
        )

        mock_sqlite_store_class.assert_called_once_with(
            database_path="storage/test_memory.db"
        )
        self.assertIs(
            store,
            mock_sqlite_store_class.return_value
        )

    @patch(
        "memory.factory.SQLiteLongTermMemoryStore",
        create=True
    )
    def test_returns_none_when_long_term_memory_is_disabled(
        self,
        mock_sqlite_store_class
    ):
        """Do not create persistent storage when the module is disabled."""

        config = MemoryConfig(
            long_term_enabled=False,
            long_term_backend="sqlite",
            long_term_database_path="storage/test_memory.db"
        )

        store = factory.create_long_term_memory(
            config=config
        )

        self.assertIsNone(
            store
        )
        mock_sqlite_store_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
