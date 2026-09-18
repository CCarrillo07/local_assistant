from config import MemoryConfig
from logger import get_logger
from memory.base import ConversationMemory
from memory.null import NullMemory
from memory.short_term import SlidingWindowMemory
from memory.long_term_base import (
    LongTermMemoryStore
)
from memory.sqlite_long_term import (
    SQLiteLongTermMemoryStore
)

logger = get_logger(__name__)

def create_short_term_memory(
    config: MemoryConfig
) -> ConversationMemory:
    if not config.short_term_enabled:
        logger.info("Short-term memory is disabled")
        return NullMemory()

    logger.info(
        "Using short-term memory with %s turns",
        config.short_term_max_turns
    )

    return SlidingWindowMemory(
        max_turns=config.short_term_max_turns
    )

def create_long_term_memory(
    config: MemoryConfig
) -> LongTermMemoryStore | None:

    if not config.long_term_enabled:
        logger.info(
            "Long-term memory is disabled"
        )

        return None

    if config.long_term_backend == "sqlite":
        logger.info(
            "Using SQLite long-term memory"
        )

        return SQLiteLongTermMemoryStore(
            database_path=(
                config.long_term_database_path
            )
        )

    raise ValueError(
        "Unsupported long-term memory backend: "
        f"{config.long_term_backend}"
    )