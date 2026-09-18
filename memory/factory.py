from config import MemoryConfig
from logger import get_logger
from memory.base import ConversationMemory
from memory.null import NullMemory
from memory.short_term import SlidingWindowMemory

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