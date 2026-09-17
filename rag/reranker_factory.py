from config import RAGConfig
from logger import get_logger
from rag.flashrank_reranker import FlashRankReranker
from rag.reranker_base import Reranker

logger = get_logger(__name__)

def create_reranker(
    config: RAGConfig
) -> Reranker | None: 
    """Create the configured reranker when enabled."""

    if not config.reranker_enabled:
        logger.info("Reranker is disabled")
        return None

    logger.info(
        "Using reranker backend: %s",
        config.reranker_backend
    )

    if config.reranker_backend == "flashrank":
        return FlashRankReranker(
            model_name=config.reranker_model,
            cache_dir=config.reranker_cache_dir,
            max_length=config.reranker_max_length,
            min_score=config.reranker_min_score
        )

    raise ValueError(
        "Unsupported reranker backend: "
        f"{config.reranker_backend}"
    )