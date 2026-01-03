import sys

from loguru import logger

from app.core.config import settings

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _normalize_log_level(level_name: str) -> str:
    cleaned = (level_name or "").strip().upper()
    if not cleaned:
        return "INFO"
    try:
        logger.level(cleaned)
        return cleaned
    except ValueError:
        sys.stderr.write(
            f"Invalid LOG_LEVEL '{level_name}', falling back to INFO.\n"
        )
        return "INFO"


logger.remove()
logger.add(sys.stdout, level=_normalize_log_level(settings.LOG_LEVEL), format=LOG_FORMAT)
