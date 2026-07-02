"""
Logging utility using Loguru for structured, multi-destination log management.

Configured to log every stage of the voice chatbot automation pipeline:
- Audio received
- Speech recognition completed
- Prompt sent
- LLM response
- Audio generated
- Execution time and latencies
- Errors and exceptions
"""

import sys
from pathlib import Path
from loguru import logger
from app.config.settings import settings


def setup_logger() -> None:
    """
    Initialize and configure Loguru logger handlers.
    
    Removes default handlers and attaches:
    1. Standard Out (console) handler with colorized formatting.
    2. Persistent Rotating File handler in `app/logs/` with JSON or structured text formatting.
    """
    # Remove any existing default handlers
    logger.remove()

    # Console Handler for real-time monitoring and development
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        enqueue=True,
    )

    # File Handler for production audit trail and error analysis
    log_file: Path = settings.LOG_FILE_PATH
    logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8",
    )

    logger.info("Loguru logging framework initialized successfully.")


# Initialize logging immediately upon module import or manual call
setup_logger()

# Export configured logger instance
__all__ = ["logger", "setup_logger"]
