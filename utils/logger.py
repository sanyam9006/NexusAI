"""
Structured logging configuration for Nexus AI.

Usage:
    from utils.logger import setup_logger
    logger = setup_logger("nexus.mymodule")
    logger.info("Something happened: %s", detail)
"""

import logging
import sys


def setup_logger(
    name: str = "nexus-ai",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a configured logger with consistent formatting.

    Each module should call this once at module level:
        logger = setup_logger("nexus.documents")

    Returns:
        A logging.Logger with a StreamHandler writing to stdout.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
