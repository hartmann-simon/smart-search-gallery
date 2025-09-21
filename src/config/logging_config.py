"""
Simple logging configuration for Smart Search Gallery
"""
import logging


def setup_logging(level: str = "ERROR") -> None:
    """
    Set up simple console logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.ERROR)

    # Simple console-only configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )

    # Suppress noisy third-party logs
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
