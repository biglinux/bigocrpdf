"""
BigOcrPdf - Logger Module

This module sets up logging for the application.
"""

import logging
from ..config import LOG_LEVEL, LOG_FORMAT, LOGGER_NAME


def setup_logger() -> logging.Logger:
    """Set up and configure the application logger

    Returns:
        A configured Logger instance
    """
    # Configure basic logging settings
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    # Create and return the logger
    logger = logging.getLogger(LOGGER_NAME)
    return logger


# Create a singleton logger instance
logger = setup_logger()
