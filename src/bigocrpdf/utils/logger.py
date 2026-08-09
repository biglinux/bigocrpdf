"""
BigOcrPdf - Logger Module

This module sets up logging for the application.
"""

import logging

# Default values if config is not available
DEFAULT_LOG_LEVEL: int = logging.WARNING
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGGER_NAME: str = "BigOcrPdf"


def setup_logger(
    log_level: int | None = None,
    log_format: str | None = None,
    logger_name: str | None = None,
) -> logging.Logger:
    """Set up and configure the application logger.

    Args:
        log_level: Logging level to use (default: WARNING).
        log_format: Logging format string (default: standard format).
        logger_name: Name for the logger (default: BigOcrPdf).

    Returns:
        A configured Logger instance.
    """
    # Use default values if not provided
    if log_level is None:
        # Try to import from config if available
        try:
            from bigocrpdf.config import LOG_LEVEL

            log_level = LOG_LEVEL
        except ImportError:
            log_level = DEFAULT_LOG_LEVEL

    if log_format is None:
        try:
            from bigocrpdf.config import LOG_FORMAT

            log_format = LOG_FORMAT
        except ImportError:
            log_format = DEFAULT_LOG_FORMAT

    if logger_name is None:
        try:
            from bigocrpdf.config import LOGGER_NAME

            logger_name = LOGGER_NAME
        except ImportError:
            logger_name = DEFAULT_LOGGER_NAME

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(level=log_level, format=log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    return logger


# Importing a module must not configure process-wide logging.
logger = logging.getLogger(DEFAULT_LOGGER_NAME)
