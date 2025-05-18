"""
BigOcrPdf - Logger Module

This module sets up logging for the application.
"""

import logging

# Default values if config is not available
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGGER_NAME = "BigOcrPdf"

def setup_logger(log_level=None, log_format=None, logger_name=None):
    """Set up and configure the application logger

    Args:
        log_level: Logging level to use (default: INFO)
        log_format: Logging format string (default: standard format)
        logger_name: Name for the logger (default: BigOcrPdf)

    Returns:
        A configured Logger instance
    """
    # Use default values if not provided
    if log_level is None:
        # Try to import from config if available
        try:
            from config import LOG_LEVEL
            log_level = LOG_LEVEL
        except ImportError:
            log_level = DEFAULT_LOG_LEVEL
            
    if log_format is None:
        try:
            from config import LOG_FORMAT
            log_format = LOG_FORMAT
        except ImportError:
            log_format = DEFAULT_LOG_FORMAT
            
    if logger_name is None:
        try:
            from config import LOGGER_NAME
            logger_name = LOGGER_NAME
        except ImportError:
            logger_name = DEFAULT_LOGGER_NAME
    
    # Configure basic logging settings
    logging.basicConfig(level=log_level, format=log_format)

    # Create and return the logger
    logger = logging.getLogger(logger_name)
    return logger

# Create a singleton logger instance
logger = setup_logger()