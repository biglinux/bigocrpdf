"""
BigOcrPdf - Services Package

This package contains business logic components like settings management
and OCR processing functionality.

Modules:
    - settings: Handles application configuration and file management
    - processor: Main OCR processing module integrating with OCRmyPDF
    - ocr_api: Low-level integration with OCRmyPDF Python API
    - ocr_options_builder: Builder pattern for OCR options
"""

from .ocr_api import OcrProcess, OcrQueue, configure_logging
from .ocr_options_builder import OcrOptionsBuilder
from .settings import OcrSettings

__all__ = [
    "OcrQueue",
    "OcrProcess",
    "OcrSettings",
    "OcrOptionsBuilder",
    "configure_logging",
]