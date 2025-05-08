"""
BigOcrPdf - Services Package

This package contains business logic components like settings management
and OCR processing functionality.

Modules:
    - settings: Handles application configuration and file management
    - processor: Main OCR processing module integrating with OCRmyPDF
    - ocr_api: Low-level integration with OCRmyPDF Python API
"""

from .ocr_api import OcrQueue, OcrProcess, configure_logging
from .settings import OcrSettings

__all__ = ['OcrQueue', 'OcrProcess', 'OcrSettings', 'configure_logging']