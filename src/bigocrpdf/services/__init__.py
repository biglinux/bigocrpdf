"""
BigOcrPdf - Services Package

Service modules for OCR processing and settings management.
"""

from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.services.settings import OcrSettings

__all__ = ["OcrProcessor", "OcrSettings"]
