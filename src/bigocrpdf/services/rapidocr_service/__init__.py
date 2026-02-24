"""
RapidOCR Integration Module for BigOcrPdf.

This module provides high-quality OCR processing using RapidOCR with PP-OCRv5 models.
It replaces the previous OCRmyPDF/Tesseract implementation.

Main components:
- OCRConfig: Configuration dataclass for OCR processing
- ModelDiscovery: Automatic detection of available models and fonts
- RapidOCREngine: Main engine for PDF processing
- ImagePreprocessor: Image preprocessing (deskew, scanner effect, etc.)
- PDFImageExtractor: Native PDF image extraction
- TextLayerRenderer: Invisible text layer generation
"""

from bigocrpdf.services.rapidocr_service.config import (
    OCRBoxData,
    OCRConfig,
    OCRResult,
    ProcessingStats,
)
from bigocrpdf.services.rapidocr_service.discovery import ModelDiscovery
from bigocrpdf.services.rapidocr_service.engine import (
    ProgressCallback,
    RapidOCREngine,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import PDFImageExtractor
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.services.rapidocr_service.renderer import (
    PageTextLayer,
    TextBox,
    TextLayerRenderer,
)

__all__ = [
    # Config
    "OCRBoxData",
    "OCRConfig",
    "OCRResult",
    "ProcessingStats",
    # Discovery
    "ModelDiscovery",
    # Engine
    "RapidOCREngine",
    "ProgressCallback",
    # Extractor
    "PDFImageExtractor",
    # Preprocessor
    "ImagePreprocessor",
    # Renderer
    "TextLayerRenderer",
    "TextBox",
    "PageTextLayer",
]
