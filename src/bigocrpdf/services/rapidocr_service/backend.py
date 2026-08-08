#!/usr/bin/env python3
"""RapidOCR PDF backend and pipeline composition."""

import threading
from collections.abc import Callable
from pathlib import Path

from bigocrpdf.services.rapidocr_service.backend_pipeline import BackendPipelineMixin
from bigocrpdf.services.rapidocr_service.backend_text_layer import BackendTextLayerMixin

# Import rapidocr with fallback to other Python versions
# Import unified OCRConfig and data classes from the single source of truth
from bigocrpdf.services.rapidocr_service.config import (
    OCRConfig,
    ProcessingStats,
)
from bigocrpdf.services.rapidocr_service.ocr_controller import OCRController
from bigocrpdf.services.rapidocr_service.ocr_subprocess_controller import (
    OCRSubprocessController,
)

# Import extracted logic
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    PDFImageExtractor,
    has_native_text,
    has_trusted_native_text,
)
from bigocrpdf.services.rapidocr_service.pdf_image_analysis import inspect_pdf_resource_metrics

# Import ImagePreprocessor from dedicated module (single source of truth)
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.services.rapidocr_service.renderer import TextLayerRenderer
from bigocrpdf.services.rapidocr_service.resource_manager import enforce_pdf_resource_limits
from bigocrpdf.services.rapidocr_service.text_formatting_controller import TextFormattingController
from bigocrpdf.utils.logger import logger


def should_use_mixed_content_pipeline(config: OCRConfig, input_pdf: Path) -> bool:
    """Decide whether the PDF should preserve trusted native text."""
    if config.force_full_ocr or config.pdf_mode == "ocr":
        return False
    if config.pdf_mode == "auto_verified":
        logger.info("pdf_mode=auto_verified uses full OCR before native text verification.")
        return False
    if config.pdf_mode not in {"auto", "auto_verified", "geometric"}:
        logger.warning(f"Unsupported pdf_mode={config.pdf_mode!r}; falling back to auto")
    if not has_native_text(input_pdf):
        return False
    return has_trusted_native_text(input_pdf)


class ProfessionalPDFOCR(
    BackendPipelineMixin,
    BackendTextLayerMixin,
):
    """High-quality PDF OCR engine for professional document processing."""

    # Class-level cache for OpenVINO availability (check once per process)
    _openvino_available: bool | None = None

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()
        self.cancel_event = threading.Event()
        self._ocr = OCRController(self.config, self._check_openvino_available)
        self._ocr_subprocess = OCRSubprocessController(
            self.config,
            self.cancel_event,
            self._check_openvino_available,
        )
        self._text_formatting = TextFormattingController(self.config)
        self.preprocessor = ImagePreprocessor(self.config)
        self.extractor = PDFImageExtractor(self.config.dpi, self.config.max_render_megapixels)
        self.renderer = TextLayerRenderer(self.config)

    @classmethod
    def _check_openvino_available(cls) -> bool:
        """Check if OpenVINO is available and compatible with current Python version.

        Result is cached to avoid import system corruption on repeated failed imports.
        """
        if cls._openvino_available is not None:
            return cls._openvino_available

        try:
            from openvino._pyopenvino import AxisSet  # noqa: F401

            cls._openvino_available = True
        except (ImportError, ModuleNotFoundError, KeyError):
            cls._openvino_available = False

        return cls._openvino_available

    def process(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process PDF and create searchable version.

        This method detects the PDF type and uses the appropriate strategy:
        - Image-only PDFs: Extract images, OCR all, create searchable PDF
        - Mixed content PDFs: Preserve original structure, OCR only images in place

        Args:
            input_pdf: Path to input PDF file
            output_pdf: Path for output searchable PDF
            progress_callback: Optional callback(current, total, status_message)

        Returns:
            ProcessingStats with processing details
        """
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)
        self._input_pdf = input_pdf

        if not input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        logger.info(f"Processing: {input_pdf}")
        logger.info(f"Output: {output_pdf}")

        resource_metrics = inspect_pdf_resource_metrics(
            input_pdf,
            max_pages=int(getattr(self.config, "max_pdf_pages", 0)),
        )
        enforce_pdf_resource_limits(
            resource_metrics.total_pages,
            resource_metrics.page_dimensions,
            self.config,
            resource_metrics.image_dimensions,
        )

        # Choose pipeline: mixed content (text + images) vs image-only.
        # force_full_ocr is set by the editor for merged files.
        # replace_existing_ocr is handled inside each pipeline: the mixed
        # content pipeline strips old OCR layers and re-OCRs, while the
        # image-only pipeline simply overwrites.  Only force_full_ocr
        # should bypass the mixed content detection entirely.
        use_mixed_pipeline = should_use_mixed_content_pipeline(self.config, input_pdf)
        if use_mixed_pipeline:
            logger.info("Detected mixed content PDF (text + images). Using preservation mode.")
            return self._process_mixed_content_pdf(input_pdf, output_pdf, progress_callback)
        else:
            if self.config.force_full_ocr or self.config.pdf_mode == "ocr":
                logger.info("Force full OCR mode (editor-merged file). Using full OCR mode.")
            else:
                logger.info("Detected image-only PDF. Using full OCR mode.")
            return self._process_image_only_pdf(input_pdf, output_pdf, progress_callback)
