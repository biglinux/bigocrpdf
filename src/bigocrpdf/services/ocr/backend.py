"""The PDF OCR job: choosing a pipeline for the document and running it."""

import threading
from collections.abc import Callable
from pathlib import Path

from bigocrpdf.services.ocr.backend_pipeline import BackendPipelineMixin
from bigocrpdf.services.ocr.backend_text_layer import BackendTextLayerMixin
from bigocrpdf.services.ocr.config import OCRConfig, ProcessingStats
from bigocrpdf.services.ocr.pdf_extractor import (
    PDFImageExtractor,
    has_native_text,
    has_trusted_native_text,
)
from bigocrpdf.services.ocr.pdf_image_analysis import inspect_pdf_resource_metrics
from bigocrpdf.services.ocr.ppocr_native import Detector, Worker, start_ocr_worker
from bigocrpdf.services.ocr.preprocessor import ImagePreprocessor
from bigocrpdf.services.ocr.renderer import TextLayerRenderer
from bigocrpdf.services.ocr.resource_manager import enforce_pdf_resource_limits
from bigocrpdf.services.ocr.text_formatting_controller import TextFormattingController
from bigocrpdf.utils.i18n import ngettext
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
    """One OCR job over a PDF, cancellable from another thread."""

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()
        self.cancel_event = threading.Event()
        self._text_formatting = TextFormattingController(self.config)
        self.extractor = PDFImageExtractor(self.config.dpi, self.config.max_render_megapixels)
        self.renderer = TextLayerRenderer(self.config)

    def cancel(self) -> None:
        """Stop the job: the pipeline checks the event between steps, and a
        worker request in flight fails at once."""
        self.cancel_event.set()

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            logger.info("Processing cancelled by user")
            raise InterruptedError("Processing cancelled by user")

    def _start_ocr_worker(self, threads: int | None = None) -> Worker:
        """The job's OCR worker; images preprocessed in this process, which
        may be the GUI, ask it for detection too, so the detector model is
        loaded once and goes away with the worker."""
        worker = start_ocr_worker(self.config, self.cancel_event, threads)
        self.preprocessor = ImagePreprocessor(self.config, Detector(self.cancel_event, worker))
        return worker

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

        Raises:
            RuntimeError: naming the pages OCR could not read and why.
        """
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)

        if not input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        logger.info(f"Processing: {input_pdf}")
        logger.info(f"Output: {output_pdf}")

        resource_metrics = inspect_pdf_resource_metrics(
            input_pdf,
            max_pages=self.config.max_pdf_pages,
        )
        enforce_pdf_resource_limits(
            resource_metrics.total_pages,
            self.config,
            resource_metrics.image_dimensions,
        )

        # Choose pipeline: mixed content (text + images) vs image-only.
        # force_full_ocr is set by the editor for merged files.
        # replace_existing_ocr is handled inside each pipeline: the mixed
        # content pipeline strips old OCR layers and re-OCRs, while the
        # image-only pipeline simply overwrites.  Only force_full_ocr
        # should bypass the mixed content detection entirely.
        if should_use_mixed_content_pipeline(self.config, input_pdf):
            logger.info("Detected mixed content PDF (text + images). Using preservation mode.")
            stats = self._process_mixed_content_pdf(input_pdf, output_pdf, progress_callback)
        else:
            if self.config.force_full_ocr or self.config.pdf_mode == "ocr":
                logger.info("Force full OCR mode (editor-merged file). Using full OCR mode.")
            else:
                logger.info("Detected image-only PDF. Using full OCR mode.")
            stats = self._process_image_only_pdf(input_pdf, output_pdf, progress_callback)
        # A page left without its text is a failed document, not a blank page.
        if stats.failed_pages:
            pages_by_cause: dict[str, list[int]] = {}
            for page in sorted(stats.failed_pages):
                pages_by_cause.setdefault(stats.failed_pages[page], []).append(page)
            raise RuntimeError(
                "\n".join(
                    ngettext(
                        "Page {pages} could not be read: {cause}",
                        "Pages {pages} could not be read: {cause}",
                        len(pages),
                    ).format(pages=", ".join(map(str, pages)), cause=cause)
                    for cause, pages in pages_by_cause.items()
                )
            )
        return stats
