"""Recording an OCR'd page into the canonical document.

``OcrDocument`` is the single source of truth for everything downstream: the
structured OCR JSON a caller may ask for, the structured TXT/Markdown/ODT
exports, the average confidence the interface reports, and every per-region
field in a benchmark record.

Only the image-only pipeline ever filled it. Mixed-content and embedded-image
documents counted their regions and then dropped them, so those documents
reported a confidence of exactly zero, exported through the ``pdftotext``
fallback instead of their own OCR geometry, and were invisible to the
zero-region regression gate. This module is what the three pipelines share so
that cannot drift again.
"""

from bigocrpdf.services.rapidocr_service.config import (
    OcrPage,
    OCRResult,
    ProcessingStats,
)
from bigocrpdf.services.rapidocr_service.ocr_document_structure import (
    build_ocr_lines_from_results,
)


def record_ocr_page(
    stats: ProcessingStats,
    page_num: int,
    ocr_results: list[OCRResult],
    image_size_px: tuple[int, int],
    dpi: int,
    *,
    text_layer_quality: str = "ocr",
    diagnostics: dict | None = None,
) -> None:
    """Append one OCR'd page to the document, and count its regions.

    Counting happens here too so a caller cannot record the page and forget the
    tally, or the reverse -- which is exactly how the two ran apart.

    Args:
        image_size_px: Size of the image the OCR boxes were measured on, which
            is what the coordinates in ``ocr_results`` are relative to.
    """
    stats.total_text_regions += len(ocr_results)
    stats.ocr_document.append_page(
        OcrPage(
            page_index=page_num,
            width_px=int(image_size_px[0]),
            height_px=int(image_size_px[1]),
            dpi=int(dpi or 300),
            text_results=list(ocr_results),
            lines=build_ocr_lines_from_results(list(ocr_results)),
            text_layer_quality=text_layer_quality if ocr_results else "absent",
            diagnostics=diagnostics or {},
        )
    )


def average_confidence(stats: ProcessingStats) -> float:
    """Mean confidence over every region in the document.

    The pipelines used to accumulate this themselves and two of them did not,
    so it is derived from the recorded pages instead of tracked separately.
    """
    confidences = [
        result.confidence for page in stats.ocr_document.pages for result in page.text_results
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0
