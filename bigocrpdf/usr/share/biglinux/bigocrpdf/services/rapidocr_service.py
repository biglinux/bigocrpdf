"""
BigOcrPdf - RapidOCR Processing Helpers

This module wraps RapidOCR so the rest of the application can process
PDF and image inputs without caring about rendering, scaling or how the
text layer is generated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.logger import logger


try:
    import fitz  # type: ignore
except ImportError:
    fitz = None  # type: ignore

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ImportError:
    RapidOCR = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None  # type: ignore


@dataclass
class RapidOCRResult:
    """Represents a single OCR span returned by RapidOCR."""

    bbox: Sequence[Sequence[float]]
    text: str
    score: float


ProgressCallback = Callable[[int, int], None]


class RapidOCREngine:
    """Runs RapidOCR and builds searchable PDF pages from the results."""

    def __init__(self, zoom: float = 2.0):
        """
        Args:
            zoom: Rendering scale applied to PDF pages before OCR.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed; install python-pymupdf.")

        if RapidOCR is None:
            raise RuntimeError("rapidocr_onnxruntime is not installed; install python-rapidocr-onnxruntime.")

        if Image is None:
            raise RuntimeError("Pillow is not installed; install python-pillow.")

        self.zoom = max(1.0, zoom)
        self.ocr = RapidOCR()

    def process_document(
        self,
        input_file: str,
        output_file: str,
        input_mode: str,
        sidecar: Optional[str] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        """Process an input file and create a searchable PDF.

        Args:
            input_file: Source PDF or image path.
            output_file: Destination PDF path.
            input_mode: Either ``pdf`` or ``image``.
            sidecar: Optional text file to write recognized content to.
            progress_cb: Optional callback invoked with (processed_pages, total_pages).

        Returns:
            All recognized text concatenated by page.
        """
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        if input_mode == "image":
            text_content = self._process_image(input_file, output_file)
            if progress_cb:
                progress_cb(1, 1)
        else:
            text_content = self._process_pdf(input_file, output_file, progress_cb)

        if sidecar:
            try:
                with open(sidecar, "w", encoding="utf-8") as handle:
                    handle.write(text_content)
            except Exception as exc:
                logger.error("Failed to write sidecar file %s: %s", sidecar, exc)

        return text_content

    def _process_pdf(
        self,
        input_file: str,
        output_file: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        """Process a PDF file and return recognized text."""
        text_buffer: List[str] = []

        with fitz.open(input_file) as doc:
            total_pages = doc.page_count or 1

            for page_index, page in enumerate(doc):
                pixmap = self._render_page(page)
                ocr_results = self._run_ocr(pixmap)

                if not ocr_results:
                    continue

                page_text = self._inject_text_layer(
                    page, pixmap.width, pixmap.height, ocr_results
                )
                if page_text:
                    text_buffer.append(page_text)

                if progress_cb:
                    progress_cb(page_index + 1, total_pages)

            doc.save(
                output_file,
                garbage=3,
                deflate=True,
            )

        return "\n\n".join(text_buffer)

    def _process_image(self, input_file: str, output_file: str) -> str:
        """Convert an image into a searchable single-page PDF."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input image not found: {input_file}")

        with Image.open(input_file) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image_width, image_height = rgb_image.size
            np_image = np.array(rgb_image)

        ocr_results = self._run_ocr_array(np_image)
        with fitz.open() as doc:
            page = doc.new_page(width=float(image_width), height=float(image_height))
            rect = fitz.Rect(0, 0, float(image_width), float(image_height))
            page.insert_image(rect, filename=input_file)

            page_text = self._inject_text_layer(
                page, image_width, image_height, ocr_results
            )

            doc.save(
                output_file,
                garbage=3,
                deflate=True,
            )

        return page_text or ""

    def _render_page(self, page: "fitz.Page") -> "fitz.Pixmap":
        """Render a PDF page to a pixmap suited for OCR."""
        matrix = fitz.Matrix(self.zoom, self.zoom)
        return page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)

    def _run_ocr(self, pixmap: "fitz.Pixmap") -> List[RapidOCRResult]:
        """Run OCR on a pixmap produced from a PDF page."""
        np_image = self._pixmap_to_array(pixmap)
        return self._run_ocr_array(np_image)

    def _run_ocr_array(self, array: np.ndarray) -> List[RapidOCRResult]:
        """Execute RapidOCR on a numpy array and normalize the output."""
        results, _elapsed = self.ocr(array)
        normalized: List[RapidOCRResult] = []

        if not results:
            return normalized

        for bbox, text, score in results:
            try:
                normalized.append(
                    RapidOCRResult(
                        bbox=bbox,
                        text=text.strip(),
                        score=float(score),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to normalize OCR span: %s", exc)

        return [span for span in normalized if span.text]

    def _pixmap_to_array(self, pixmap: "fitz.Pixmap") -> np.ndarray:
        """Convert a pixmap to a RGB numpy array."""
        array = np.frombuffer(pixmap.samples, dtype=np.uint8)
        array = array.reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 4:
            array = array[..., :3]
        return array

    def _inject_text_layer(
        self,
        page: "fitz.Page",
        image_width: int,
        image_height: int,
        spans: Iterable[RapidOCRResult],
    ) -> str:
        """Overlay invisible text onto a PDF page."""
        collected: List[Tuple["fitz.Rect", str]] = []
        scale_x = page.rect.width / max(1, image_width)
        scale_y = page.rect.height / max(1, image_height)

        for span in spans:
            rect = self._bbox_to_rect(span.bbox, scale_x, scale_y)
            if rect is None or rect.width == 0 or rect.height == 0:
                continue

            fontsize = max(6.0, rect.height * 0.8)
            try:
                page.insert_textbox(
                    rect,
                    span.text,
                    fontname="helv",
                    fontsize=fontsize,
                    align=0,
                    render_mode=3,  # Invisible text
                    overlay=True,
                )
                collected.append((rect, span.text))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to insert text span: %s", exc)

        if not collected:
            return ""

        return self._group_text_lines(collected)

    @staticmethod
    def _group_text_lines(spans: List[Tuple["fitz.Rect", str]]) -> str:
        """Group recognized spans into text lines preserving reading order."""
        if not spans:
            return ""

        spans_sorted = sorted(spans, key=lambda item: (item[0].y0, item[0].x0))
        lines: List[str] = []
        current_line: List[str] = []
        current_mid = None
        current_height = None

        for rect, text in spans_sorted:
            if not text:
                continue

            midpoint = (rect.y0 + rect.y1) / 2.0
            height = rect.height or 1.0

            if current_line:
                tolerance = max(current_height or height, height) * 0.6
                if abs(midpoint - (current_mid or midpoint)) <= tolerance:
                    current_line.append(text)
                    current_mid = (current_mid + midpoint) / 2.0 if current_mid else midpoint
                    current_height = (current_height + height) / 2.0 if current_height else height
                    continue

                lines.append(" ".join(current_line))

            current_line = [text]
            current_mid = midpoint
            current_height = height

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    @staticmethod
    def _bbox_to_rect(
        bbox: Sequence[Sequence[float]],
        scale_x: float,
        scale_y: float,
    ) -> Optional["fitz.Rect"]:
        """Convert a RapidOCR bounding box to a PyMuPDF Rect."""
        if not bbox or len(bbox) < 4:
            return None

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        x0 = min(xs) * scale_x
        y0 = min(ys) * scale_y
        x1 = max(xs) * scale_x
        y1 = max(ys) * scale_y

        if x0 == x1 or y0 == y1:
            return None

        return fitz.Rect(x0, y0, x1, y1)
