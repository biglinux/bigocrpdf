#!/usr/bin/env python3
"""
Professional PDF OCR Application
High-quality searchable PDF generation using RapidOCR PP-OCRv5

Features:
- High-resolution image extraction (300 DPI)
- Adaptive image preprocessing
- Server-grade OCR models for maximum accuracy
- Precise invisible text layer positioning with rotation support
- PDF/A-2b compliance
- Progress tracking and comprehensive logging

Author: Professional OCR Suite
License: MIT
"""

import re
import threading
from collections.abc import Callable
from pathlib import Path

# Import rapidocr with fallback to other Python versions
# Import unified OCRConfig and data classes from the single source of truth
from bigocrpdf.services.rapidocr_service.config import (
    OCRBoxData,
    OCRConfig,
    OCRResult,
    ProcessingStats,
)

# Import extracted logic
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    PDFImageExtractor,
    has_native_text,
)

# Import ImagePreprocessor from dedicated module (single source of truth)
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.services.rapidocr_service.renderer import TextLayerRenderer
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.python_compat import (
    setup_python_compatibility,
)

# Setup compatibility paths before importing rapidocr
setup_python_compatibility()


from bigocrpdf.services.rapidocr_service.backend_ocr import BackendOCRMixin
from bigocrpdf.services.rapidocr_service.backend_pipeline import BackendPipelineMixin
from bigocrpdf.services.rapidocr_service.backend_text_layer import BackendTextLayerMixin


def _is_structural_line(text: str) -> bool:
    """Check if a line is a structural element (heading, list, section)."""
    s = text.strip()
    if not s:
        return False
    # Numbered section: "1.", "2.3", etc.
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-ZÀ-Ú]", s):
        return True
    # Bullet/dash lists
    if re.match(r"^[-•●○■□▪►]\s", s):
        return True
    # ALL CAPS heading (> 4 chars, single or multi-word)
    if s.isupper() and len(s) > 4:
        return True
    return False


class ProfessionalPDFOCR(
    BackendPipelineMixin,
    BackendTextLayerMixin,
    BackendOCRMixin,
):
    """High-quality PDF OCR engine for professional document processing."""

    # Class-level cache for OpenVINO availability (check once per process)
    _openvino_available: bool | None = None

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()
        self.cancel_event = threading.Event()  # Cooperative cancellation
        self.preprocessor = ImagePreprocessor(self.config)
        self.extractor = PDFImageExtractor(self.config.dpi)
        self.renderer = TextLayerRenderer(self.config.dpi)

    def _format_ocr_text(self, ocr_results: list[OCRResult], page_width: float) -> str:
        """Format OCR results into readable text preserving spatial layout.

        Performs full layout analysis: line grouping, page margin detection,
        centering detection, indentation, table alignment, and paragraph breaks.
        """
        if not ocr_results:
            return ""

        # Sort by y, then x
        sorted_results = sorted(
            ocr_results,
            key=lambda r: (min(p[1] for p in r.box), min(p[0] for p in r.box)),
        )

        # --- Phase 1: Group into visual lines by y-proximity ---
        lines = self._group_into_lines(sorted_results)

        # --- Phase 2: Compute page metrics ---
        metrics = self._compute_page_metrics(lines, page_width)

        # --- Phase 3: Classify lines and detect table regions ---
        table_ranges = self._detect_table_ranges(lines, page_width)
        table_line_set: set[int] = set()
        for start, end in table_ranges:
            for idx in range(start, end):
                table_line_set.add(idx)

        line_props = []
        for i, line in enumerate(lines):
            in_table = i in table_line_set
            props = self._classify_line_layout(line, metrics, in_table)
            line_props.append(props)

        # --- Phase 4: Build formatted text ---
        parts: list[str] = []
        prev_bottom = -1.0
        i = 0

        while i < len(lines):
            line = lines[i]
            props = line_props[i]
            line_top = props["top"]
            line_bottom = props["bottom"]
            line_h = max(line_bottom - line_top, 1.0)

            # Check if this line starts a table
            table_range = None
            for tr in table_ranges:
                if tr[0] == i:
                    table_range = tr
                    break

            if table_range:
                start, end = table_range
                if prev_bottom >= 0:
                    parts.append("")
                table_text = self._format_table_region(lines[start:end], metrics)
                parts.append(table_text)
                last = lines[end - 1]
                prev_bottom = max(max(p[1] for p in r.box) for r in last)
                i = end
                continue

            # Paragraph break detection
            if prev_bottom >= 0:
                gap = line_top - prev_bottom
                is_gap_break = gap > line_h * 0.6
                # Also break if previous line was non-indented and this one is indented
                prev_props = line_props[i - 1] if i > 0 else None
                is_indent_break = (
                    prev_props is not None
                    and props["indent_chars"] > 0
                    and prev_props["indent_chars"] == 0
                    and not prev_props["is_centered"]
                )
                if is_gap_break or is_indent_break:
                    parts.append("")

            # Format the line based on classification
            raw_text = self._format_text_line(line, metrics)

            if props["is_centered"]:
                pad = max(0, (metrics["output_width"] - len(raw_text)) // 2)
                parts.append(" " * pad + raw_text)
            elif props["indent_chars"] > 0:
                parts.append(" " * props["indent_chars"] + raw_text)
            else:
                parts.append(raw_text)

            prev_bottom = line_bottom
            i += 1

        raw = "\n".join(parts)

        # --- Phase 5: Post-process text ---
        raw = self._reflow_paragraphs(raw, line_props, table_line_set)
        return raw

    @staticmethod
    def _group_into_lines(sorted_results: list["OCRResult"]) -> list[list["OCRResult"]]:
        """Group sorted OCR results into visual lines by y-proximity."""
        lines: list[list["OCRResult"]] = []
        current_line: list["OCRResult"] = [sorted_results[0]]

        for r in sorted_results[1:]:
            ref = current_line[0]
            ref_ys = [p[1] for p in ref.box]
            ref_center = (min(ref_ys) + max(ref_ys)) / 2
            ref_h = max(ref_ys) - min(ref_ys)

            curr_ys = [p[1] for p in r.box]
            curr_center = (min(curr_ys) + max(curr_ys)) / 2
            curr_h = max(curr_ys) - min(curr_ys)

            threshold = min(ref_h, curr_h) * 0.5 if min(ref_h, curr_h) > 0 else 10
            if abs(curr_center - ref_center) <= threshold:
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
        lines.append(current_line)

        for line in lines:
            line.sort(key=lambda r: min(p[0] for p in r.box))

        return lines

    @staticmethod
    def _compute_page_metrics(lines: list[list["OCRResult"]], page_width: float) -> dict:
        """Compute page-level metrics for layout formatting.

        Returns dict with: body_left_margin, avg_char_width,
        page_center, output_width.
        """
        page_center = page_width / 2

        # Collect left-x of all lines and compute character width stats
        left_xs: list[float] = []
        total_text_len = 0
        total_text_width = 0.0

        for line in lines:
            left_x = min(min(p[0] for p in r.box) for r in line)
            left_xs.append(left_x)
            for r in line:
                xs = [p[0] for p in r.box]
                w = max(xs) - min(xs)
                if w > 0 and len(r.text) > 0:
                    total_text_len += len(r.text)
                    total_text_width += w

        avg_char_width = total_text_width / total_text_len if total_text_len > 0 else 15.0

        # Body left margin: use percentile 15 of left-x values
        # This captures the typical body text start, ignoring outliers
        # (centered text, indented first-lines, table data)
        if left_xs:
            sorted_lx = sorted(left_xs)
            idx = max(0, int(len(sorted_lx) * 0.15))
            body_left_margin = sorted_lx[idx]
        else:
            body_left_margin = 0.0

        # Estimate output width in characters from page content
        max_text_len = 0
        for line in lines:
            text_parts = [r.text for r in line]
            line_len = sum(len(t) for t in text_parts) + len(text_parts) - 1
            max_text_len = max(max_text_len, line_len)
        output_width = max(max_text_len + 8, 80)

        return {
            "body_left_margin": body_left_margin,
            "avg_char_width": avg_char_width,
            "page_center": page_center,
            "page_width": page_width,
            "output_width": output_width,
        }

    @staticmethod
    def _classify_line_layout(line: list["OCRResult"], metrics: dict, in_table: bool) -> dict:
        """Classify a single line's layout properties.

        Returns dict with: left_x, right_x, top, bottom, is_centered,
        indent_chars, in_table.
        """
        left_x = min(min(p[0] for p in r.box) for r in line)
        right_x = max(max(p[0] for p in r.box) for r in line)
        top = min(min(p[1] for p in r.box) for r in line)
        bottom = max(max(p[1] for p in r.box) for r in line)
        line_center = (left_x + right_x) / 2
        line_width = right_x - left_x

        page_center = metrics["page_center"]
        page_width = metrics["page_width"]
        body_left = metrics["body_left_margin"]
        avg_cw = metrics["avg_char_width"]

        # Centering: short line roughly symmetric around page center
        is_centered = False
        if line_width < page_width * 0.55 and not in_table:
            left_margin = left_x
            right_margin = page_width - right_x
            if left_margin > page_width * 0.15 and right_margin > page_width * 0.15:
                margin_diff = abs(left_margin - right_margin)
                if margin_diff < page_width * 0.08:
                    is_centered = True

        # Indentation: how far right of the body margin (in characters)
        indent_chars = 0
        if not is_centered and not in_table:
            indent_px = left_x - body_left
            if indent_px > avg_cw * 1.5:
                indent_chars = round(indent_px / avg_cw)
                indent_chars = min(indent_chars, 16)  # Cap at 16

        return {
            "left_x": left_x,
            "right_x": right_x,
            "top": top,
            "bottom": bottom,
            "is_centered": is_centered,
            "indent_chars": indent_chars,
            "in_table": in_table,
        }

    @staticmethod
    def _detect_table_ranges(
        lines: list[list["OCRResult"]], page_width: float
    ) -> list[tuple[int, int]]:
        """Detect table regions in a list of visual lines.

        A table region is 2+ consecutive multi-box lines where
        box x-positions form a consistent column grid.
        """
        ranges: list[tuple[int, int]] = []
        n = len(lines)
        i = 0

        while i < n:
            if len(lines[i]) < 2:
                i += 1
                continue

            start = i
            col_starts = [min(p[0] for p in r.box) for r in lines[i]]

            j = i + 1
            while j < n and len(lines[j]) >= 2:
                line_cols = sorted(min(p[0] for p in r.box) for r in lines[j])
                ref_cols = sorted(col_starts)
                min_cols = min(len(line_cols), len(ref_cols))
                aligned = 0
                for lc in line_cols:
                    for rc in ref_cols:
                        if abs(lc - rc) < page_width * 0.05:
                            aligned += 1
                            break
                if aligned >= max(min_cols * 0.5, 1):
                    for lc in line_cols:
                        if not any(abs(lc - rc) < page_width * 0.05 for rc in col_starts):
                            col_starts.append(lc)
                    j += 1
                else:
                    break

            if j - start >= 2:
                ranges.append((start, j))
                i = j
            else:
                i += 1

        return ranges

    @staticmethod
    def _format_table_region(table_lines: list[list["OCRResult"]], metrics: dict) -> str:
        """Format a table region with aligned columns.

        Uses ordinal box position for rows with the modal box count,
        and x-position fallback for rows with different counts.
        Adds indentation if the table is offset from body margin.
        """
        if not table_lines:
            return ""

        from collections import Counter

        counts = Counter(len(line) for line in table_lines)
        modal_cols = counts.most_common(1)[0][0]

        # Collect reference column x-centers from ordinal rows
        ref_x: list[list[float]] = [[] for _ in range(modal_cols)]
        for line in table_lines:
            if len(line) == modal_cols:
                for ci, r in enumerate(line):
                    xs = [p[0] for p in r.box]
                    ref_x[ci].append((min(xs) + max(xs)) / 2)

        ref_centers = [sum(xs) / len(xs) if xs else 0 for xs in ref_x]

        # Build cell grid
        grid: list[list[str]] = []
        for line in table_lines:
            row = [""] * modal_cols
            if len(line) == modal_cols:
                for ci, r in enumerate(line):
                    row[ci] = r.text
            else:
                for r in line:
                    xs = [p[0] for p in r.box]
                    x_center = (min(xs) + max(xs)) / 2
                    best_col = 0
                    best_dist = float("inf")
                    for ci, cx in enumerate(ref_centers):
                        d = abs(x_center - cx)
                        if d < best_dist:
                            best_dist = d
                            best_col = ci
                    if row[best_col]:
                        row[best_col] += " " + r.text
                    else:
                        row[best_col] = r.text
            grid.append(row)

        # Column widths
        col_widths = [0] * modal_cols
        for row in grid:
            for ci, cell in enumerate(row):
                col_widths[ci] = max(col_widths[ci], len(cell))

        # Table indentation from body margin
        table_left = min(min(min(p[0] for p in r.box) for r in line) for line in table_lines)
        body_left = metrics["body_left_margin"]
        avg_cw = metrics["avg_char_width"]
        indent_px = table_left - body_left
        indent_chars = max(0, round(indent_px / avg_cw)) if indent_px > avg_cw * 1.5 else 0
        indent_str = " " * indent_chars

        # Format rows
        formatted_rows: list[str] = []
        for row in grid:
            cells = [cell.ljust(col_widths[ci]) for ci, cell in enumerate(row)]
            formatted_rows.append(indent_str + "  ".join(cells).rstrip())

        return "\n".join(formatted_rows)

    @staticmethod
    def _format_text_line(line: list["OCRResult"], metrics: dict) -> str:
        """Format a single non-table text line with proportional spacing."""
        avg_cw = metrics["avg_char_width"]
        line_text = ""
        for j, r in enumerate(line):
            if j > 0:
                prev_r = line[j - 1]
                prev_right = max(p[0] for p in prev_r.box)
                curr_left = min(p[0] for p in r.box)
                h_gap = curr_left - prev_right

                if avg_cw > 0:
                    space_count = max(1, round(h_gap / avg_cw))
                    space_count = min(space_count, 8)
                else:
                    space_count = 1

                line_text += " " * space_count
            line_text += r.text
        return line_text

    @staticmethod
    def _reflow_paragraphs(
        text: str,
        line_props: list[dict],
        table_line_set: set[int],
    ) -> str:
        """Join continuation lines to form flowing paragraphs.

        Detects lines that are continuations of the previous line (word
        wrapping in the original document) and joins them, eliminating
        unnecessary line breaks within paragraphs.
        """
        import re

        lines = text.split("\n")
        result: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip()

            # Keep empty lines (paragraph breaks)
            if not stripped:
                result.append("")
                i += 1
                continue

            # Accumulate paragraph lines
            para = stripped
            i += 1

            while i < len(lines):
                next_line = lines[i].rstrip()

                # Stop at empty line (paragraph break)
                if not next_line:
                    break

                next_stripped = next_line.lstrip()

                # Stop at lines that look like new structural elements
                if _is_structural_line(next_stripped):
                    break

                # Stop if indentation changes significantly (new paragraph/section)
                cur_indent = len(para) - len(para.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())
                # Centered lines have leading spaces — don't join them
                if next_indent > cur_indent + 4 and next_indent > 8:
                    break

                # Stop if next line looks centered (many leading spaces)
                # vs current line that isn't centered
                if next_indent > 15 and cur_indent < 8:
                    break

                # Check if this is a continuation
                para_end = para.rstrip()
                if not para_end:
                    break

                last_char = para_end[-1]

                # Full-width line continuation: line fills the width
                # and next line starts at left margin
                is_continuation = (
                    len(para_end.lstrip()) > 50
                    and not para_end.rstrip().endswith((".", "!", "?", ":"))
                    and abs(next_indent - cur_indent) <= 2
                )

                # Explicit hyphenated word break: "word-\n" + "continuation"
                is_hyphen_break = (
                    para_end.endswith("-")
                    and len(para_end) > 1
                    and para_end[-2].isalpha()
                    and next_stripped
                    and next_stripped[0].islower()
                )

                if is_hyphen_break:
                    # Join removing hyphen (word was split with hyphen)
                    para = para_end[:-1] + next_stripped
                    i += 1
                elif is_continuation:
                    # Join with space (paragraph continuation)
                    para = para_end + " " + next_stripped
                    i += 1
                else:
                    break

            result.append(para)

        return "\n".join(result)

    def _collect_ocr_boxes(
        self,
        ocr_results: list[OCRResult],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> list[OCRBoxData]:
        """Collect structured OCR box data for high-fidelity export.

        Converts pixel coordinates to percentages (x, y, width) and
        calculates height in points for font size estimation.
        """
        boxes = []
        dpi = self.config.dpi or 300  # Default DPI for conversion

        for r in ocr_results:
            xs = [p[0] for p in r.box]
            ys = [p[1] for p in r.box]
            box_x = min(xs)
            box_y = min(ys)
            box_w = max(xs) - box_x
            box_h = max(ys) - box_y

            x_pct = (box_x / page_width) * 100 if page_width > 0 else 0
            y_pct = (box_y / page_height) * 100 if page_height > 0 else 0
            w_pct = (box_w / page_width) * 100 if page_width > 0 else 0

            # Convert height from pixels to points (1 inch = 72 points)
            # height_pts = box_h_pixels * (72 points/inch) / (dpi pixels/inch)
            height_pts = (box_h * 72) / dpi

            boxes.append(
                OCRBoxData(
                    text=r.text,
                    x=x_pct,
                    y=y_pct,
                    width=w_pct,
                    height=height_pts,
                    confidence=r.confidence,
                    page_num=page_num,
                )
            )
        return boxes

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

        if not input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        logger.info(f"Processing: {input_pdf}")
        logger.info(f"Output: {output_pdf}")

        # Check if PDF has native text (mixed content)
        if (
            not self.config.force_full_ocr
            and not self.config.replace_existing_ocr
            and has_native_text(input_pdf)
        ):
            logger.info("Detected mixed content PDF (text + images). Using preservation mode.")
            return self._process_mixed_content_pdf(input_pdf, output_pdf, progress_callback)
        else:
            if self.config.force_full_ocr:
                logger.info("Force full OCR mode (editor-merged file). Using full OCR mode.")
            elif self.config.replace_existing_ocr:
                logger.info("Replace existing OCR mode. Using full OCR mode.")
            else:
                logger.info("Detected image-only PDF. Using full OCR mode.")
            return self._process_image_only_pdf(input_pdf, output_pdf, progress_callback)
