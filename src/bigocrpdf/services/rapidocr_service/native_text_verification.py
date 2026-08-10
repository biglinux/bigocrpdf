"""Box-level native PDF text verification for auto-verified OCR mode."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

from bigocrpdf.services.rapidocr_service.config import OcrLine, OCRResult, OcrWord

_MIN_OVERLAP_RATIO = 0.20
_MIN_TEXT_SIMILARITY = 0.45
_MIN_LENGTH_RATIO = 0.45
_MAX_SUSPICIOUS_RATIO = 0.03
_MAX_NONPRINT_RATIO = 0.02


@dataclass(frozen=True)
class NativeTextSpan:
    """Native PDF text span in OCR image pixel coordinates."""

    text: str
    bbox: list[float]


@dataclass(frozen=True)
class AutoVerifiedPage:
    """Result of replacing OCR lines with trusted native PDF spans."""

    lines: list[OcrLine]
    native_spans: int
    accepted_lines: int
    rejected_lines: int


@dataclass
class _ParsedWord:
    text: str
    bbox: list[float]


@dataclass
class _ParsedLine:
    words: list[_ParsedWord]


@dataclass
class _ParsedPage:
    width: float
    height: float
    lines: list[_ParsedLine]


def extract_native_text_spans(
    pdf_path: Path,
    page_num: int,
    image_size_px: tuple[int, int],
    source_rect_pts: tuple[float, float, float, float] | None = None,
    timeout: int = 30,
) -> list[NativeTextSpan]:
    """Extract native text spans for one page using Poppler bbox output."""
    try:
        result = subprocess.run(
            [
                "pdftotext",
                "-bbox-layout",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                str(pdf_path),
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return parse_pdftotext_bbox_layout(result.stdout, image_size_px, source_rect_pts)


def parse_pdftotext_bbox_layout(
    html: str,
    image_size_px: tuple[int, int],
    source_rect_pts: tuple[float, float, float, float] | None = None,
) -> list[NativeTextSpan]:
    """Parse ``pdftotext -bbox-layout`` HTML into native text spans."""
    spans: list[NativeTextSpan] = []
    parser = _BboxLayoutParser()
    parser.feed(html)
    for page in parser.pages:
        if page.width <= 0 or page.height <= 0:
            continue
        for line in page.lines:
            for word in line.words:
                word_span = _word_span_from_parsed_word(
                    word,
                    image_size_px,
                    page.width,
                    page.height,
                    source_rect_pts,
                )
                if word_span is not None:
                    spans.append(word_span)
    return spans


class _BboxLayoutParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[_ParsedPage] = []
        self._current_page: _ParsedPage | None = None
        self._current_line_words: list[_ParsedWord] = []
        self._current_word_bbox: list[float] = []
        self._current_word_parts: list[str] = []
        self._in_word = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        if tag == "page":
            self._current_page = _ParsedPage(
                width=_float_attr(attr_map, "width"),
                height=_float_attr(attr_map, "height"),
                lines=[],
            )
            self.pages.append(self._current_page)
        elif tag == "line":
            self._current_line_words = []
        elif tag == "word":
            self._in_word = True
            self._current_word_bbox = _bbox_from_attrs(attr_map)
            self._current_word_parts = []

    def collect_word_text(self, data: str) -> None:
        if self._in_word:
            self._current_word_parts.append(data)

    handle_data = collect_word_text

    def handle_endtag(self, tag: str) -> None:
        if tag == "word":
            self._append_current_word()
        elif tag == "line" and self._current_page is not None:
            self._append_current_line()
        elif tag == "page":
            self._current_page = None

    def _append_current_word(self) -> None:
        word = _normalize_text("".join(self._current_word_parts))
        if word and self._current_word_bbox:
            self._current_line_words.append(_ParsedWord(word, self._current_word_bbox))
        self._current_word_bbox = []
        self._current_word_parts = []
        self._in_word = False

    def _append_current_line(self) -> None:
        if self._current_line_words and self._current_page is not None:
            self._current_page.lines.append(_ParsedLine(list(self._current_line_words)))
        self._current_line_words = []


def verify_ocr_lines_with_native_spans(
    ocr_lines: list[OcrLine],
    native_spans: list[NativeTextSpan],
) -> AutoVerifiedPage:
    """Replace OCR lines with overlapping native text when it is safe."""
    verified_lines: list[OcrLine] = []
    accepted = 0
    rejected = 0

    for line in ocr_lines:
        candidate_spans = _overlapping_spans(line.bbox, native_spans)
        candidate_text = _normalize_text(" ".join(span.text for span in candidate_spans))
        if _should_accept_native_text(line.text, candidate_text):
            verified_lines.append(_native_line_from_spans(line, candidate_text, candidate_spans))
            accepted += 1
        else:
            verified_lines.append(line)
            rejected += 1

    return AutoVerifiedPage(
        lines=verified_lines,
        native_spans=len(native_spans),
        accepted_lines=accepted,
        rejected_lines=rejected,
    )


def verify_ocr_results_with_native_spans(
    ocr_results: list[OCRResult],
    native_spans: list[NativeTextSpan],
) -> tuple[list[OCRResult], int]:
    """Give each region the native text it sits on, when the two agree.

    Where a page keeps its native text and an image containing the same words
    is OCR'd on top, the native layer is the authoritative reading: it is
    exact, while OCR truncates long URLs and mangles rare characters. Measured
    on a real certificate, OCR lost
    ``https://assinador-web.onr.org.br/docs/UB7MR-ZF2N3-NTFLP-JMF2B`` that the
    native layer had in full.

    Substitution only, never removal: a region with no native counterpart, or
    one whose native text disagrees, keeps exactly what OCR read. So this
    cannot lose text -- at worst it changes nothing.

    Returns the results and how many were replaced.
    """
    verified: list[OCRResult] = []
    accepted = 0
    for result in ocr_results:
        bbox = _bbox_from_quad(result.box)
        if bbox is None:
            verified.append(result)
            continue
        candidates = _overlapping_spans(bbox, native_spans)
        candidate_text = _normalize_text(" ".join(span.text for span in candidates))
        if _should_accept_native_text(result.text, candidate_text):
            verified.append(
                OCRResult(text=candidate_text, box=result.box, confidence=result.confidence)
            )
            accepted += 1
        else:
            verified.append(result)
    return verified, accepted


def _bbox_from_quad(box: object) -> list[float] | None:
    """Axis-aligned bounds of an OCR quadrilateral, or None if malformed."""
    if not isinstance(box, (list, tuple)) or not box:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _word_span_from_parsed_word(
    word: _ParsedWord,
    image_size_px: tuple[int, int],
    page_width_pts: float,
    page_height_pts: float,
    source_rect_pts: tuple[float, float, float, float] | None,
) -> NativeTextSpan | None:
    bbox_px = _pdf_bbox_to_image_bbox(
        word.bbox,
        image_size_px,
        page_width_pts,
        page_height_pts,
        source_rect_pts,
    )
    if not bbox_px:
        return None
    return NativeTextSpan(text=word.text, bbox=bbox_px)


def _bbox_from_attrs(attrs: dict[str, str]) -> list[float]:
    return [
        _float_attr(attrs, "xMin"),
        _float_attr(attrs, "yMin"),
        _float_attr(attrs, "xMax"),
        _float_attr(attrs, "yMax"),
    ]


def _pdf_bbox_to_image_bbox(
    bbox_pts: list[float],
    image_size_px: tuple[int, int],
    page_width_pts: float,
    page_height_pts: float,
    source_rect_pts: tuple[float, float, float, float] | None,
) -> list[float]:
    x_min, y_min, x_max, y_max = bbox_pts
    if source_rect_pts is None:
        rect_left, rect_top, rect_width, rect_height = 0.0, 0.0, page_width_pts, page_height_pts
    else:
        rect_left, rect_bottom, rect_width, rect_height = source_rect_pts
        rect_top = page_height_pts - rect_bottom - rect_height

    if rect_width <= 0 or rect_height <= 0:
        return []
    rect_right = rect_left + rect_width
    rect_bottom = rect_top + rect_height
    if x_max <= rect_left or x_min >= rect_right:
        return []
    if y_max <= rect_top or y_min >= rect_bottom:
        return []

    x_min = max(x_min, rect_left)
    y_min = max(y_min, rect_top)
    x_max = min(x_max, rect_right)
    y_max = min(y_max, rect_bottom)

    image_width_px, image_height_px = image_size_px
    return [
        (x_min - rect_left) * image_width_px / rect_width,
        (y_min - rect_top) * image_height_px / rect_height,
        (x_max - rect_left) * image_width_px / rect_width,
        (y_max - rect_top) * image_height_px / rect_height,
    ]


def _overlapping_spans(line_bbox: list[float], spans: list[NativeTextSpan]) -> list[NativeTextSpan]:
    if len(line_bbox) != 4:
        return []
    candidates = [
        span for span in spans if _bbox_overlap_ratio(line_bbox, span.bbox) >= _MIN_OVERLAP_RATIO
    ]
    return sorted(candidates, key=lambda span: (span.bbox[1], span.bbox[0]))


def _bbox_overlap_ratio(left_bbox: list[float], right_bbox: list[float]) -> float:
    if len(left_bbox) != 4 or len(right_bbox) != 4:
        return 0.0
    left_x1, left_y1, left_x2, left_y2 = left_bbox
    right_x1, right_y1, right_x2, right_y2 = right_bbox
    inter_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    left_area = max((left_x2 - left_x1) * (left_y2 - left_y1), 1.0)
    right_area = max((right_x2 - right_x1) * (right_y2 - right_y1), 1.0)
    return inter_area / min(left_area, right_area)


def _should_accept_native_text(ocr_text: str, native_text: str) -> bool:
    native_text = _normalize_text(native_text)
    ocr_text = _normalize_text(ocr_text)
    if not native_text or not _has_meaningful_text(native_text):
        return False
    if _suspicious_ratio(native_text) > _MAX_SUSPICIOUS_RATIO:
        return False
    if _nonprint_ratio(native_text) > _MAX_NONPRINT_RATIO:
        return False

    native_len = len(native_text)
    ocr_len = len(ocr_text)
    if ocr_len > 0 and min(native_len, ocr_len) / max(native_len, ocr_len) < _MIN_LENGTH_RATIO:
        return False

    return _text_similarity(ocr_text, native_text) >= _MIN_TEXT_SIMILARITY


def _native_line_from_spans(
    original_line: OcrLine,
    native_text: str,
    spans: list[NativeTextSpan],
) -> OcrLine:
    bbox = _union_bbox([span.bbox for span in spans]) or list(original_line.bbox)
    words = [
        OcrWord(text=word, bbox=list(span.bbox), confidence=1.0)
        for span in spans
        for word in span.text.split()
    ]
    return OcrLine(
        text=native_text,
        bbox=bbox,
        words=words,
        reading_order=original_line.reading_order,
        source="pdf",
    )


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    valid_boxes = [box for box in boxes if len(box) == 4]
    if not valid_boxes:
        return []
    return [
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    ]


def _text_similarity(left: str, right: str) -> float:
    left_key = _comparison_key(left)
    right_key = _comparison_key(right)
    if not left_key and not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def _comparison_key(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def _has_meaningful_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _suspicious_ratio(text: str) -> float:
    suspicious = sum(1 for char in text if char in "�□")
    question_runs = text.count("???")
    return (suspicious + question_runs * 3) / max(len(text.strip()), 1)


def _nonprint_ratio(text: str) -> float:
    return sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t") / max(len(text), 1)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _float_attr(attrs: dict[str, str], name: str) -> float:
    try:
        return float(attrs.get(name, attrs.get(name.lower(), "0")) or 0)
    except ValueError:
        return 0.0
