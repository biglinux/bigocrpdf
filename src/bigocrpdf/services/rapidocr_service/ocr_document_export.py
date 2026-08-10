"""Export structured OCR documents through the shared document pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from statistics import median

from bigocrpdf.services.rapidocr_service.config import (
    OcrDocument,
    OcrLayoutBlock,
    OcrLine,
    OcrPage,
    OCRResult,
    OcrWord,
)
from bigocrpdf.utils.odf_builder import ExportCancelled, create_odf
from bigocrpdf.utils.tsv_odf_converter import (
    _build_front_matter,
    create_markdown,
    create_text,
    fix_cross_page_breaks,
    process_page,
)
from bigocrpdf.utils.tsv_parser import DocElement, Word

_WORD_GAP_PX = 6.0
_FORM_FIELD_GAP_PX = 120.0
_SYNTHETIC_WORD_HEIGHT = 12.0
_SYNTHETIC_WORD_WIDTH = 8.0
_SYNTHETIC_LINE_GAP = 18.0


@dataclass(frozen=True)
class _OcrToken:
    text: str
    gap_after: float = _WORD_GAP_PX


def ocr_document_to_pages_elements(document: OcrDocument) -> list[list[DocElement]]:
    """Convert canonical OCR output into structured document elements."""
    pages_elements: list[list[DocElement]] = []
    for page in _sorted_pages(document):
        elements = _page_elements(page)
        _apply_horizontal_alignment(page, elements)
        pages_elements.append(elements)
    return fix_cross_page_breaks(pages_elements)


def enrich_ocr_document_layout(document: OcrDocument) -> None:
    """Populate layout blocks for pages that only have OCR lines/results."""
    for page in _sorted_pages(document):
        if page.layout_blocks:
            continue
        page.layout_blocks = _layout_blocks_from_elements(_page_elements_without_blocks(page))


def convert_ocr_document_to_text(document: OcrDocument) -> str:
    """Convert a structured OCR document to formatted plain text."""
    return create_text(ocr_document_to_pages_elements(document))


def convert_ocr_document_to_markdown(
    document: OcrDocument,
    source_path: str | None = None,
    include_front_matter: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    """Convert a structured OCR document to Markdown."""
    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled
    pages_elements = ocr_document_to_pages_elements(document)
    body = create_markdown(pages_elements)
    if include_front_matter:
        source = source_path or "ocr-document"
        return "\n".join(_build_front_matter(source, len(pages_elements))) + body
    return body


def convert_ocr_document_to_odf(
    document: OcrDocument,
    odf_path: str,
    cancel_event: threading.Event | None = None,
) -> str:
    """Convert a structured OCR document to an ODT file."""
    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled
    pages_elements = ocr_document_to_pages_elements(document)
    create_odf(
        pages_elements,
        odf_path,
        page_size_cm=_odf_page_size_cm(document),
        body_font_size_pt=_odf_body_font_size_pt(document),
        cancel_event=cancel_event,
    )
    return odf_path


def _odf_page_size_cm(document: OcrDocument) -> tuple[float, float] | None:
    """Scale the first valid source-page aspect ratio to an A4-length edge."""
    for page in _sorted_pages(document):
        if page.width_px <= 0 or page.height_px <= 0:
            continue
        width_ratio = page.width_px / page.height_px
        if width_ratio <= 1:
            return (29.7 * width_ratio, 29.7)
        return (29.7, 29.7 / width_ratio)
    return None


def _odf_body_font_size_pt(document: OcrDocument) -> float:
    """Estimate editable body typography from OCR line geometry."""
    line_sizes = [
        (float(line.bbox[3]) - float(line.bbox[1])) * 72.0 / page.dpi
        for page in _sorted_pages(document)
        if page.dpi > 0
        for line in page.lines
        if len(line.bbox) >= 4 and float(line.bbox[3]) > float(line.bbox[1])
    ]
    result_sizes = [
        (max(point[1] for point in result.box) - min(point[1] for point in result.box))
        * 72.0
        / page.dpi
        for page in _sorted_pages(document)
        if page.dpi > 0
        for result in page.text_results
        if len(result.box) >= 2 and all(len(point) >= 2 for point in result.box)
    ]
    sizes = line_sizes or result_sizes
    return min(max(median(sizes), 6.5), 10.5) if sizes else 9.0


def _page_words(text_results: list[OCRResult]) -> list[Word]:
    sorted_results = [
        result
        for _original_index, result in sorted(
            enumerate(text_results),
            key=lambda indexed_result: _ocr_result_sort_key(*indexed_result),
        )
    ]
    words: list[Word] = []
    for line_index, result in enumerate(sorted_results):
        words.extend(_ocr_result_words(result, fallback_top=line_index * _SYNTHETIC_LINE_GAP))
    return words


def _sorted_pages(document: OcrDocument) -> list[OcrPage]:
    return sorted(document.pages, key=lambda ocr_page: ocr_page.page_index)


def _page_elements(page: OcrPage) -> list[DocElement]:
    if page.layout_blocks:
        return _layout_blocks_to_elements(page.layout_blocks)
    return _page_elements_without_blocks(page)


def _page_elements_without_blocks(page: OcrPage) -> list[DocElement]:
    words = _page_structured_words(page.lines)
    if not words:
        words = _page_words(page.text_results)
    if words:
        return process_page(words, page.page_index)
    if page.native_text.strip():
        return _native_text_elements(page.native_text)
    return []


def _apply_horizontal_alignment(page: OcrPage, elements: list[DocElement]) -> None:
    """Recover centered/right-aligned blocks from positioned OCR lines."""
    if page.width_px <= 0 or not page.lines:
        return
    ordered = sorted(elements, key=lambda element: element.y_top)
    for index, element in enumerate(ordered):
        if element.kind == "table":
            continue
        next_top = ordered[index + 1].y_top if index + 1 < len(ordered) else float("inf")
        block_lines = [
            line
            for line in page.lines
            if len(line.bbox) >= 4
            and _bbox_top(line.bbox) >= element.y_top - 2
            and _bbox_top(line.bbox) < next_top - 2
        ]
        if not block_lines:
            continue
        left = min(_bbox_left(line.bbox) for line in block_lines)
        right = max(float(line.bbox[2]) for line in block_lines)
        width = right - left
        center_delta = abs((left + right) / 2 - page.width_px / 2)
        if width < page.width_px * 0.8 and center_delta <= page.width_px * 0.08:
            element.text_align = "center"
        elif right >= page.width_px * 0.92 and left >= page.width_px * 0.35:
            element.text_align = "end"


def _layout_blocks_from_elements(elements: list[DocElement]) -> list[OcrLayoutBlock]:
    return [
        OcrLayoutBlock(
            kind=element.kind,
            text=element.text,
            rows=[list(row) for row in element.rows],
            raw_lines=list(element.raw_lines),
            indent_chars=element.indent_chars,
            y_top=element.y_top,
            reading_order=index,
        )
        for index, element in enumerate(elements)
    ]


def _layout_blocks_to_elements(blocks: list[OcrLayoutBlock]) -> list[DocElement]:
    return [
        DocElement(
            kind=block.kind,
            text=block.text,
            rows=[list(row) for row in block.rows],
            raw_lines=list(block.raw_lines),
            indent_chars=block.indent_chars,
            y_top=block.y_top,
        )
        for block in sorted(blocks, key=lambda item: (item.reading_order, item.y_top))
    ]


def _page_structured_words(lines: list[OcrLine]) -> list[Word]:
    words: list[Word] = []
    for fallback_top, line in enumerate(
        sorted(
            lines,
            key=lambda item: (item.reading_order, _bbox_top(item.bbox), _bbox_left(item.bbox)),
        )
    ):
        if line.words:
            for word in line.words:
                words.extend(_structured_word_to_words(word, fallback_top * _SYNTHETIC_LINE_GAP))
        elif line.text.strip():
            result = OCRResult(line.text, _box_from_bbox(line.bbox), 0.0)
            words.extend(_ocr_result_words(result, fallback_top * _SYNTHETIC_LINE_GAP))
    return words


def _structured_word_to_words(word: OcrWord, fallback_top: float) -> list[Word]:
    text = word.text.strip()
    if not text:
        return []
    tokens = _split_form_tokens(text)
    if not tokens:
        return []
    if not word.bbox:
        return _synthetic_words(tokens, fallback_top)

    left, top, right, bottom = word.bbox
    width = max(float(right) - float(left), 1.0)
    height = max(float(bottom) - float(top), 1.0)
    if len(tokens) == 1:
        return [Word(tokens[0].text, float(left), float(top), width, height)]

    total_gap = sum(token.gap_after for token in tokens[:-1])
    available_width = max(width - total_gap, float(len(tokens)))
    total_chars = max(sum(len(token.text) for token in tokens), 1)
    cursor = float(left)
    words: list[Word] = []
    for token in tokens:
        token_width = max(1.0, available_width * len(token.text) / total_chars)
        words.append(Word(token.text, cursor, float(top), token_width, height))
        cursor += token_width + token.gap_after
    return words


def _bbox_left(bbox: list[float]) -> float:
    return float(bbox[0]) if len(bbox) >= 1 else 0.0


def _bbox_top(bbox: list[float]) -> float:
    return float(bbox[1]) if len(bbox) >= 2 else 0.0


def _box_from_bbox(bbox: list[float]) -> list[list[float]]:
    if len(bbox) < 4:
        return []
    left, top, right, bottom = [float(value) for value in bbox[:4]]
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _ocr_result_sort_key(original_index: int, result: OCRResult) -> tuple[float, float, int]:
    bounds = _ocr_result_bounds(result)
    if bounds is None:
        return (0.0, 0.0, original_index)
    left, top, _width, _height = bounds
    return (top, left, original_index)


def _ocr_result_words(result: OCRResult, fallback_top: float) -> list[Word]:
    text = result.text.strip()
    if not text:
        return []
    tokens = _split_form_tokens(text)
    if not tokens:
        return []

    bounds = _ocr_result_bounds(result)
    if bounds is None:
        return _synthetic_words(tokens, fallback_top)

    left, top, width, height = bounds
    if len(tokens) == 1:
        return [Word(tokens[0].text, left, top, max(width, 1.0), max(height, 1.0))]

    total_gap = sum(token.gap_after for token in tokens[:-1])
    available_width = max(width - total_gap, float(len(tokens)))
    total_chars = max(sum(len(token.text) for token in tokens), 1)
    cursor = left
    words: list[Word] = []
    for token in tokens:
        token_width = max(1.0, available_width * len(token.text) / total_chars)
        words.append(Word(token.text, cursor, top, token_width, max(height, 1.0)))
        cursor += token_width + token.gap_after
    return words


def _ocr_result_bounds(result: OCRResult) -> tuple[float, float, float, float] | None:
    if not result.box:
        return None
    points = [(float(point[0]), float(point[1])) for point in result.box if len(point) >= 2]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    top = min(ys)
    return (left, top, max(xs) - left, max(ys) - top)


def _split_form_tokens(text: str) -> list[_OcrToken]:
    tokens: list[_OcrToken] = []
    for raw_token in text.split():
        tokens.extend(_split_form_token(raw_token))
    return tokens


def _split_form_token(raw_token: str) -> list[_OcrToken]:
    if not raw_token:
        return []

    token_parts: list[_OcrToken] = []
    cursor = 0
    for index, char in enumerate(raw_token):
        if not _is_form_separator(raw_token, index):
            continue
        separator_end = index + 1
        while separator_end < len(raw_token) and raw_token[separator_end] == char:
            separator_end += 1
        before = raw_token[cursor:separator_end]
        if before:
            token_parts.append(_OcrToken(before, _FORM_FIELD_GAP_PX))
        cursor = separator_end
    if cursor < len(raw_token):
        token_parts.append(_OcrToken(raw_token[cursor:]))
    return token_parts or [_OcrToken(raw_token)]


def _is_form_separator(token: str, index: int) -> bool:
    """Whether this colon separates a form label from its value.

    ``Nome: Ana`` is a label and a value; ``https://onr.org.br`` and ``14:30``
    are one thing each. Splitting them corrupts the text -- measured, a
    certificate's validation URL was cut into ``https`` and
    ``//assinador-web.onr.org.br/...``, and the address a reader needs to
    follow no longer existed anywhere in the export.
    """
    if token[index] != ":":
        return False
    if token[index + 1 : index + 3] == "//":
        return False
    if _looks_like_time(token, index):
        return False
    has_label = index > 0 and any(character.isalnum() for character in token[:index])
    has_value = index + 1 < len(token) and any(
        character.isalnum() for character in token[index + 1 :]
    )
    return has_label and has_value


def _looks_like_time(token: str, index: int) -> bool:
    """A colon between digits joins them: 14:30, 08:15:42."""
    before = token[index - 1] if index > 0 else ""
    after = token[index + 1] if index + 1 < len(token) else ""
    return before.isdigit() and after.isdigit()


def _synthetic_words(tokens: list[_OcrToken], top: float) -> list[Word]:
    cursor = 0.0
    words: list[Word] = []
    for token in tokens:
        width = max(float(len(token.text)) * _SYNTHETIC_WORD_WIDTH, 1.0)
        words.append(Word(token.text, cursor, top, width, _SYNTHETIC_WORD_HEIGHT))
        cursor += width + token.gap_after
    return words


def _native_text_elements(native_text: str) -> list[DocElement]:
    raw_lines = [line.rstrip() for line in native_text.splitlines() if line.strip()]
    text = " ".join(line.strip() for line in raw_lines).strip()
    if not text:
        return []
    return [DocElement("paragraph", text, raw_lines=raw_lines)]
