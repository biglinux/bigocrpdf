"""Build structured OCR document lines from RapidOCR regions."""

from __future__ import annotations

from bigocrpdf.services.rapidocr_service.config import OcrLine, OCRResult, OcrWord

_WORD_GAP_PX = 6.0


def build_ocr_lines_from_results(text_results: list[OCRResult]) -> list[OcrLine]:
    """Convert OCR regions into positioned lines and words."""
    lines: list[OcrLine] = []
    sorted_results = sorted(
        enumerate(text_results),
        key=lambda indexed_result: _ocr_result_sort_key(*indexed_result),
    )
    for reading_order, (_index, result) in enumerate(sorted_results):
        text = result.text.strip()
        if not text:
            continue
        bbox = _ocr_result_bbox(result)
        words = _ocr_words_from_result(result, bbox)
        lines.append(
            OcrLine(
                text=text,
                bbox=bbox,
                words=words,
                reading_order=reading_order,
                source="ocr",
            )
        )
    return lines


def _ocr_result_sort_key(original_index: int, result: OCRResult) -> tuple[float, float, int]:
    bbox = _ocr_result_bbox(result)
    if not bbox:
        return (0.0, 0.0, original_index)
    return (bbox[1], bbox[0], original_index)


def _ocr_result_bbox(result: OCRResult) -> list[float]:
    if not result.box:
        return []
    points = [(float(point[0]), float(point[1])) for point in result.box if len(point) >= 2]
    if not points:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _ocr_words_from_result(result: OCRResult, bbox: list[float]) -> list[OcrWord]:
    tokens = result.text.strip().split()
    if not tokens:
        return []
    if not bbox:
        return [OcrWord(token, [], result.confidence) for token in tokens]

    left, top, right, bottom = bbox
    width = max(right - left, float(len(tokens)))
    total_gap = _WORD_GAP_PX * max(len(tokens) - 1, 0)
    available_width = max(width - total_gap, float(len(tokens)))
    total_chars = max(sum(len(token) for token in tokens), 1)
    cursor = left
    words: list[OcrWord] = []
    for token in tokens:
        token_width = max(1.0, available_width * len(token) / total_chars)
        words.append(
            OcrWord(
                text=token,
                bbox=[cursor, top, cursor + token_width, bottom],
                confidence=result.confidence,
            )
        )
        cursor += token_width + _WORD_GAP_PX
    return words
