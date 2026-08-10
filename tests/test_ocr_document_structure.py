"""Turning OCR regions into positioned lines and words.

Everything structured -- reading order, the TXT/Markdown/ODT exports, the
per-word boxes in the sidecar -- is built on this module, and no test imported
it. A region arrives as one box for a whole line, so the word boxes inside it
are estimated by splitting that box proportionally; getting that wrong
misplaces every word in a document without changing a single character of the
extracted text.
"""

import pytest

from bigocrpdf.services.rapidocr_service.config import OCRResult
from bigocrpdf.services.rapidocr_service.ocr_document_structure import (
    _WORD_GAP_PX,
    build_ocr_lines_from_results,
)


def _region(text: str, left=0.0, top=0.0, right=300.0, bottom=20.0, confidence=0.9) -> OCRResult:
    return OCRResult(
        text=text,
        box=[[left, top], [right, top], [right, bottom], [left, bottom]],
        confidence=confidence,
    )


class TestReadingOrder:
    def test_lines_are_ordered_top_to_bottom_then_left_to_right(self):
        results = [
            _region("terceira", top=200.0, bottom=220.0),
            _region("primeira", top=100.0, bottom=120.0),
            _region("segunda-direita", left=400.0, right=700.0, top=150.0, bottom=170.0),
            _region("segunda-esquerda", left=0.0, right=300.0, top=150.0, bottom=170.0),
        ]

        lines = build_ocr_lines_from_results(results)

        assert [line.text for line in lines] == [
            "primeira",
            "segunda-esquerda",
            "segunda-direita",
            "terceira",
        ]

    def test_reading_order_is_numbered_from_zero(self):
        lines = build_ocr_lines_from_results(
            [_region("b", top=50.0, bottom=70.0), _region("a", top=10.0, bottom=30.0)]
        )

        assert [line.reading_order for line in lines] == [0, 1]
        assert lines[0].text == "a"

    def test_ties_keep_the_engine_order(self):
        """Two regions at the same position must not reorder arbitrarily."""
        results = [_region("primeiro"), _region("segundo")]

        lines = build_ocr_lines_from_results(results)

        assert [line.text for line in lines] == ["primeiro", "segundo"]

    def test_the_input_order_does_not_change_the_result(self):
        results = [
            _region("a", top=10.0, bottom=30.0),
            _region("b", top=50.0, bottom=70.0),
            _region("c", top=90.0, bottom=110.0),
        ]

        forward = [line.text for line in build_ocr_lines_from_results(results)]
        backward = [line.text for line in build_ocr_lines_from_results(list(reversed(results)))]

        assert forward == backward == ["a", "b", "c"]


class TestWordPartition:
    def test_words_run_left_to_right_without_overlapping(self):
        (line,) = build_ocr_lines_from_results([_region("um dois tres quatro")])

        lefts = [word.bbox[0] for word in line.words]
        rights = [word.bbox[2] for word in line.words]
        assert lefts == sorted(lefts)
        for index in range(len(line.words) - 1):
            assert rights[index] <= lefts[index + 1]

    def test_the_first_word_starts_at_the_line(self):
        (line,) = build_ocr_lines_from_results([_region("um dois", left=40.0, right=340.0)])

        assert line.words[0].bbox[0] == pytest.approx(40.0)

    def test_the_last_word_ends_within_the_line(self):
        (line,) = build_ocr_lines_from_results([_region("um dois tres", left=0.0, right=300.0)])

        assert line.words[-1].bbox[2] <= 300.0 + 1.0

    def test_widths_and_gaps_account_for_the_whole_line(self):
        (line,) = build_ocr_lines_from_results([_region("um dois tres", left=0.0, right=300.0)])

        widths = sum(word.bbox[2] - word.bbox[0] for word in line.words)
        gaps = _WORD_GAP_PX * (len(line.words) - 1)
        assert widths + gaps == pytest.approx(300.0, rel=1e-6)

    def test_width_is_proportional_to_character_count(self):
        (line,) = build_ocr_lines_from_results([_region("aa aaaa", left=0.0, right=600.0)])

        first, second = (word.bbox[2] - word.bbox[0] for word in line.words)
        assert second == pytest.approx(2 * first, rel=1e-6)

    def test_every_word_spans_the_full_line_height(self):
        (line,) = build_ocr_lines_from_results([_region("um dois", top=10.0, bottom=34.0)])

        for word in line.words:
            assert (word.bbox[1], word.bbox[3]) == (10.0, 34.0)

    def test_confidence_is_carried_to_every_word(self):
        (line,) = build_ocr_lines_from_results([_region("um dois", confidence=0.77)])

        assert [word.confidence for word in line.words] == [0.77, 0.77]


class TestDegenerateInput:
    def test_blank_text_produces_no_line(self):
        assert build_ocr_lines_from_results([_region("   ")]) == []

    def test_an_empty_result_list_is_tolerated(self):
        assert build_ocr_lines_from_results([]) == []

    def test_a_region_without_a_box_still_yields_words(self):
        """Position is unknown, but the text must not be dropped."""
        result = OCRResult(text="sem caixa", box=[], confidence=0.5)

        (line,) = build_ocr_lines_from_results([result])

        assert [word.text for word in line.words] == ["sem", "caixa"]
        assert line.bbox == []

    def test_a_box_narrower_than_the_token_count_stays_positive(self):
        """Guards the max(1.0, ...) floor: no negative or zero-width words."""
        (line,) = build_ocr_lines_from_results([_region("a b c d e", left=0.0, right=2.0)])

        for word in line.words:
            assert word.bbox[2] - word.bbox[0] >= 1.0

    def test_text_without_spaces_becomes_one_word(self):
        """CJK arrives unsegmented; splitting it would invent boundaries."""
        (line,) = build_ocr_lines_from_results([_region("中文本测试")])

        assert len(line.words) == 1
        assert line.words[0].bbox == [0.0, 0.0, 300.0, 20.0]

    def test_a_malformed_point_is_ignored_rather_than_fatal(self):
        result = OCRResult(text="texto", box=[[0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])

        (line,) = build_ocr_lines_from_results([result])

        assert line.bbox == [0.0, 0.0, 10.0, 5.0]


class TestLineBoundingBox:
    def test_the_bbox_is_the_extent_of_the_quadrilateral(self):
        skewed = OCRResult(
            text="inclinado",
            box=[[10.0, 20.0], [110.0, 25.0], [108.0, 60.0], [8.0, 55.0]],
            confidence=0.9,
        )

        (line,) = build_ocr_lines_from_results([skewed])

        assert line.bbox == [8.0, 20.0, 110.0, 60.0]

    def test_the_source_is_marked_as_ocr(self):
        """auto_verified replaces some lines with native text; the two must
        stay distinguishable in the sidecar."""
        (line,) = build_ocr_lines_from_results([_region("texto")])

        assert line.source == "ocr"
