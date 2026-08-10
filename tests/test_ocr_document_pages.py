"""Every pipeline must record what it OCR'd into the canonical document.

``OcrDocument`` feeds the ``.bigocr.json`` sidecar, the structured TXT/Markdown
/ODT exports, the confidence the interface shows, and every per-region field in
a benchmark record. Only the image-only pipeline ever filled it, so a
mixed-content or embedded-image document reported a confidence of exactly zero,
exported through the ``pdftotext`` fallback instead of its own OCR geometry,
and was invisible to the zero-region regression gate.
"""

import pytest

from bigocrpdf.services.rapidocr_service.config import OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_document_pages import (
    average_confidence,
    record_ocr_page,
)


def _result(text: str, top: float = 0.0, confidence: float = 0.9) -> OCRResult:
    return OCRResult(
        text=text,
        box=[[10.0, top], [200.0, top], [200.0, top + 20.0], [10.0, top + 20.0]],
        confidence=confidence,
    )


class TestRecordOcrPage:
    def test_the_page_lands_in_the_document(self):
        stats = ProcessingStats()

        record_ocr_page(stats, 1, [_result("Relatorio")], (1275, 1755), 300)

        (page,) = stats.ocr_document.pages
        assert page.page_index == 1
        assert page.width_px == 1275
        assert page.height_px == 1755
        assert page.dpi == 300

    def test_regions_are_counted_once(self):
        """Counting lives with recording, so the two cannot run apart.

        They did: the mixed-content pipelines incremented the tally and never
        recorded the page.
        """
        stats = ProcessingStats()

        record_ocr_page(stats, 1, [_result("um"), _result("dois", top=40.0)], (100, 100), 300)

        assert stats.total_text_regions == 2
        assert len(stats.ocr_document.pages[0].text_results) == 2

    def test_structured_lines_are_built(self):
        """The exports read lines, not raw regions."""
        stats = ProcessingStats()

        record_ocr_page(stats, 1, [_result("Relatorio tecnico")], (100, 100), 300)

        (line,) = stats.ocr_document.pages[0].lines
        assert line.text == "Relatorio tecnico"
        assert [word.text for word in line.words] == ["Relatorio", "tecnico"]

    def test_a_page_without_text_is_marked_absent(self):
        """So the zero-region gate can tell a blank page from an unread one."""
        stats = ProcessingStats()

        record_ocr_page(stats, 3, [], (100, 100), 300)

        page = stats.ocr_document.pages[0]
        assert page.text_layer_quality == "absent"
        assert page.text_results == []
        assert stats.total_text_regions == 0

    def test_several_pages_accumulate(self):
        stats = ProcessingStats()

        record_ocr_page(stats, 1, [_result("a")], (100, 100), 300)
        record_ocr_page(stats, 2, [_result("b")], (100, 100), 300)

        assert [page.page_index for page in stats.ocr_document.pages] == [1, 2]

    def test_a_missing_dpi_falls_back_to_300(self):
        stats = ProcessingStats()

        record_ocr_page(stats, 1, [_result("a")], (100, 100), 0)

        assert stats.ocr_document.pages[0].dpi == 300

    def test_the_recorded_results_are_a_copy(self):
        """The caller often reuses its list for the next image on the page."""
        stats = ProcessingStats()
        results = [_result("a")]

        record_ocr_page(stats, 1, results, (100, 100), 300)
        results.clear()

        assert len(stats.ocr_document.pages[0].text_results) == 1


class TestAverageConfidence:
    def test_it_is_the_mean_over_every_region(self):
        stats = ProcessingStats()
        record_ocr_page(stats, 1, [_result("a", confidence=1.0)], (100, 100), 300)
        record_ocr_page(
            stats,
            2,
            [_result("b", confidence=0.8), _result("c", top=40.0, confidence=0.6)],
            (100, 100),
            300,
        )

        assert average_confidence(stats) == pytest.approx((1.0 + 0.8 + 0.6) / 3)

    def test_an_empty_document_is_zero_rather_than_an_error(self):
        assert average_confidence(ProcessingStats()) == 0.0

    def test_pages_without_regions_do_not_drag_the_mean_down(self):
        """A blank page has no confidence, which is not the same as zero."""
        stats = ProcessingStats()
        record_ocr_page(stats, 1, [_result("a", confidence=0.9)], (100, 100), 300)
        record_ocr_page(stats, 2, [], (100, 100), 300)

        assert average_confidence(stats) == pytest.approx(0.9)


class TestSkippedPageWarning:
    """A page left alone because it already has OCR must say so.

    Skipping is the right default -- re-OCRing over an existing layer doubles
    every word -- but it used to be silent. A run reported success with zero
    regions and no warning, so a user whose earlier OCR was poor had no signal
    that nothing had happened and no hint that ``replace existing OCR`` is the
    setting that redoes it.
    """

    def test_a_skipped_page_produces_a_warning(self):
        from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import (
            _record_skipped_page,
        )

        stats = ProcessingStats()

        _record_skipped_page(stats, 3)

        assert len(stats.warnings) == 1
        assert "3" in stats.warnings[0]

    def test_the_warning_names_the_setting_that_fixes_it(self):
        from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import (
            _record_skipped_page,
        )

        stats = ProcessingStats()

        _record_skipped_page(stats, 1)

        assert "replace existing OCR" in stats.warnings[0]

    def test_no_stats_object_is_tolerated(self):
        """The helper is also reachable from paths that do not carry stats."""
        from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import (
            _record_skipped_page,
        )

        _record_skipped_page(None, 1)
