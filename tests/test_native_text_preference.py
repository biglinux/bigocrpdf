"""Keeping every character a mixed-content page has to offer.

A page that keeps its native text and gets its images OCR'd on top holds the
same words twice wherever the image repeats the page. Measured across two real
documents, 21% of OCR regions sat on native text.

The native reading is the authoritative one -- it is exact, where OCR truncates
long URLs and mangles rare characters -- so it wins. Three things had to hold
for that to be an improvement rather than a trade:

* substitution never removes a region (``verify_ocr_results_with_native_spans``)
* the export must not discard a narrow box that holds real text (``filter_words``)
* a colon inside a URL or a clock time is not a form separator

Each was found by measuring what reached the export, not by inspection.
"""

import pytest

from bigocrpdf.services.rapidocr_service.config import OCRResult
from bigocrpdf.services.rapidocr_service.native_text_verification import (
    NativeTextSpan,
    verify_ocr_results_with_native_spans,
)
from bigocrpdf.services.rapidocr_service.ocr_document_export import _split_form_tokens
from bigocrpdf.utils.tsv_parser import MAX_ARTIFACT_TEXT_LEN, Word, filter_words

URL = "https://assinador-web.onr.org.br/docs/UB7MR-ZF2N3-NTFLP-JMF2B"


def _result(text: str, bbox=(10.0, 10.0, 300.0, 40.0), confidence: float = 0.8) -> OCRResult:
    left, top, right, bottom = bbox
    return OCRResult(
        text=text,
        box=[[left, top], [right, top], [right, bottom], [left, bottom]],
        confidence=confidence,
    )


class TestNativeSubstitution:
    def test_a_truncated_url_is_restored_from_the_native_layer(self):
        """The measured case: OCR read the address short, the page had it whole."""
        results = [_result(URL[:-12])]
        spans = [NativeTextSpan(text=URL, bbox=[10.0, 10.0, 300.0, 40.0])]

        verified, accepted = verify_ocr_results_with_native_spans(results, spans)

        assert accepted == 1
        assert verified[0].text == URL

    def test_a_region_without_native_text_is_untouched(self):
        """Scanned content that the page never stated must survive verbatim."""
        results = [_result("texto que so existe no scan")]

        verified, accepted = verify_ocr_results_with_native_spans(results, [])

        assert accepted == 0
        assert verified[0].text == "texto que so existe no scan"

    def test_disagreeing_native_text_is_rejected(self):
        """Different words at the same place mean the overlap is coincidental."""
        results = [_result("CONSELHO REGIONAL DE ENGENHARIA")]
        spans = [NativeTextSpan(text="pagina 3 de 12", bbox=[10.0, 10.0, 300.0, 40.0])]

        verified, accepted = verify_ocr_results_with_native_spans(results, spans)

        assert accepted == 0
        assert verified[0].text == "CONSELHO REGIONAL DE ENGENHARIA"

    def test_no_region_is_ever_dropped(self):
        """The guarantee that makes this safe: substitution only."""
        results = [
            _result(f"regiao {index}", bbox=(10.0, index * 50.0, 300.0, index * 50.0 + 40))
            for index in range(6)
        ]

        verified, _accepted = verify_ocr_results_with_native_spans(results, [])

        assert len(verified) == len(results)

    def test_geometry_and_confidence_are_preserved(self):
        """Only the text changes; the box still describes where it sits."""
        original = _result(URL[:-12], confidence=0.61)
        spans = [NativeTextSpan(text=URL, bbox=[10.0, 10.0, 300.0, 40.0])]

        verified, _ = verify_ocr_results_with_native_spans([original], spans)

        assert verified[0].box == original.box
        assert verified[0].confidence == 0.61

    @pytest.mark.parametrize(
        "box",
        [pytest.param([], id="empty"), pytest.param([[1]], id="short-point")],
    )
    def test_a_malformed_box_keeps_its_region(self, box):
        result = OCRResult(text="texto", box=box, confidence=0.9)

        verified, accepted = verify_ocr_results_with_native_spans([result], [])

        assert (len(verified), accepted) == (1, 0)


class TestFilterWordsKeepsRealText:
    def test_a_long_word_in_a_sliver_box_survives(self):
        """Vertical captions arrive with a genuinely narrow box.

        The URL above reached the export 1.8 pixels wide because the export
        lays words out horizontally; width alone would have deleted it.
        """
        words = [Word(text=URL, left=246.0, top=658.0, width=1.8, height=1440.0)]

        assert filter_words(words, 1) == words

    def test_short_noise_in_a_sliver_box_is_still_removed(self):
        """A traced rule or hatch pattern reads as a couple of characters."""
        words = [Word(text="il", left=100.0, top=200.0, width=0.4, height=30.0)]

        assert filter_words(words, 1) == []

    def test_the_boundary_is_the_text_length(self):
        noise = Word(text="x" * MAX_ARTIFACT_TEXT_LEN, left=0.0, top=0.0, width=0.5, height=10.0)
        content = Word(
            text="x" * (MAX_ARTIFACT_TEXT_LEN + 1), left=0.0, top=0.0, width=0.5, height=10.0
        )

        assert filter_words([noise, content], 1) == [content]

    def test_ordinary_words_are_untouched(self):
        words = [Word(text="Relatorio", left=72.0, top=100.0, width=90.0, height=12.0)]

        assert filter_words(words, 1) == words


class TestFormSeparatorLeavesUrlsAndTimesWhole:
    def test_a_url_is_not_split_at_its_scheme(self):
        assert [token.text for token in _split_form_tokens(URL)] == [URL]

    def test_a_clock_time_is_not_split(self):
        assert [token.text for token in _split_form_tokens("23:32")] == ["23:32"]

    def test_a_seconds_time_is_not_split(self):
        assert [token.text for token in _split_form_tokens("08:15:42")] == ["08:15:42"]

    def test_a_form_field_is_still_split(self):
        """The behaviour this separator exists for must survive the fix."""
        assert [token.text for token in _split_form_tokens("Nome:Ana")] == ["Nome:", "Ana"]

    def test_a_label_before_a_time_still_splits(self):
        tokens = [token.text for token in _split_form_tokens("Hora:23:32")]

        assert tokens == ["Hora:", "23:32"]

    def test_a_scheme_without_slashes_still_splits(self):
        """mailto: is a label-like prefix; the address is the value."""
        assert [token.text for token in _split_form_tokens("mailto:a@b.c")] == ["mailto:", "a@b.c"]
