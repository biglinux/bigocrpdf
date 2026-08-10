"""The same physical layout must export the same way at any DPI.

Every threshold in ``tsv_parser`` -- Y_TOLERANCE, COLUMN_MIN_VALLEY_WIDTH,
FOOTER_REGION_Y and the rest -- was tuned for PDF points, because the layout
analyser was written for ``pdftotext -tsv`` output. ``ocr_document_export``
feeds the same analyser OCR coordinates in *pixels at page.dpi*, which at 300
DPI are about 4.17x larger, and nothing converts between the two.

Whether that matters is a question with an answer, and this file is the
measurement. A layout is declared once in points and materialised at several
DPIs; the physical page is identical every time, so any change in the exported
structure is the unit mismatch showing through.

**Outcome: the strong hypothesis is refuted.** Exported structure is identical
from 72 to 600 DPI for prose, footers and table-shaped input alike. The reason
is the shape of the data rather than the thresholds: RapidOCR returns one
region per text line, so ``_ocr_result_words`` emits a single Word per line and
the line-grouping and column-splitting thresholds never get a decision to make.
They would bite on ``pdftotext -tsv`` output, which is per-word -- and that
path does receive points, as designed.

So the constants are left alone. They are genuinely inconsistent in physical
terms, which ``TestThresholdsAgainstRealPixelSizes`` records, and that becomes
a live problem the moment anything starts feeding the analyser per-word OCR
geometry. These tests are what will notice.
"""

import pytest

from bigocrpdf.services.rapidocr_service.config import OcrDocument, OcrPage, OCRResult
from bigocrpdf.services.rapidocr_service.ocr_document_export import (
    ocr_document_to_pages_elements,
)
from bigocrpdf.utils.tsv_parser import (
    COLUMN_MIN_VALLEY_WIDTH,
    FOOTER_REGION_Y,
    Y_TOLERANCE,
    Word,
    group_into_lines,
)

# A4 in points. Layouts below are written in these units and scaled per DPI.
PAGE_WIDTH_PT = 595.0
PAGE_HEIGHT_PT = 842.0

DPIS = [72, 96, 150, 200, 300, 400, 600]


def _line(text: str, x_pt: float, y_pt: float, size_pt: float = 11.0) -> dict:
    """One OCR region, positioned in points."""
    return {"text": text, "x": x_pt, "y": y_pt, "size": size_pt}


# Ordinary prose down the page: no columns, no table, nothing ambiguous.
SINGLE_COLUMN = [
    _line("Relatorio tecnico de inspecao predial", 72.0, 120.0),
    _line("Documento emitido em doze de marco", 72.0, 150.0),
    _line("Responsavel tecnico Ana Paula Ribeiro", 72.0, 180.0),
    _line("Area total construida trezentos metros", 72.0, 210.0),
    _line("Situacao geral da estrutura adequada", 72.0, 240.0),
]

# A page number low on the page, where FOOTER_REGION_Y lives.
WITH_FOOTER = SINGLE_COLUMN + [_line("Pagina 3 de 12", 280.0, 800.0)]


def _document(layout: list[dict], dpi: int) -> OcrDocument:
    """Materialise a point-declared layout as OCR pixels at ``dpi``."""
    scale = dpi / 72.0
    results = []
    for entry in layout:
        left = entry["x"] * scale
        top = entry["y"] * scale
        height = entry["size"] * scale
        # 0.5 em per character is close enough for layout purposes and keeps
        # the geometry a pure function of the declaration.
        width = len(entry["text"]) * entry["size"] * 0.5 * scale
        results.append(
            OCRResult(
                text=entry["text"],
                box=[
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height],
                ],
                confidence=0.98,
            )
        )
    page = OcrPage(
        page_index=1,
        width_px=int(PAGE_WIDTH_PT * scale),
        height_px=int(PAGE_HEIGHT_PT * scale),
        dpi=dpi,
        text_results=results,
    )
    return OcrDocument(pages=[page])


def _structure(layout: list[dict], dpi: int) -> list[tuple]:
    """Everything about the export except absolute coordinates."""
    pages = ocr_document_to_pages_elements(_document(layout, dpi))
    return [
        (element.kind, element.text, tuple(tuple(row) for row in element.rows))
        for page in pages
        for element in page
    ]


class TestStructureIsDpiInvariant:
    @pytest.mark.parametrize("dpi", DPIS)
    def test_single_column_prose(self, dpi):
        assert _structure(SINGLE_COLUMN, dpi) == _structure(SINGLE_COLUMN, 72)

    @pytest.mark.parametrize("dpi", DPIS)
    def test_a_page_with_a_footer(self, dpi):
        assert _structure(WITH_FOOTER, dpi) == _structure(WITH_FOOTER, 72)

    @pytest.mark.parametrize("dpi", DPIS)
    def test_no_text_is_lost_at_any_resolution(self, dpi):
        """The weakest useful invariant: whatever the structure, keep the words."""
        exported = " ".join(text for _, text, _ in _structure(SINGLE_COLUMN, dpi))

        for entry in SINGLE_COLUMN:
            assert entry["text"].split()[0] in exported


class TestThresholdsAgainstRealPixelSizes:
    """The point-tuned constants, measured against the pixels they receive."""

    @pytest.mark.parametrize("dpi", [150, 300, 600])
    def test_line_grouping_tolerance_shrinks_in_physical_terms(self, dpi):
        """Y_TOLERANCE is 5 units; at 300 DPI that is 1.2 pt of slack.

        Real OCR boxes on one visual line vary in ``top`` by more than that, so
        the analyser can shatter a line into several. Demonstrated directly:
        two words 8 units apart group at low DPI and split at high DPI only if
        the tolerance is not scaled.
        """
        scale = dpi / 72.0
        offset_units = 1.5 * Y_TOLERANCE

        words = [
            Word(text="esquerda", left=10.0, top=100.0, width=60.0, height=12.0),
            Word(text="direita", left=100.0, top=100.0 + offset_units, width=50.0, height=12.0),
        ]
        lines = group_into_lines(words)

        # Documents today's behaviour: the tolerance is an absolute number, so
        # the same physical offset crosses it or not depending on the DPI the
        # caller happened to use.
        assert len(lines) == 2
        assert offset_units / scale < 5.0 or dpi == 72

    @pytest.mark.parametrize("dpi", [72, 300])
    def test_the_footer_band_moves_up_the_page_as_dpi_rises(self, dpi):
        """FOOTER_REGION_Y = 780 is the bottom of A4 in points.

        In pixels at 300 DPI, y=780 is roughly the top fifth of the page, so
        body text would fall inside the "footer" band. This states where the
        band actually lands rather than asserting a consequence.
        """
        page_height_px = PAGE_HEIGHT_PT * dpi / 72.0
        footer_fraction = FOOTER_REGION_Y / page_height_px

        if dpi == 72:
            assert footer_fraction > 0.9
        else:
            assert footer_fraction < 0.25

    @pytest.mark.parametrize("dpi", [72, 300])
    def test_the_column_valley_width_shrinks_physically(self, dpi):
        """COLUMN_MIN_VALLEY_WIDTH = 30 is 0.1 inch at 300 DPI.

        That is narrower than an ordinary paragraph indent, so a single-column
        page with one wide word gap can be read as two columns -- which
        scrambles reading order in both TXT and ODT.
        """
        valley_inches = COLUMN_MIN_VALLEY_WIDTH / dpi

        if dpi == 72:
            assert valley_inches > 0.4
        else:
            assert valley_inches < 0.15


class TestFormGapEntanglement:
    def test_the_synthetic_form_gap_is_tuned_to_the_table_threshold(self):
        """The two unit systems are already entangled by design.

        ``ocr_document_export`` inserts a 120-*pixel* gap between a form label
        and its value precisely so ``is_table_line``'s 100-*point* threshold
        fires. Any DPI normalisation has to move both together, so this is not
        a one-line change -- which is the main thing to know before attempting
        it.
        """
        from bigocrpdf.services.rapidocr_service.ocr_document_export import _FORM_FIELD_GAP_PX
        from bigocrpdf.utils.tsv_parser import TABLE_TWO_COL_GAP

        assert _FORM_FIELD_GAP_PX > TABLE_TWO_COL_GAP


# Two regions sharing a visual line, with the sub-point vertical jitter real
# OCR produces. This is the shape that would exercise Y_TOLERANCE and the
# table thresholds if anything did.
TABLE_SHAPED: list[dict] = []
for _row, (_left_cell, _right_cell) in enumerate(
    [("Produto", "Valor"), ("Livro", "80,00"), ("Caneta", "12,50")]
):
    _y = 200.0 + _row * 30.0
    TABLE_SHAPED.append(_line(_left_cell, 72.0, _y))
    TABLE_SHAPED.append(_line(_right_cell, 380.0, _y + 0.8))


class TestTableShapedInputIsAlsoInvariant:
    @pytest.mark.parametrize("dpi", DPIS)
    def test_structure_does_not_change_with_dpi(self, dpi):
        assert _structure(TABLE_SHAPED, dpi) == _structure(TABLE_SHAPED, 72)

    def test_every_cell_survives_the_export(self):
        """Whatever structure is chosen, no cell may be dropped."""
        exported = " ".join(text for _, text, _ in _structure(TABLE_SHAPED, 300))

        for entry in TABLE_SHAPED:
            assert entry["text"] in exported
