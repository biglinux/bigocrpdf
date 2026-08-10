"""TXT and ODT export, driven from a real OCR text layer.

This is where the two halves of the project meet. ``export_service`` runs
``pdftotext`` over the *finished* searchable PDF, so the quality of every TXT
and ODT a user gets is a direct function of whether the invisible text layer
landed on the right glyphs -- the property tests/test_text_layer_placement
establishes. Here that layer is built by the real renderer and then exported.

``save_odf_file`` had no test at all before this file.
"""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from bigocrpdf.services.export_service import save_odf_file, save_text_file
from tests.positional_oracle import (
    TruthWord,
    by_token,
    extract_words,
    render_layer_pdf,
    requires_pdftotext,
    truth_to_metric_quads,
)

pytestmark = requires_pdftotext

PAGE = (612.0, 792.0)
IMAGE = (2550, 3300)

# Plain prose down the page: unambiguous reading order, so any scrambling in
# the export is the export's doing.
LINES = [
    "Relatorio tecnico de inspecao predial",
    "Documento emitido em doze de marco",
    "Responsavel tecnico Ana Paula Ribeiro",
    "Area total construida trezentos metros",
    "Situacao geral da estrutura adequada",
]


@pytest.fixture
def searchable_pdf(tmp_path: Path) -> Path:
    """A PDF whose only content is a real, correctly-placed OCR text layer."""
    words = [
        TruthWord(token=line, x=72.0, baseline=700.0 - index * 40.0, size=12.0)
        for index, line in enumerate(LINES)
    ]
    return render_layer_pdf(
        truth_to_metric_quads(words, PAGE, IMAGE),
        IMAGE,
        PAGE,
        tmp_path / "searchable.pdf",
    )


class TestSaveTextFile:
    def test_it_writes_a_txt_beside_the_pdf(self, searchable_pdf: Path):
        result = save_text_file(str(searchable_pdf), extracted_text="")

        assert result is not None
        assert Path(result).suffix == ".txt"
        assert Path(result).parent == searchable_pdf.parent

    def test_the_text_comes_from_the_layer_not_the_fallback(self, searchable_pdf: Path):
        """The raw argument is empty, so anything present was extracted."""
        result = save_text_file(str(searchable_pdf), extracted_text="")

        content = Path(result).read_text(encoding="utf-8")
        for line in LINES:
            assert line in content

    def test_reading_order_is_preserved(self, searchable_pdf: Path):
        result = save_text_file(str(searchable_pdf), extracted_text="")

        content = Path(result).read_text(encoding="utf-8")
        positions = [content.index(line) for line in LINES]
        assert positions == sorted(positions)

    def test_a_separate_folder_is_honoured(self, searchable_pdf: Path, tmp_path: Path):
        destination = tmp_path / "exports"
        destination.mkdir()

        result = save_text_file(
            str(searchable_pdf), extracted_text="", separate_folder=str(destination)
        )

        assert Path(result).parent == destination

    def test_a_failing_extraction_falls_back_to_the_raw_text(
        self, searchable_pdf: Path, monkeypatch
    ):
        """The user gets something rather than an error."""
        monkeypatch.setattr(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("pdftotext missing")),
        )

        result = save_text_file(str(searchable_pdf), extracted_text="texto bruto")

        assert result is not None
        assert "texto bruto" in Path(result).read_text(encoding="utf-8")


class TestSaveOdfFile:
    """The reflowable and fixed-layout ODT paths, neither previously tested."""

    @pytest.mark.parametrize("include_images", [False, True])
    def test_it_writes_a_readable_odt(self, searchable_pdf: Path, include_images: bool):
        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
            include_images=include_images,
        )

        assert result is not None
        assert Path(result).suffix == ".odt"
        with zipfile.ZipFile(result) as archive:
            assert "content.xml" in archive.namelist()
            assert "META-INF/manifest.xml" in archive.namelist()

    @pytest.mark.parametrize("include_images", [False, True])
    def test_every_line_survives_into_content_xml(self, searchable_pdf: Path, include_images: bool):
        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
            include_images=include_images,
        )

        with zipfile.ZipFile(result) as archive:
            content = archive.read("content.xml").decode("utf-8")
        for line in LINES:
            assert line in content

    def test_reading_order_is_preserved_in_the_document(self, searchable_pdf: Path):
        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
            include_images=False,
        )

        with zipfile.ZipFile(result) as archive:
            content = archive.read("content.xml").decode("utf-8")
        positions = [content.index(line) for line in LINES]
        assert positions == sorted(positions)

    def test_the_fixed_layout_variant_anchors_text_in_frames(self, searchable_pdf: Path):
        """include_images=True positions each line, rather than reflowing it."""
        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
            include_images=True,
        )

        with zipfile.ZipFile(result) as archive:
            content = archive.read("content.xml").decode("utf-8")
        assert "draw:frame" in content

    def test_an_unwritable_destination_reports_failure_without_raising(
        self, searchable_pdf: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_odf",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only filesystem")),
        )

        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
        )

        assert result is None

    def test_an_unexpected_error_is_contained(self, searchable_pdf: Path, monkeypatch):
        """A broken converter must not take the whole export down."""
        monkeypatch.setattr(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_odf",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        result = save_odf_file(
            str(searchable_pdf),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(searchable_pdf),
        )

        assert result is None


class TestOdtDocumentStructure:
    """A heading must be a heading, not a paragraph that looks like one.

    The layout analyser identified 29 headings in an eighteen-page contract,
    and the ODT builder emitted every one of them as ``text:p``. The document
    opened as a flat wall of paragraphs: no outline in the navigator, no
    automatic table of contents, and nothing for a screen reader to navigate
    by. Only ``text:h`` with an outline level provides those.
    """

    @staticmethod
    def _content_xml(odt_path: str) -> str:
        with zipfile.ZipFile(odt_path) as archive:
            return archive.read("content.xml").decode("utf-8")

    @pytest.fixture
    def structured_odt(self, tmp_path: Path) -> str:
        """A page whose headings the analyser recognises: short and upper case."""
        lines = [
            "CLAUSULA PRIMEIRA DO OBJETO",
            "O presente contrato tem por objeto a prestacao de servicos",
            "tecnicos especializados descritos no anexo do instrumento.",
            "CLAUSULA SEGUNDA DO PRAZO",
            "O prazo de vigencia sera de doze meses contados da assinatura.",
        ]
        words = [
            TruthWord(token=line, x=72.0, baseline=700.0 - index * 40.0, size=12.0)
            for index, line in enumerate(lines)
        ]
        pdf = render_layer_pdf(
            truth_to_metric_quads(words, PAGE, IMAGE), IMAGE, PAGE, tmp_path / "estruturado.pdf"
        )
        result = save_odf_file(
            str(pdf), extracted_text="", ocr_boxes=[], source_pdf=str(pdf), include_images=False
        )
        assert result is not None
        return result

    def test_headings_are_emitted_as_headings(self, structured_odt: str):
        content = self._content_xml(structured_odt)

        assert "<text:h" in content

    def test_headings_carry_an_outline_level(self, structured_odt: str):
        """Without it there is no outline, whatever the styling says."""
        import re

        content = self._content_xml(structured_odt)
        levels = re.findall(r'<text:h[^>]*text:outline-level="(\d)"', content)

        assert levels
        assert all(1 <= int(level) <= 3 for level in levels)

    def test_body_text_stays_paragraphs(self, structured_odt: str):
        """Promoting prose to headings would be as wrong as the reverse."""
        content = self._content_xml(structured_odt)

        assert "<text:p" in content
        assert "prestacao de servicos" in content

    def test_no_text_is_lost_to_the_structure_change(self, structured_odt: str):
        import re

        text = re.sub(r"<[^>]+>", " ", self._content_xml(structured_odt))

        for fragment in ("CLAUSULA PRIMEIRA", "prazo de vigencia", "anexo do instrumento"):
            assert fragment in text


@pytest.mark.slow
class TestFixedLayoutOdtPlacesTextWhereItWas:
    """The fixed-layout export promises position, so position is measured.

    A page used to be stretched to A4's 29.7 cm on its longer side whatever it
    measured. A US Letter document therefore came out 6.3% too large and every
    word landed 6.3% away from where it belongs -- 42 points adrift by the foot
    of the page, which is more than half an inch on paper.

    LibreOffice renders the ODT back to PDF and the words are compared against
    the page they came from, so the whole chain is measured rather than the
    XML we happened to write.
    """

    TOKENS = [
        ("TITULOCENTRAL", 240.0, 720.0),
        ("ESQUERDAUM", 72.0, 660.0),
        ("DIREITAUM", 380.0, 660.0),
        ("CORPOPRIMEIRO", 72.0, 560.0),
        ("RODAPEFINAL", 72.0, 120.0),
    ]
    # The frame carries a little internal padding above its text, which is a
    # property of the format, not a placement error.
    TOLERANCE_PT = 2.0

    @pytest.fixture
    def rendered_pair(self, tmp_path: Path):
        libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
        if libreoffice is None:
            pytest.skip("LibreOffice is not installed")

        words = [TruthWord(token=token, x=x, baseline=y, size=12.0) for token, x, y in self.TOKENS]
        source = render_layer_pdf(
            truth_to_metric_quads(words, PAGE, IMAGE), IMAGE, PAGE, tmp_path / "fonte.pdf"
        )
        odt = save_odf_file(
            str(source),
            extracted_text="",
            ocr_boxes=[],
            source_pdf=str(source),
            include_images=True,
        )
        assert odt is not None
        subprocess.run(
            [libreoffice, "--headless", "--convert-to", "pdf", odt, "--outdir", str(tmp_path)],
            capture_output=True,
            timeout=300,
            check=False,
        )
        rendered = Path(odt).with_suffix(".pdf")
        if not rendered.exists():
            pytest.skip("LibreOffice produced no output")
        return by_token(extract_words(source)), by_token(extract_words(rendered))

    def test_every_word_survives_the_round_trip(self, rendered_pair):
        source_words, rendered_words = rendered_pair

        for token, _x, _y in self.TOKENS:
            assert token in rendered_words, f"{token} was lost between PDF and ODT"
            assert token in source_words

    def test_words_land_where_they_started(self, rendered_pair):
        source_words, rendered_words = rendered_pair

        failures = []
        for token, _x, _y in self.TOKENS:
            source, rendered = source_words[token], rendered_words[token]
            delta_x = rendered.x0 - source.x0
            delta_y = rendered.y_top - source.y_top
            if abs(delta_x) > self.TOLERANCE_PT or abs(delta_y) > self.TOLERANCE_PT:
                failures.append(f"{token}: dx={delta_x:+.1f}pt dy={delta_y:+.1f}pt")
        assert not failures, "fixed-layout export moved the text:\n  " + "\n  ".join(failures)

    def test_the_error_does_not_grow_down_the_page(self, rendered_pair):
        """A page-size mistake shows up as drift proportional to position.

        The header was 4 points out and the footer 42 -- the signature of a
        scale error rather than a constant offset, and what makes this test
        worth having separately from the tolerance above.
        """
        source_words, rendered_words = rendered_pair

        top = abs(rendered_words["TITULOCENTRAL"].y_top - source_words["TITULOCENTRAL"].y_top)
        bottom = abs(rendered_words["RODAPEFINAL"].y_top - source_words["RODAPEFINAL"].y_top)

        assert bottom - top < 1.0, f"error grows down the page: top={top:.1f} bottom={bottom:.1f}"
