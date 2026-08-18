import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pikepdf
import pytest
from PIL import Image, ImageOps

from bigocrpdf.services.rapidocr_service.config import OcrDocument, OcrPage
from bigocrpdf.services.rapidocr_service.ocr_document_io import (
    load_ocr_document_json,
    ocr_document_json_path,
    write_ocr_document_json,
)
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.page_operations import (
    _page_flip_matrix,
    apply_changes_to_pdf,
    apply_changes_to_pdf_atomically,
)


def _render_first_page(pdf_path: Path, output_prefix: Path) -> Image.Image:
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm is required for rendered PDF regression tests")
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            "72",
            "-cropbox",
            "-singlefile",
            "-png",
            str(pdf_path),
            str(output_prefix),
        ],
        check=True,
        capture_output=True,
    )
    with Image.open(output_prefix.with_suffix(".png")) as rendered:
        return rendered.convert("RGB")


@pytest.mark.parametrize(
    ("flip_horizontal", "flip_vertical"),
    ((True, False), (False, True)),
)
def test_saved_flip_matches_display_axis_after_rotation_and_preserves_vectors(
    tmp_path: Path,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> None:
    source_pdf = tmp_path / "quadrants.pdf"
    baseline_pdf = tmp_path / "baseline.pdf"
    flipped_pdf = tmp_path / "flipped.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(100, 60))
        page.contents_add(
            b"1 0 0 rg 0 0 50 30 re f "
            b"0 1 0 rg 50 0 50 30 re f "
            b"0 0 1 rg 0 30 50 30 re f "
            b"1 1 0 rg 50 30 50 30 re f"
        )
        pdf.save(source_pdf)

    baseline = PDFDocument(path=str(source_pdf), total_pages=1)
    baseline.pages[0].rotation = 90
    flipped = PDFDocument(path=str(source_pdf), total_pages=1)
    flipped.pages[0].rotation = 90
    flipped.pages[0].flip_horizontal = flip_horizontal
    flipped.pages[0].flip_vertical = flip_vertical

    assert apply_changes_to_pdf(baseline, str(baseline_pdf))
    assert apply_changes_to_pdf(flipped, str(flipped_pdf))

    baseline_image = _render_first_page(baseline_pdf, tmp_path / "baseline-render")
    flipped_image = _render_first_page(flipped_pdf, tmp_path / "flipped-render")
    expected = ImageOps.mirror(baseline_image) if flip_horizontal else ImageOps.flip(baseline_image)
    assert flipped_image.size == expected.size
    width, height = expected.size
    for x_fraction, y_fraction in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        point = (int(width * x_fraction), int(height * y_fraction))
        actual_pixel = flipped_image.getpixel(point)
        expected_pixel = expected.getpixel(point)
        assert isinstance(actual_pixel, tuple)
        assert isinstance(expected_pixel, tuple)
        assert max(abs(a - b) for a, b in zip(actual_pixel, expected_pixel, strict=True)) <= 5

    with pikepdf.open(flipped_pdf) as saved:
        operators = {
            str(instruction.operator)
            for instruction in pikepdf.parse_content_stream(saved.pages[0])
        }
        assert {"cm", "re"} <= operators


@pytest.mark.parametrize(
    ("rotation", "horizontal", "vertical", "expected"),
    (
        (0, True, False, (-1.0, 0.0, 0.0, 1.0, 120.0, 0.0)),
        (90, True, False, (1.0, 0.0, 0.0, -1.0, 0.0, 100.0)),
        (180, True, False, (-1.0, 0.0, 0.0, 1.0, 120.0, 0.0)),
        (270, True, False, (1.0, 0.0, 0.0, -1.0, 0.0, 100.0)),
        (0, False, True, (1.0, 0.0, 0.0, -1.0, 0.0, 100.0)),
        (90, False, True, (-1.0, 0.0, 0.0, 1.0, 120.0, 0.0)),
        (180, False, True, (1.0, 0.0, 0.0, -1.0, 0.0, 100.0)),
        (270, False, True, (-1.0, 0.0, 0.0, 1.0, 120.0, 0.0)),
    ),
)
def test_page_flip_matrix_uses_displaced_page_box(
    rotation: int,
    horizontal: bool,
    vertical: bool,
    expected: tuple[float, ...],
) -> None:
    assert _page_flip_matrix(
        [10, 20, 110, 80],
        rotation,
        horizontal=horizontal,
        vertical=vertical,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("flip_horizontal", "flip_vertical"),
    ((True, False), (False, True)),
)
def test_saved_flip_uses_cropbox_and_preserves_page_boxes(
    tmp_path: Path,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> None:
    source_pdf = tmp_path / "cropped.pdf"
    baseline_pdf = tmp_path / "cropped-baseline.pdf"
    flipped_pdf = tmp_path / "cropped-flipped.pdf"
    media_box = [0, 0, 200, 100]
    crop_box = [20, 10, 120, 70]
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(200, 100))
        page.CropBox = pikepdf.Array(crop_box)
        page.contents_add(
            b"1 0 0 rg 20 10 50 30 re f "
            b"0 1 0 rg 70 10 50 30 re f "
            b"0 0 1 rg 20 40 50 30 re f "
            b"1 1 0 rg 70 40 50 30 re f"
        )
        pdf.save(source_pdf)

    baseline = PDFDocument(path=str(source_pdf), total_pages=1)
    flipped = PDFDocument(path=str(source_pdf), total_pages=1)
    flipped.pages[0].flip_horizontal = flip_horizontal
    flipped.pages[0].flip_vertical = flip_vertical

    assert apply_changes_to_pdf(baseline, str(baseline_pdf))
    assert apply_changes_to_pdf(flipped, str(flipped_pdf))

    baseline_image = _render_first_page(baseline_pdf, tmp_path / "crop-baseline-render")
    flipped_image = _render_first_page(flipped_pdf, tmp_path / "crop-flipped-render")
    expected = ImageOps.mirror(baseline_image) if flip_horizontal else ImageOps.flip(baseline_image)
    assert flipped_image.size == expected.size
    assert flipped_image.tobytes() == expected.tobytes()

    with pikepdf.open(flipped_pdf) as saved:
        assert [float(value) for value in saved.pages[0].MediaBox] == media_box
        assert [float(value) for value in saved.pages[0].CropBox] == crop_box


def _add_asymmetric_stamp(
    pdf: pikepdf.Pdf,
    page,
    *,
    rect: tuple[int, int, int, int],
    appearance,
) -> None:
    annotation = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Annot"),
                "/Subtype": pikepdf.Name("/Stamp"),
                "/Rect": pikepdf.Array(rect),
                "/F": 4,
                "/AP": pikepdf.Dictionary({"/N": appearance}),
            }
        )
    )
    if "/Annots" not in page:
        page.Annots = pikepdf.Array()
    page.Annots.append(annotation)


def test_saved_flip_mirrors_annotation_appearance_not_only_its_rect(tmp_path: Path) -> None:
    source_pdf = tmp_path / "stamp.pdf"
    baseline_pdf = tmp_path / "stamp-baseline.pdf"
    flipped_pdf = tmp_path / "stamp-flipped.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(100, 60))
        appearance = pdf.make_stream(b"1 0 0 rg 0 0 15 20 re f 0 0 1 rg 15 0 25 20 re f")
        appearance.Type = pikepdf.Name("/XObject")
        appearance.Subtype = pikepdf.Name("/Form")
        appearance.BBox = pikepdf.Array([0, 0, 40, 20])
        appearance.Matrix = pikepdf.Array([0, 1, -1, 0, 20, 0])
        _add_asymmetric_stamp(pdf, page, rect=(10, 10, 90, 50), appearance=appearance)
        pdf.save(source_pdf)

    baseline = PDFDocument(path=str(source_pdf), total_pages=1)
    flipped = PDFDocument(path=str(source_pdf), total_pages=1)
    flipped.pages[0].flip_horizontal = True

    assert apply_changes_to_pdf(baseline, str(baseline_pdf))
    assert apply_changes_to_pdf(flipped, str(flipped_pdf))

    baseline_image = _render_first_page(baseline_pdf, tmp_path / "stamp-baseline-render")
    flipped_image = _render_first_page(flipped_pdf, tmp_path / "stamp-flipped-render")
    expected = ImageOps.mirror(baseline_image)
    assert flipped_image.size == expected.size
    assert flipped_image.tobytes() == expected.tobytes()


def test_annotation_state_appearances_are_wrapped_without_mutating_shared_stream(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "shared-appearance.pdf"
    output_pdf = tmp_path / "shared-appearance-flipped.pdf"
    original_bytes = b"1 0 0 rg 0 0 20 10 re f"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(100, 60))
        appearance = pdf.make_stream(original_bytes)
        appearance.Type = pikepdf.Name("/XObject")
        appearance.Subtype = pikepdf.Name("/Form")
        appearance.BBox = pikepdf.Array([0, 0, 20, 10])
        states = pikepdf.Dictionary({"/On": appearance, "/Off": appearance})
        shared_ap = pikepdf.Dictionary(
            {
                "/N": states,
                "/R": appearance,
                "/D": appearance,
            }
        )
        for rect in ((5, 5, 45, 25), (55, 35, 95, 55)):
            annotation = pdf.make_indirect(
                pikepdf.Dictionary(
                    {
                        "/Type": pikepdf.Name("/Annot"),
                        "/Subtype": pikepdf.Name("/Stamp"),
                        "/Rect": pikepdf.Array(rect),
                        "/AP": shared_ap,
                    }
                )
            )
            page.Annots = page.get("/Annots", pikepdf.Array())
            page.Annots.append(annotation)
        pdf.save(source_pdf)

    document = PDFDocument(path=str(source_pdf), total_pages=1)
    document.pages[0].flip_vertical = True
    assert apply_changes_to_pdf(document, str(output_pdf))

    with pikepdf.open(output_pdf) as saved:
        annotations = list(saved.pages[0].Annots)
        wrappers = []
        originals = []
        for annotation in annotations:
            for appearance in (
                annotation.AP.N.On,
                annotation.AP.N.Off,
                annotation.AP.R,
                annotation.AP.D,
            ):
                wrappers.append(appearance.objgen)
                original = appearance.Resources.XObject.Original
                originals.append(original.objgen)
                assert original.read_bytes() == original_bytes
                assert b"/Original Do" in appearance.read_bytes()
        assert len(set(wrappers)) == len(wrappers)
        assert len(set(originals)) == 1


def test_apply_changes_merges_pdf_and_image_pages(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_image = tmp_path / "source.png"
    output_pdf = tmp_path / "merged.pdf"

    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(200, 300))
        pdf.save(source_pdf)
    Image.new("RGBA", (80, 40), (255, 255, 255, 128)).save(source_image)

    document = PDFDocument(path=str(source_pdf), total_pages=0)
    document.pages = [
        PageState(page_number=1, position=0, source_file=str(source_pdf)),
        PageState(page_number=1, position=1, source_file=str(source_image), rotation=90),
    ]
    document.total_pages = len(document.pages)

    assert apply_changes_to_pdf(document, str(output_pdf))

    with pikepdf.open(output_pdf) as merged:
        assert len(merged.pages) == 2
        assert list(merged.pages[0].MediaBox) == [0, 0, 200, 300]
        assert list(merged.pages[1].MediaBox) == [0, 0, 80, 40]
        assert int(merged.pages[1].Rotate) == 90


def test_materialized_pdf_omits_pages_excluded_from_the_final_document(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "edited.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.add_blank_page()
        pdf.save(source_pdf)
    document = PDFDocument(path=str(source_pdf), total_pages=2)
    document.pages[1].included_for_ocr = False

    assert apply_changes_to_pdf(document, str(output_pdf))

    with pikepdf.open(output_pdf) as edited:
        assert len(edited.pages) == 1


def test_atomic_apply_preserves_existing_destination_on_publication_failure(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "edited.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source_pdf)
    output_pdf.write_bytes(b"existing destination")
    document = PDFDocument(path=str(source_pdf), total_pages=1)

    with (
        patch(
            "bigocrpdf.utils.durable_writes.publish_file_atomically",
            side_effect=OSError("simulated publication failure"),
        ),
    ):
        saved = apply_changes_to_pdf_atomically(document, output_pdf)

    assert saved is False
    assert output_pdf.read_bytes() == b"existing destination"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "edited.pdf",
        "source.pdf",
    ]


def test_structured_json_is_refused_after_the_pdf_it_describes_is_edited(
    tmp_path: Path,
) -> None:
    """Editing no longer rewrites a companion file, so the reader must refuse.

    Structured OCR is bound to the PDF by SHA-256. Once the editor republishes
    the PDF, a JSON exported earlier describes a document that no longer
    exists, and loading it returns nothing instead of stale text.
    """
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "edited.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source_pdf)
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.add_blank_page()
        pdf.save(output_pdf)
    json_path = ocr_document_json_path(output_pdf)
    two_pages = OcrDocument(
        pages=[OcrPage(1, 100, 100, 300), OcrPage(2, 100, 100, 300)],
    )
    write_ocr_document_json(two_pages, output_pdf, json_path)
    assert load_ocr_document_json(json_path, output_pdf) is not None

    document = PDFDocument(path=str(source_pdf), total_pages=1)
    assert apply_changes_to_pdf_atomically(document, output_pdf)

    assert load_ocr_document_json(json_path, output_pdf) is None


@pytest.mark.parametrize("invalid_source", ["missing", "invalid_page", "no_source"])
def test_apply_changes_fails_instead_of_saving_partial_output(
    tmp_path: Path, invalid_source: str
) -> None:
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "partial.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source_pdf)

    bad_page = PageState(
        page_number=1,
        position=1,
        source_file=str(tmp_path / "missing.pdf"),
    )
    if invalid_source == "invalid_page":
        bad_page.page_number = 2
        bad_page.source_file = str(source_pdf)
    elif invalid_source == "no_source":
        bad_page.source_file = ""

    document = PDFDocument(
        path="" if invalid_source == "no_source" else str(source_pdf),
        total_pages=0,
    )
    document.pages = [
        PageState(page_number=1, position=0, source_file=str(source_pdf)),
        bad_page,
    ]
    document.total_pages = len(document.pages)

    assert not apply_changes_to_pdf(document, str(output_pdf))
    assert not output_pdf.exists()
