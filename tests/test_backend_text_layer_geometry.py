"""Geometry contracts for the OCR text-layer metadata pass."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pikepdf
import pytest
from PIL import Image

from bigocrpdf.services.ocr import backend_pipeline, pdf_page_geometry
from bigocrpdf.services.ocr import backend_text_layer_geometry as geometry
from bigocrpdf.services.ocr.config import ProcessingStats
from bigocrpdf.services.ocr.pdf_page_geometry import render_pdf_page_to_ppm
from bigocrpdf.services.ocr.rotation import PageRotation


class _MetadataBackend(backend_pipeline.BackendPipelineMixin):
    config: Any


def _add_image_page(
    pdf: pikepdf.Pdf,
    rect: tuple[int, int, int, int],
) -> None:
    page = pdf.add_blank_page(page_size=(200, 100))
    image = pdf.make_stream(b"\x00" * 100)
    image["/Type"] = pikepdf.Name.XObject
    image["/Subtype"] = pikepdf.Name.Image
    image["/Width"] = 10
    image["/Height"] = 10
    image["/ColorSpace"] = pikepdf.Name.DeviceGray
    image["/BitsPerComponent"] = 8
    page["/Resources"] = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary({"/Im0": image})})
    x, y, width, height = rect
    page["/Contents"] = pdf.make_stream(f"q {width} 0 0 {height} {x} {y} cm /Im0 Do Q".encode())


def _write_geometry_pdf(path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    _add_image_page(pdf, (10, 10, 180, 80))
    _add_image_page(pdf, (20, 20, 100, 60))
    pdf.add_blank_page(page_size=(200, 100))
    pdf.save(path)


def test_rectangles_are_read_in_page_order_with_one_pdf_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "geometry.pdf"
    _write_geometry_pdf(pdf_path)
    real_open = geometry.pikepdf.open
    open_count = 0

    def counted_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal open_count
        open_count += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(geometry.pikepdf, "open", counted_open)

    placements = geometry._extract_image_placements_from_pdf(pdf_path, page_count=3)

    assert open_count == 1
    assert placements == [
        ((10.0, 10.0, 180.0, 80.0), (False, 0)),
        ((20.0, 20.0, 100.0, 60.0), (False, 0)),
        None,
    ]


def test_bulk_rectangles_preserve_none_results_when_pdf_cannot_open(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    assert geometry._extract_image_placements_from_pdf(missing_pdf, page_count=3) == [
        None,
        None,
        None,
    ]


def test_reused_image_uses_largest_drawn_rectangle(tmp_path: Path) -> None:
    pdf_path = tmp_path / "reused-image.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(200, 100))
    image = pdf.make_stream(b"\x00" * 100)
    image["/Type"] = pikepdf.Name.XObject
    image["/Subtype"] = pikepdf.Name.Image
    image["/Width"] = 10
    image["/Height"] = 10
    image["/ColorSpace"] = pikepdf.Name.DeviceGray
    image["/BitsPerComponent"] = 8
    page["/Resources"] = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary({"/Im0": image})})
    page["/Contents"] = pdf.make_stream(
        b"q 20 0 0 20 0 0 cm /Im0 Do Q q 180 0 0 80 10 10 cm /Im0 Do Q"
    )
    pdf.save(pdf_path)

    assert geometry._extract_image_placements_from_pdf(pdf_path) == [
        ((10.0, 10.0, 180.0, 80.0), (False, 0))
    ]


@pytest.mark.parametrize(
    ("matrix", "turn"),
    [
        ("200 0 0 100 0 0", (False, 0)),
        ("-200 0 0 -100 200 100", (False, 180)),
        ("0 -200 100 0 0 200", (False, 90)),
        ("0 200 -100 0 100 0", (False, 270)),
        ("200 0 0 -100 0 100", (True, 180)),
        ("-200 0 0 100 200 0", (True, 0)),
    ],
)
def test_placement_reports_image_turn(tmp_path: Path, matrix: str, turn: tuple[bool, int]) -> None:
    pdf_path = tmp_path / "turned.pdf"
    pdf = pikepdf.Pdf.new()
    _add_image_page(pdf, (0, 0, 200, 100))
    pdf.pages[0]["/Contents"] = pdf.make_stream(f"q {matrix} cm /Im0 Do Q".encode())
    pdf.save(pdf_path)

    [placement] = geometry._extract_image_placements_from_pdf(pdf_path)

    assert placement is not None
    assert placement[1] == turn


def _wrap_in_form(pdf: pikepdf.Pdf, content: str, matrix: str, form_content: str) -> None:
    page = pdf.pages[0]
    form = pdf.make_stream(form_content.encode())
    form["/Type"] = pikepdf.Name.XObject
    form["/Subtype"] = pikepdf.Name.Form
    form["/BBox"] = [0, 0, 200, 100]
    form["/Matrix"] = [float(value) for value in matrix.split()]
    form["/Resources"] = page["/Resources"]
    page["/Resources"] = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary({"/Fm0": form})})
    page["/Contents"] = pdf.make_stream(content.encode())


@pytest.mark.parametrize(
    ("matrix", "form_content"),
    [
        ("1 0 0 1 0 0", "q -200 0 0 -100 200 100 cm /Im0 Do Q"),
        ("-1 0 0 -1 200 100", "q 200 0 0 100 0 0 cm /Im0 Do Q"),
    ],
)
def test_form_image_reports_its_turn(tmp_path: Path, matrix: str, form_content: str) -> None:
    pdf_path = tmp_path / "form.pdf"
    pdf = pikepdf.Pdf.new()
    _add_image_page(pdf, (0, 0, 200, 100))
    _wrap_in_form(pdf, "q /Fm0 Do Q", matrix, form_content)
    pdf.save(pdf_path)

    assert geometry._extract_image_placements_from_pdf(pdf_path) == [
        ((0.0, 0.0, 200.0, 100.0), (False, 180))
    ]


def test_self_drawing_form_is_read_once(tmp_path: Path) -> None:
    pdf_path = tmp_path / "loop.pdf"
    pdf = pikepdf.Pdf.new()
    _add_image_page(pdf, (0, 0, 200, 100))
    _wrap_in_form(pdf, "/Fm0 Do", "1 0 0 1 0 0", "/Fm0 Do q 200 0 0 100 0 0 cm /Im0 Do Q")
    form = pdf.pages[0]["/Resources"]["/XObject"]["/Fm0"]
    form["/Resources"]["/XObject"]["/Fm0"] = form
    pdf.save(pdf_path)

    assert geometry._extract_image_placements_from_pdf(pdf_path) == [
        ((0.0, 0.0, 200.0, 100.0), (False, 0))
    ]


def test_shared_page_renderer_produces_uncompressed_ppm(tmp_path: Path) -> None:
    pdf_path = tmp_path / "page.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(72, 72))
    pdf.save(pdf_path)

    rendered_path = render_pdf_page_to_ppm(
        pdf_path,
        1,
        dpi=72,
        output_dir=tmp_path,
    )

    assert rendered_path is not None
    rendered = Path(rendered_path)
    try:
        assert rendered.parent == tmp_path
        assert rendered.read_bytes().startswith(b"P6")
    finally:
        rendered.unlink(missing_ok=True)


def test_metadata_analysis_consumes_bulk_placements_in_page_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = Path("input.pdf")
    rotations = [
        PageRotation(page_number=1, original_pdf_rotation=0, mediabox=[0, 0, 200, 100]),
        PageRotation(
            page_number=2,
            original_pdf_rotation=90,
            mediabox=[0, 0, 100, 200],
            user_unit=2,
        ),
    ]
    rectangles = [(10.0, 10.0, 180.0, 80.0), None]
    extract_placements = Mock(return_value=[(rectangles[0], (False, 180)), None])
    monkeypatch.setattr(backend_pipeline, "extract_page_rotations", lambda _path: rotations)
    monkeypatch.setattr(backend_pipeline, "_extract_image_placements_from_pdf", extract_placements)
    monkeypatch.setattr(backend_pipeline, "get_page_image_encodings", lambda _path: {})
    backend = _MetadataBackend()
    backend.config = SimpleNamespace(
        page_modifications=None,
        force_full_ocr=False,
    )

    context = backend_pipeline.BackendPipelineMixin._analyze_pdf_metadata(
        backend,
        input_pdf,
        ProcessingStats(),
        None,
    )

    extract_placements.assert_called_once_with(input_pdf, page_count=2)
    assert [item["image_rect"] for item in context["all_rotation_dicts"]] == rectangles
    assert [rotation.image_turn for rotation in rotations] == [(False, 180), (False, 0)]


def test_exif_image_loader_closes_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (4, 3), "white").save(image_path)
    source = Image.open(image_path)
    monkeypatch.setattr(pdf_page_geometry.Image, "open", lambda _path: source)

    result = pdf_page_geometry.load_image_with_exif_rotation(image_path, 128.0)

    assert result is not None
    assert result.shape == (3, 4, 3)
    assert source.fp is None
