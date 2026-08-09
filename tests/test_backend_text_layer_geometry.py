"""Geometry contracts for the OCR text-layer metadata pass."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pikepdf
import pytest
from PIL import Image

from bigocrpdf.services.rapidocr_service import backend_pipeline, pdf_page_geometry
from bigocrpdf.services.rapidocr_service import backend_text_layer_geometry as geometry
from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_page_geometry import render_pdf_page_to_ppm
from bigocrpdf.services.rapidocr_service.rotation import PageRotation


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


def test_bulk_rectangles_match_unit_wrapper_with_one_pdf_open(
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

    unit_rectangles = [
        geometry._extract_image_rect_from_page(pdf_path, page_num) for page_num in range(1, 4)
    ]
    unit_open_count = open_count
    open_count = 0
    bulk_rectangles = geometry._extract_image_rects_from_pdf(pdf_path, page_count=3)

    assert unit_open_count == 3
    assert open_count == 1
    assert bulk_rectangles == unit_rectangles
    assert bulk_rectangles == [
        (10.0, 10.0, 180.0, 80.0),
        (20.0, 20.0, 100.0, 60.0),
        None,
    ]


def test_bulk_rectangles_preserve_none_results_when_pdf_cannot_open(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    assert geometry._extract_image_rect_from_page(missing_pdf, 1) is None
    assert geometry._extract_image_rects_from_pdf(missing_pdf, page_count=3) == [None, None, None]


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

    assert geometry._extract_image_rect_from_page(pdf_path, 1) == (
        10.0,
        10.0,
        180.0,
        80.0,
    )


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


def test_metadata_analysis_consumes_bulk_rectangles_in_page_order(
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
    extract_rectangles = Mock(return_value=rectangles)
    monkeypatch.setattr(backend_pipeline, "extract_page_rotations", lambda _path: rotations)
    monkeypatch.setattr(backend_pipeline, "_extract_image_rects_from_pdf", extract_rectangles)
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
        object(),
        None,
    )

    extract_rectangles.assert_called_once_with(input_pdf, page_count=2)
    assert [item["image_rect"] for item in context["all_rotation_dicts"]] == rectangles


def test_exif_image_loader_closes_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (4, 3), "white").save(image_path)
    source = Image.open(image_path)
    monkeypatch.setattr(pdf_page_geometry.Image, "open", lambda _path: source)

    result = pdf_page_geometry.load_image_with_exif_rotation(image_path)

    assert result is not None
    assert result.shape == (3, 4, 3)
    assert source.fp is None
