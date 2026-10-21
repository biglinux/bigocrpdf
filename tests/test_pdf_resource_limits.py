"""Integration tests for PDF resource limits before OCR pipeline selection."""

from pathlib import Path
from unittest.mock import Mock

import pikepdf
import pytest

from bigocrpdf.services.ocr import backend as backend_module
from bigocrpdf.services.ocr.backend import ProfessionalPDFOCR
from bigocrpdf.services.ocr.config import OCRConfig, ProcessingStats
from bigocrpdf.services.ocr.pdf_extractor import _page_dimensions
from bigocrpdf.services.ocr.pdf_image_analysis import inspect_pdf_resource_metrics
from bigocrpdf.services.ocr.pipeline_chunked_ocr import _chunk_result_render_size
from bigocrpdf.services.ocr.rotation import extract_page_rotations


def _write_metadata_only_pdf(
    path: Path,
    *,
    page_count: int = 1,
    image_size: tuple[int, int] | None = None,
    nested_image: bool = False,
) -> None:
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(612, 792))

    if image_size is not None:
        image = pdf.make_stream(b"\0")
        image["/Type"] = pikepdf.Name.XObject
        image["/Subtype"] = pikepdf.Name.Image
        image["/Width"], image["/Height"] = image_size
        image["/ColorSpace"] = pikepdf.Name.DeviceGray
        image["/BitsPerComponent"] = 1

        xobject = image
        if nested_image:
            form = pdf.make_stream(b"q Q")
            form["/Type"] = pikepdf.Name.XObject
            form["/Subtype"] = pikepdf.Name.Form
            form["/BBox"] = pikepdf.Array([0, 0, 612, 792])
            form["/Resources"] = pikepdf.Dictionary(
                {"/XObject": pikepdf.Dictionary({"/ImTest": image})}
            )
            xobject = form

        pdf.pages[0]["/Resources"] = pikepdf.Dictionary(
            {"/XObject": pikepdf.Dictionary({"/ResourceTest": xobject})}
        )

    pdf.save(path)


def _backend(config: OCRConfig) -> ProfessionalPDFOCR:
    engine = object.__new__(ProfessionalPDFOCR)
    engine.config = config
    return engine


def test_ocr_config_defaults_enable_resource_guards() -> None:
    config = OCRConfig()

    assert config.max_pdf_pages == 2000
    assert config.max_image_megapixels == 128.0


def test_inspection_accounts_for_page_user_unit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "large-user-unit.pdf"
    _write_metadata_only_pdf(pdf_path)
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        pdf.pages[0]["/UserUnit"] = 10
        pdf.save(pdf_path)

    metrics = inspect_pdf_resource_metrics(pdf_path)

    assert metrics.page_dimensions == ((6120.0, 7920.0),)


def test_image_only_fallback_budget_accounts_for_user_unit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "large-user-unit.pdf"
    _write_metadata_only_pdf(pdf_path)
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        pdf.pages[0]["/UserUnit"] = 2
        pdf.pages[0]["/MediaBox"] = pikepdf.Array([612, 792, 0, 0])
        pdf.save(pdf_path)

    rotations = extract_page_rotations(pdf_path)
    with pikepdf.open(pdf_path) as pdf:
        assert _page_dimensions(pdf.pages[0]) == (1224.0, 1584.0)
    rotation_dicts = [{"mediabox": rotations[0].mediabox, "page_rotation": rotations[0]}]

    assert rotations[0].pdf_dimensions == (1224.0, 1584.0)
    assert _chunk_result_render_size(0, rotation_dicts) == (1224.0, 1584.0)


@pytest.mark.parametrize("use_mixed_pipeline", [False, True])
def test_process_allows_safe_pdf_for_both_pipelines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_mixed_pipeline: bool,
) -> None:
    pdf_path = tmp_path / "safe.pdf"
    output_path = tmp_path / "safe-ocr.pdf"
    _write_metadata_only_pdf(pdf_path, image_size=(1000, 1000))
    engine = _backend(OCRConfig(max_pdf_pages=1, max_image_megapixels=2))
    expected = ProcessingStats(pages_total=1)
    mixed = Mock(return_value=expected)
    image_only = Mock(return_value=expected)
    engine._process_mixed_content_pdf = mixed
    engine._process_image_only_pdf = image_only
    monkeypatch.setattr(
        backend_module,
        "should_use_mixed_content_pipeline",
        lambda _config, _path: use_mixed_pipeline,
    )

    assert engine.process(pdf_path, output_path) is expected
    assert mixed.call_count == int(use_mixed_pipeline)
    assert image_only.call_count == int(not use_mixed_pipeline)


def test_process_rejects_page_count_before_pipeline_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "too-many-pages.pdf"
    _write_metadata_only_pdf(pdf_path, page_count=2)
    engine = _backend(OCRConfig(max_pdf_pages=1))
    pipeline_choice = Mock(return_value=False)
    monkeypatch.setattr(backend_module, "should_use_mixed_content_pipeline", pipeline_choice)

    with pytest.raises(ValueError, match="PDF has 2 pages; configured limit is 1"):
        engine.process(pdf_path, tmp_path / "unused.pdf")

    pipeline_choice.assert_not_called()


def test_process_accepts_photo_sized_page_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page box mapping one point per source pixel reaches the pipeline.

    ImageMagick and phone-photo PDFs produce page boxes that exceed the
    megapixel budget at the preferred DPI; the renderer lowers the DPI for
    them, so refusing the document up front would block OCR entirely.
    """
    pdf_path = tmp_path / "photo-page.pdf"
    _write_metadata_only_pdf(pdf_path)
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        pdf.pages[0]["/MediaBox"] = pikepdf.Array([0, 0, 1920, 2560])
        pdf.save(pdf_path)

    engine = _backend(OCRConfig())
    expected = ProcessingStats(pages_total=1)
    engine._process_image_only_pdf = Mock(return_value=expected)
    monkeypatch.setattr(
        backend_module,
        "should_use_mixed_content_pipeline",
        lambda _config, _path: False,
    )

    assert engine.process(pdf_path, tmp_path / "out.pdf") is expected


def test_process_accepts_oversized_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page holding it is read from a render within the render budget."""
    pdf_path = tmp_path / "oversized-image.pdf"
    _write_metadata_only_pdf(
        pdf_path,
        image_size=(20_000, 10_000),
        nested_image=True,
    )
    engine = _backend(OCRConfig(max_image_megapixels=128))
    expected = ProcessingStats(pages_total=1)
    engine._process_mixed_content_pdf = Mock(return_value=expected)
    monkeypatch.setattr(backend_module, "should_use_mixed_content_pipeline", lambda *_: True)

    assert engine.process(pdf_path, tmp_path / "out.pdf") is expected


def _write_header_only_png(path: Path, width: int, height: int) -> None:
    """A PNG that declares its size but carries no pixels: bytes, not gigabytes."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize(
    "side",
    [
        12_000,  # 144 MP: under Pillow's own bomb limit, over the configured 128 MP
        20_000,  # 400 MP: Pillow refuses it before the configured check runs
    ],
)
def test_oversized_page_image_is_rejected_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, side: int
) -> None:
    from bigocrpdf.services.ocr import pdf_page_geometry

    image_path = tmp_path / "page.png"
    _write_header_only_png(image_path, side, side)
    monkeypatch.setattr(
        pdf_page_geometry.cv2,
        "imread",
        Mock(side_effect=AssertionError("an oversized image reached OpenCV")),
    )

    with pytest.raises(ValueError, match="limit"):
        pdf_page_geometry.load_image_with_exif_rotation(image_path, 128.0)
