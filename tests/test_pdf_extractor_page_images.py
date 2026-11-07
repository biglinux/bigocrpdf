"""Choosing between an embedded page image and re-rendering the page."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from bigocrpdf.services.ocr.backend_embedded_image_pipeline import (
    BackendEmbeddedImagePipelineMixin,
)
from bigocrpdf.services.ocr.config import OCRConfig
from bigocrpdf.services.ocr.page_worker import (
    _apply_pdf_rotation,
    detect_image_quality,
    detect_original_format,
)
from bigocrpdf.services.ocr.pdf_extractor import (
    Extraction,
    PDFImageExtractor,
    _add_pdfimages_mapping_line,
    _PageImage,
    _small_page_image_reason,
    load_image_with_exif_rotation,
)
from bigocrpdf.utils.pdf_utils import open_jpeg_within

# A page whose box maps one point per source pixel, as ImageMagick and phone
# photo converters produce: the image spans it at 72 ppi.
PHOTO_PAGE = (1920.0, 2560.0)
PHOTO_IMAGE = _PageImage(index=0, width=1920, height=2560, x_ppi=72, y_ppi=72)


def test_full_page_image_at_72_ppi_is_used_as_is():
    """Re-rendering could only upscale it, so the embedded image wins."""
    assert _small_page_image_reason(PHOTO_PAGE, PHOTO_IMAGE) is None


def test_full_page_scan_at_300_ppi_is_used_as_is():
    scan = _PageImage(index=0, width=2550, height=3300, x_ppi=300, y_ppi=300)
    assert _small_page_image_reason((612.0, 792.0), scan) is None


def test_logo_on_a_page_falls_back_to_rendering():
    logo = _PageImage(index=0, width=600, height=600, x_ppi=300, y_ppi=300)
    reason = _small_page_image_reason((612.0, 792.0), logo)
    assert reason is not None
    assert "24%w" in reason


def test_unknown_placement_resolution_keeps_the_image():
    unplaced = _PageImage(index=0, width=1920, height=2560, x_ppi=0, y_ppi=0)
    assert _small_page_image_reason(PHOTO_PAGE, unplaced) is None


def test_oversized_scan_is_rendered_instead(tmp_path: Path):
    """Decoding a 150 MP scan takes gigabytes; a page render stays in budget."""
    extractor = PDFImageExtractor(max_render_megapixels=45, max_image_megapixels=128)
    scan = _PageImage(index=0, width=10_000, height=15_000, x_ppi=1210, y_ppi=1282)
    fallback_pages: list[int] = []
    extraction = Extraction([None])

    extractor._process_extracted_page(
        1, 0, tmp_path, {1: [scan]}, {1: (595.0, 842.0)}, set(), fallback_pages, extraction
    )

    assert fallback_pages == [1]
    assert extraction.oversized_pages == {1}


def _oversized_page(tmp_path: Path, suffix: str) -> tuple[list[int], Extraction]:
    """Page 1 holding a 400x300 image over a 0.01 MP decode limit."""
    Image.new("RGB", (400, 300), (90, 90, 90)).save(tmp_path / f"obj-000{suffix}")
    extractor = PDFImageExtractor(max_render_megapixels=45, max_image_megapixels=0.01)
    image = _PageImage(index=0, width=400, height=300, x_ppi=72, y_ppi=72)
    fallback_pages: list[int] = []
    extraction = Extraction([None])
    extractor._process_extracted_page(
        1, 0, tmp_path, {1: [image]}, {1: (400.0, 300.0)}, set(), fallback_pages, extraction
    )
    return fallback_pages, extraction


def test_oversized_jpeg_takes_the_normal_path(tmp_path: Path):
    """It decodes at a reduced scale, so the page keeps its corrections."""
    fallback_pages, extraction = _oversized_page(tmp_path, ".jpg")

    assert fallback_pages == []
    assert extraction.oversized_pages == set()
    assert extraction.images == [tmp_path / "page_1.jpg"]


def test_oversized_png_is_rendered(tmp_path: Path):
    fallback_pages, extraction = _oversized_page(tmp_path, ".png")

    assert fallback_pages == [1]
    assert extraction.oversized_pages == {1}


def test_oversized_jpeg_loads_reduced(tmp_path: Path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), (90, 90, 90)).save(path)

    assert load_image_with_exif_rotation(path, 0.05, 0.03).shape == (150, 200, 3)


def test_oversized_png_is_refused(tmp_path: Path):
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 300)).save(path)

    with pytest.raises(ValueError, match="configured limit"):
        load_image_with_exif_rotation(path, 0.05, 0.03)


def test_mapping_line_reads_placement_resolution():
    mapping: dict[int, list[_PageImage]] = {}
    masked: set[int] = set()
    line = "   1     0 image    1920  2560  icc     3   8  jpeg   no         8  0    72    72  550K 3.8%"

    _add_pdfimages_mapping_line(line, mapping, masked)

    assert mapping == {1: [PHOTO_IMAGE]}
    assert not masked


def test_mapping_line_records_masked_page():
    mapping: dict[int, list[_PageImage]] = {}
    masked: set[int] = set()
    line = "   2     3 mask     1920  2560  gray    1   1  jbig2  no        11  0    72    72   12K 1.0%"

    _add_pdfimages_mapping_line(line, mapping, masked)

    assert mapping == {}
    assert masked == {2}


def _phone_jpeg_header(path: Path, width: int, height: int, **save) -> None:
    """A small JPEG whose frame header claims ``width`` x ``height``: its
    header reads like a phone photo's, which no decode needs here."""
    Image.new("RGB", (16, 16), (90, 90, 90)).save(path, **save)
    data = bytearray(path.read_bytes())
    frame = next(i for i in range(len(data) - 1) if data[i] == 0xFF and data[i + 1] in (0xC0, 0xC2))
    data[frame + 5 : frame + 9] = height.to_bytes(2, "big") + width.to_bytes(2, "big")
    path.write_bytes(bytes(data))


def test_turned_phone_photo_keeps_its_exif(tmp_path: Path):
    """EXIF 6 under /Rotate 90 cancels out, past the decompression-bomb size."""
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6
    _phone_jpeg_header(path, 16_000, 12_500, exif=exif)
    pixels = np.zeros((10, 20, 3), np.uint8)

    turned, *_ = _apply_pdf_rotation(pixels, str(path), 1, 90, (False, 0))

    assert turned.shape == (10, 20, 3)


def test_phone_photo_quality_and_format_read(tmp_path: Path):
    path = tmp_path / "photo.jpg"
    _phone_jpeg_header(path, 16_000, 12_500, quality=92)

    assert detect_image_quality(str(path)) == 92
    assert detect_original_format(str(path)) == "jpeg"


def test_phone_photo_decodes_at_a_quarter(tmp_path: Path):
    path = tmp_path / "photo.jpg"
    _phone_jpeg_header(path, 16_384, 12_288)

    image, scale = open_jpeg_within(path, 128e6, 45e6)
    with image:
        assert (image.size, scale) == ((4096, 3072), 4)


def test_jpeg_storing_too_much_is_refused(tmp_path: Path):
    path = tmp_path / "photo.jpg"
    _phone_jpeg_header(path, 17_000, 16_000)

    with pytest.raises(ValueError, match="too large"):
        open_jpeg_within(path, 128e6, 45e6)


def test_large_progressive_jpeg_is_refused(tmp_path: Path):
    path = tmp_path / "photo.jpg"
    _phone_jpeg_header(path, 16_384, 12_288, progressive=True)

    with pytest.raises(ValueError, match="progressive"):
        open_jpeg_within(path, 128e6, 45e6)


def test_broken_reduced_jpeg_stays_reduced(tmp_path: Path):
    """A JPEG opened at a reduced scale never falls back to a full decode."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), (90, 90, 90)).save(path)
    path.write_bytes(path.read_bytes()[:-200])

    assert load_image_with_exif_rotation(path, 0.05, 0.03) is None


@pytest.mark.parametrize(("limit", "written_back"), [(1.0, True), (0.05, False)])
def test_reduced_photo_keeps_its_original(tmp_path: Path, limit, written_back):
    """An embedded photo decoded reduced is read, never written over its original."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), (90, 90, 90)).save(path)
    pipeline = BackendEmbeddedImagePipelineMixin()
    pipeline.config = OCRConfig(max_image_megapixels=limit, max_render_megapixels=0.03)
    pipeline.preprocessor = SimpleNamespace(process=lambda img, cancel_check: img)
    pipeline._raise_if_cancelled = lambda: None
    pipeline._replace_pdf_image = MagicMock()
    pipeline._ocr_via_persistent = MagicMock(return_value=[])
    position = SimpleNamespace(width=400, height=300, x=0, y=0, name="/Im0")
    stored = SimpleNamespace(Width=400, Height=300)
    page = SimpleNamespace(Resources=SimpleNamespace(XObject={"/Im0": stored}))

    pipeline._ocr_image_in_page(path, position, None, page, 1, 400, 300, None, None)

    assert pipeline._replace_pdf_image.called is written_back
