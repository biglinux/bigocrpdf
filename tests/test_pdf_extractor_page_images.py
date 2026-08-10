"""Choosing between an embedded page image and re-rendering the page."""

from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    _add_pdfimages_mapping_line,
    _PageImage,
    _small_page_image_reason,
)

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
