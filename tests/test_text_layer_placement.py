"""The OCR text layer must land on top of the glyphs it describes.

Nothing in this project asserted that before. The searchable PDF could have its
text displaced by any amount and every existing test would still pass, because
they all check that text is *present* and extractable, never *where* it is.

Method: see tests/positional_oracle. Each test builds a visible-text PDF and an
invisible-OCR-layer PDF with identical geometry and compares what poppler
reports for both, so poppler's own conventions cancel out.
"""

from pathlib import Path

import pytest

from tests.positional_oracle import (
    TOL_TIGHT,
    assert_boxes_match,
    assert_selection_alignment,
    assert_tz_is_unity,
    by_token,
    extract_words,
    reading_order,
    render_layer_pdf,
    requires_pdftotext,
    tokens,
    truth_to_metric_quads,
    write_truth_pdf,
)

pytestmark = requires_pdftotext

# (page size in points, OCR image size in pixels). The image sizes are the ones
# these page sizes really produce at 300 DPI, plus one deliberately odd pair so
# an accidentally isotropic mapping cannot pass.
PAGE_GEOMETRIES = [
    pytest.param((612.0, 792.0), (2550, 3300), id="letter-300dpi"),
    pytest.param((595.0, 842.0), (2480, 3508), id="a4-300dpi"),
    pytest.param((200.0, 100.0), (1000, 250), id="wide-anisotropic"),
    pytest.param((1224.0, 1584.0), (2448, 3168), id="large-scan-144dpi"),
]


def _truth_and_layer(
    tmp_path: Path,
    page_size: tuple[float, float],
    image_size: tuple[int, int],
    *,
    word_count: int = 5,
    size: float = 12.0,
    image_offset: tuple[float, float] | None = None,
    quad_shift: tuple[float, float] = (0.0, 0.0),
):
    """Build the matched pair and return (truth words, layer words, layer path)."""
    width, height = page_size
    words = tokens(
        word_count,
        size=size,
        x=0.12 * width,
        top=0.88 * height,
        leading=min(40.0, 0.7 * height / max(word_count, 1)),
    )
    truth_pdf = write_truth_pdf(words, page_size, tmp_path / "truth.pdf")
    quads = truth_to_metric_quads(words, page_size, image_size)
    if quad_shift != (0.0, 0.0):
        shift_x, shift_y = quad_shift
        for quad in quads:
            quad.box = [[x - shift_x, y - shift_y] for x, y in quad.box]
    layer_pdf = render_layer_pdf(
        quads,
        image_size,
        page_size,
        tmp_path / "layer.pdf",
        image_offset=image_offset,
    )
    return extract_words(truth_pdf), extract_words(layer_pdf), layer_pdf


def test_the_oracle_agrees_with_itself(tmp_path: Path):
    """A truth PDF compared against itself must match exactly.

    Runs first because nothing below means anything if the comparison itself
    is not sound.
    """
    words = tokens(4)
    truth_pdf = write_truth_pdf(words, (612.0, 792.0), tmp_path / "truth.pdf")

    extracted = extract_words(truth_pdf)

    assert [word.text for word in extracted] == [word.token for word in words]
    assert_boxes_match(extracted, extracted, dx=1e-9, dy=1e-9)


@pytest.mark.parametrize("page_size,image_size", PAGE_GEOMETRIES)
def test_text_lands_on_the_glyphs(tmp_path: Path, page_size, image_size):
    truth_words, layer_words, layer_pdf = _truth_and_layer(tmp_path, page_size, image_size)

    assert_tz_is_unity(layer_pdf)
    assert_boxes_match(truth_words, layer_words)
    assert_selection_alignment(truth_words, layer_words)


@pytest.mark.parametrize("size", [6.0, 9.0, 12.0, 24.0, 48.0])
def test_placement_holds_across_font_sizes(tmp_path: Path, size):
    """The descent compensation scales with font size, so sweep it."""
    truth_words, layer_words, _ = _truth_and_layer(
        tmp_path, (612.0, 792.0), (2550, 3300), size=size, word_count=4
    )

    assert_boxes_match(truth_words, layer_words)


def test_horizontal_mapping_is_anisotropic(tmp_path: Path):
    """A page twice as wide must move x twice as far and leave y alone.

    Guards against the averaged ``px_to_pt`` (renderer.py:220) leaking from
    font sizing into placement, which no other test would notice on the square
    -ish page sizes real scans produce.
    """
    words = tokens(3)
    narrow_truth = extract_words(write_truth_pdf(words, (300.0, 792.0), tmp_path / "n.pdf"))
    wide_words = [
        type(word)(token=word.token, x=word.x * 2, baseline=word.baseline, size=word.size)
        for word in words
    ]
    wide_truth = extract_words(write_truth_pdf(wide_words, (600.0, 792.0), tmp_path / "w.pdf"))

    narrow_layer = extract_words(
        render_layer_pdf(
            truth_to_metric_quads(words, (300.0, 792.0), (1250, 3300)),
            (1250, 3300),
            (300.0, 792.0),
            tmp_path / "nl.pdf",
        )
    )
    wide_layer = extract_words(
        render_layer_pdf(
            truth_to_metric_quads(wide_words, (600.0, 792.0), (1250, 3300)),
            (1250, 3300),
            (600.0, 792.0),
            tmp_path / "wl.pdf",
        )
    )

    assert_boxes_match(narrow_truth, narrow_layer)
    assert_boxes_match(wide_truth, wide_layer)
    narrow_by_token, wide_by_token = by_token(narrow_layer), by_token(wide_layer)
    for token in narrow_by_token:
        assert wide_by_token[token].x0 == pytest.approx(
            2 * narrow_by_token[token].x0, abs=TOL_TIGHT
        )
        assert wide_by_token[token].y_top == pytest.approx(
            narrow_by_token[token].y_top, abs=TOL_TIGHT
        )


def test_reading_order_matches_the_visible_page(tmp_path: Path):
    """What a user copies out must be what a reader sees, line for line.

    This is the assertion export_service depends on: its TXT and ODT come from
    pdftotext over this very layer.
    """
    words = tokens(6)
    truth_pdf = write_truth_pdf(words, (612.0, 792.0), tmp_path / "truth.pdf")
    layer_pdf = render_layer_pdf(
        truth_to_metric_quads(words, (612.0, 792.0), (2550, 3300)),
        (2550, 3300),
        (612.0, 792.0),
        tmp_path / "layer.pdf",
    )

    assert reading_order(layer_pdf) == reading_order(truth_pdf)


def test_image_offset_shifts_the_whole_layer(tmp_path: Path):
    """An image not at the page origin must carry its text layer with it.

    OCR coordinates are relative to the image, so the renderer translates by
    the image rect. The branch is never exercised by any other test.
    """
    offset = (72.0, 108.0)
    truth_words, layer_words, _ = _truth_and_layer(
        tmp_path,
        (612.0, 792.0),
        (2550, 3300),
        image_offset=offset,
        quad_shift=(offset[0] * 2550 / 612.0, -offset[1] * 3300 / 792.0),
    )

    assert_boxes_match(truth_words, layer_words)


def test_an_empty_page_produces_no_text(tmp_path: Path):
    layer_pdf = render_layer_pdf([], (2550, 3300), (612.0, 792.0), tmp_path / "layer.pdf")

    assert extract_words(layer_pdf) == []
