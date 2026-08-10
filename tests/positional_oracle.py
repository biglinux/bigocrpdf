"""Helpers for proving OCR text lands where the glyphs it describes are.

Not a test module -- it is imported, not collected.

The method is differential. For each scenario two PDFs are built with the
*same* page geometry:

* ``truth.pdf``  -- visible Helvetica text drawn at known coordinates.
* ``layer.pdf``  -- the invisible OCR layer produced by the real render path,
  from synthetic OCR quadrilaterals derived from those same coordinates.

Both are read back with ``pdftotext -bbox-layout`` and compared word for word.
Every convention we do not control -- whether poppler applies ``/Rotate`` to
the boxes it reports, whether it subtracts the MediaBox origin, how it derives
a word's vertical extent from font metrics -- applies identically to both sides
and cancels out. That is what makes these assertions stable, and it is why no
absolute poppler behaviour is hard-coded anywhere below.

A consequence worth stating: proving the *mapping* needs no OCR engine. The
engine's job is to produce quadrilaterals; turning those into PDF text
coordinates is fully exercised with synthetic ones.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pikepdf
import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.renderer import (
    FONT_SIZE_SCALE_FACTOR,
    TextLayerRenderer,
)

FONT = "Helvetica"
# renderer.py:238 -- Helvetica descent as a fraction of the em size.
DESCENT_FRAC = 0.207

# Placement error expected to be exactly zero for the metric-box family, so the
# tolerance only has to absorb serialisation rounding: ReportLab and
# pdf_assembly both write coordinates with two decimals, and poppler prints six
# significant figures, which bounds the accumulated error below 0.02 pt. 0.5 pt
# leaves a 25x margin over that while staying far below any real defect -- a
# wrong page size, a missing Y-flip or a double conversion all displace text by
# tens to hundreds of points.
TOL_TIGHT = 0.5

# The ink-box family feeds real ink bounding boxes, where font_size is
# estimated as ink_height * FONT_SIZE_SCALE_FACTOR. Helvetica ink height ranges
# from about 0.72 em (no descenders) to 0.93 em (with them), so the estimate
# legitimately varies with the *content* of the line and the descent
# compensation drifts by up to ~0.2 * font_size. 0.30 sits just above that and
# just below the 0.35 baseline-snapping threshold, so a failure here means a
# modelling change, not a regrouping of lines.
TOL_INK_DX = 2.0
TOL_INK_DY_FRAC = 0.30

_WORD_RE = re.compile(
    r'<word xMin="([\d.eE+-]+)" yMin="([\d.eE+-]+)" '
    r'xMax="([\d.eE+-]+)" yMax="([\d.eE+-]+)">([^<]*)</word>'
)

requires_pdftotext = pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="poppler's pdftotext is not installed"
)


@dataclass(frozen=True)
class TruthWord:
    """A word drawn at a known position, in PDF points, bottom-up."""

    token: str
    x: float
    baseline: float
    size: float

    @property
    def width(self) -> float:
        return pdfmetrics.stringWidth(self.token, FONT, self.size)


@dataclass(frozen=True)
class ExtractedWord:
    """A word as poppler reports it: top-down, in points."""

    text: str
    x0: float
    y_top: float
    x1: float
    y_bot: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y_top + self.y_bot) / 2.0)


def tokens(
    count: int, size: float = 12.0, *, x: float = 72.0, top: float = 720.0, leading: float = 40.0
) -> list[TruthWord]:
    """Distinct, well-separated words: one per line, no ligature hazards.

    Tokens are uppercase ASCII plus digits so nothing in ``escape_pdf_text``
    rewrites them, and contain no colon so the export form-splitter leaves them
    alone. One word per line, spaced far enough apart that ``_snap_baselines``
    cannot merge them and no space runs are inserted.
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return [
        TruthWord(
            token=f"W{index:02d}{alphabet[index % len(alphabet)] * 3}",
            x=x,
            baseline=top - index * leading,
            size=size,
        )
        for index in range(count)
    ]


def write_truth_pdf(
    words: list[TruthWord],
    page_size_pts: tuple[float, float],
    out_path: Path,
    *,
    mediabox_origin: tuple[float, float] = (0.0, 0.0),
    rotate: int = 0,
    user_unit: float | None = None,
) -> Path:
    """Draw ``words`` visibly at their stated coordinates."""
    width, height = page_size_pts
    pdf = rl_canvas.Canvas(str(out_path), pagesize=(width, height))
    for word in words:
        pdf.setFont(FONT, word.size)
        pdf.drawString(word.x, word.baseline, word.token)
    pdf.save()
    _apply_page_attributes(out_path, mediabox_origin, rotate, user_unit, page_size_pts)
    return out_path


def _apply_page_attributes(
    path: Path,
    mediabox_origin: tuple[float, float],
    rotate: int,
    user_unit: float | None,
    page_size_pts: tuple[float, float],
) -> None:
    """Rewrite page attributes ReportLab cannot express directly."""
    if mediabox_origin == (0.0, 0.0) and not rotate and user_unit is None:
        return
    width, height = page_size_pts
    origin_x, origin_y = mediabox_origin
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        page = pdf.pages[0]
        if mediabox_origin != (0.0, 0.0):
            page.MediaBox = pikepdf.Array([origin_x, origin_y, origin_x + width, origin_y + height])
        if rotate:
            page.Rotate = rotate
        if user_unit is not None:
            page.UserUnit = user_unit
        pdf.save(path)


def truth_to_metric_quads(
    words: list[TruthWord],
    page_size_pts: tuple[float, float],
    image_size_px: tuple[int, int],
) -> list[OCRResult]:
    """Quadrilaterals chosen so the renderer's model error is exactly zero.

    The renderer derives ``font_size = box_height * FONT_SIZE_SCALE_FACTOR``
    and places the baseline at ``box_bottom + DESCENT_FRAC * font_size``. Both
    are inverted here, so any discrepancy in the comparison is a mapping bug
    rather than the descent approximation.

    The box width is set to the exact rendered string width, which keeps
    ``_line_horizontal_scale`` at 100 -- see ``assert_tz_is_unity``.
    """
    width_pts, height_pts = page_size_pts
    image_width_px, image_height_px = image_size_px
    px_per_pt_x = image_width_px / width_pts
    px_per_pt_y = image_height_px / height_pts

    results = []
    for word in words:
        box_height_px = (word.size / FONT_SIZE_SCALE_FACTOR) * px_per_pt_y
        bottom_pts = word.baseline - DESCENT_FRAC * word.size
        bottom_px = (height_pts - bottom_pts) * px_per_pt_y
        top_px = bottom_px - box_height_px
        left_px = word.x * px_per_pt_x
        right_px = (word.x + word.width) * px_per_pt_x
        results.append(
            OCRResult(
                text=word.token,
                box=[
                    [left_px, top_px],
                    [right_px, top_px],
                    [right_px, bottom_px],
                    [left_px, bottom_px],
                ],
                confidence=0.99,
            )
        )
    return results


def render_layer_pdf(
    ocr_results: list[OCRResult],
    image_size_px: tuple[int, int],
    page_size_pts: tuple[float, float],
    out_path: Path,
    *,
    rotation: int = 0,
    page_rotate: int = 0,
    image_offset: tuple[float, float] | None = None,
    mediabox_origin: tuple[float, float] = (0.0, 0.0),
    user_unit: float | None = None,
) -> Path:
    """Run the production render path (path A) onto a blank page.

    The renderer is pointed at an empty font directory so it falls back to
    Helvetica, matching the truth PDF. With the shipped ``latin.ttf`` the two
    sides would use different metrics, and the comparison would measure the
    font difference instead of the coordinate mapping.
    """
    renderer = TextLayerRenderer(OCRConfig(font_base_path=Path("/nonexistent-font-dir")))
    pdf = rl_canvas.Canvas(str(out_path), pagesize=page_size_pts)
    renderer.render(
        pdf,
        ocr_results,
        image_size_px,
        rotation=rotation,
        page_size_pts=page_size_pts,
        image_offset=image_offset,
    )
    pdf.save()
    # ``rotation`` is the counter-rotation applied to the text; ``page_rotate``
    # is the /Rotate attribute a viewer will apply on top. They are separate
    # knobs on purpose -- conflating them is what makes rotation behaviour hard
    # to reason about, and telling them apart is the point of the rotation
    # tests in test_text_layer_page_geometry_suspected.py.
    _apply_page_attributes(out_path, mediabox_origin, page_rotate, user_unit, page_size_pts)
    return out_path


def extract_words(pdf_path: Path) -> list[ExtractedWord]:
    """Word boxes as poppler reports them, in page points."""
    result = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return [
        ExtractedWord(
            text=match.group(5),
            x0=float(match.group(1)),
            y_top=float(match.group(2)),
            x1=float(match.group(3)),
            y_bot=float(match.group(4)),
        )
        for match in _WORD_RE.finditer(result.stdout)
    ]


def reading_order(pdf_path: Path) -> list[str]:
    """Non-empty extracted lines, in the order a reader would meet them."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def by_token(words: list[ExtractedWord]) -> dict[str, ExtractedWord]:
    return {word.text: word for word in words}


def assert_boxes_match(
    truth_words: list[ExtractedWord],
    layer_words: list[ExtractedWord],
    *,
    dx: float = TOL_TIGHT,
    dy: float = TOL_TIGHT,
) -> None:
    """Every truth word has a layer word of the same token in the same place."""
    truth_by_token = by_token(truth_words)
    layer_by_token = by_token(layer_words)
    assert set(truth_by_token) == set(layer_by_token), (
        f"token sets differ: only in truth={sorted(set(truth_by_token) - set(layer_by_token))}, "
        f"only in layer={sorted(set(layer_by_token) - set(truth_by_token))}"
    )

    failures = []
    for token, truth in truth_by_token.items():
        layer = layer_by_token[token]
        delta_x = layer.x0 - truth.x0
        delta_y = layer.y_top - truth.y_top
        if abs(delta_x) > dx or abs(delta_y) > dy:
            failures.append(f"{token}: dx={delta_x:+.2f}pt dy={delta_y:+.2f}pt")
    assert not failures, "text layer is displaced from the glyphs:\n  " + "\n  ".join(failures)


def assert_selection_alignment(
    truth_words: list[ExtractedWord],
    layer_words: list[ExtractedWord],
    *,
    frac: float = 0.5,
) -> None:
    """The nearest invisible word to each glyph must be that same word.

    This is the operational form of "text selection lands on the right glyph",
    and being ratio-based it holds unchanged for rotated, offset and
    oddly-sized pages.
    """
    failures = []
    for truth in truth_words:
        truth_cx, truth_cy = truth.center
        nearest = min(
            layer_words,
            key=lambda w: (w.center[0] - truth_cx) ** 2 + (w.center[1] - truth_cy) ** 2,
        )
        limit = frac * min(abs(truth.x1 - truth.x0), abs(truth.y_bot - truth.y_top))
        distance = (
            (nearest.center[0] - truth_cx) ** 2 + (nearest.center[1] - truth_cy) ** 2
        ) ** 0.5
        if nearest.text != truth.text or distance > limit:
            failures.append(
                f"{truth.text}: nearest layer word is {nearest.text!r} "
                f"at {distance:.2f}pt (limit {limit:.2f}pt)"
            )
    assert not failures, "selection would hit the wrong glyph:\n  " + "\n  ".join(failures)


def assert_tz_is_unity(pdf_path: Path) -> None:
    """No horizontal stretch is in play, so the test really measures placement.

    Without this guard a change to the stretch logic could silently turn a
    positional assertion into a no-op.
    """
    with pikepdf.open(pdf_path) as pdf:
        stream = bytes(pdf.pages[0].Contents.read_bytes()).decode("latin-1")
    scales = [float(value) for value in re.findall(r"([\d.]+) Tz", stream)]
    assert scales, "no Tz operator found; the renderer always emits one"
    assert all(abs(scale - 100.0) < 0.1 for scale in scales), (
        f"line is being stretched (Tz={scales}); the fixture no longer measures placement"
    )
