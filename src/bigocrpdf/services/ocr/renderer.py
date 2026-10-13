"""Invisible OCR text placement for searchable PDFs.

TextLayerRenderer draws through ReportLab for image-only pages; pdf_assembly
writes the same placement as raw content-stream operators for mixed-content
and embedded-image pages.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE, MIN_FONT_SIZE

if TYPE_CHECKING:
    from bigocrpdf.services.ocr.config import OCRConfig, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    """Represents a text box with position and content.

    All coordinates are in PDF points (1/72 inch), with origin at bottom-left.

    Attributes:
        text: The text content
        x: X coordinate (left edge)
        y: Y coordinate (baseline)
        width: Box width
        height: Box height
        confidence: OCR confidence score (0-1)
        font_size: Calculated font size in points
    """

    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    font_size: float = 12.0


@dataclass
class PageTextLayer:
    """Text layer data for one page.

    Attributes:
        page_num: 1-based page number
        boxes: List of text boxes
        width_pts: Page width in points
        height_pts: Page height in points
        image_width_px: Original image width in pixels
        image_height_px: Original image height in pixels
    """

    page_num: int
    boxes: list[TextBox] = field(default_factory=list)
    width_pts: float = A4[0]
    height_pts: float = A4[1]
    image_width_px: int = 0
    image_height_px: int = 0


def _line_text_with_spacing(
    line_boxes: list[TextBox],
    font_name: str,
    line_font_size: float,
) -> str:
    space_width = pdfmetrics.stringWidth(" ", font_name, line_font_size)
    if space_width <= 0:
        space_width = line_font_size * 0.25

    parts: list[str] = []
    for i, box in enumerate(line_boxes):
        parts.append(box.text)
        if i < len(line_boxes) - 1:
            next_box = line_boxes[i + 1]
            gap = next_box.x - (box.x + box.width)
            num_spaces = max(1, round(gap / space_width)) if gap > 0 else 1
            parts.append(" " * num_spaces)
    return "".join(parts)


# Bounds on the invisible run's horizontal stretch, in percent.
MIN_HORIZ_SCALE = 30.0
MAX_HORIZ_SCALE = 300.0


def _line_horizontal_scale(
    line_text: str,
    line_width: float,
    font_name: str,
    line_font_size: float,
) -> float:
    """Horizontal stretch, as a percentage, bounded to the HORIZ_SCALE range.

    A single OCR box much wider or narrower than its text (a stray detection,
    or a CJK line whose placeholder glyphs bear no relation to the real widths)
    would otherwise stretch the invisible run by a factor of ten, making the
    selection rectangle cover half the page.
    """
    natural_width = pdfmetrics.stringWidth(line_text, font_name, line_font_size)
    if natural_width > 0 and line_width > 0:
        scale = line_width / natural_width * 100.0
    else:
        scale = 100.0
    return max(MIN_HORIZ_SCALE, min(MAX_HORIZ_SCALE, scale))


def _normalize_ocr_quadrilateral(
    box: object,
) -> (
    tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None
):
    """Validate nested or legacy flat OCR coordinates as four numeric points."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    raw_points = (
        box[:4]
        if isinstance(box[0], (list, tuple))
        else ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
    )
    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw_points)
    except (IndexError, TypeError, ValueError):
        return None
    if len(points) != 4 or not all(math.isfinite(value) for point in points for value in point):
        return None
    return points[0], points[1], points[2], points[3]


# Helvetica's descent as a fraction of the em size (207/1000). The OCR box
# bottom is the descender line; the PDF text origin is the baseline above it.
_DESCENT_FRAC = 0.207


def place_text_box(
    result: "OCRResult",
    scale_x: float,
    scale_y: float,
    origin_x: float,
    top_y: float,
) -> TextBox | None:
    """Map one OCR result from image pixels to PDF points.

    ``origin_x`` and ``top_y`` are the PDF coordinates of the image's left and
    top edges. Width and height come from the quadrilateral's own edges, so a
    skewed box keeps its glyph height, and the baseline sits on the mean of the
    bottom corners. Text stays horizontal so extractors rebuild line order.
    Returns None for blank text or an invalid quadrilateral.
    """
    text = result.text.strip()
    quadrilateral = _normalize_ocr_quadrilateral(result.box)
    if not text or quadrilateral is None:
        return None
    # OCR box order: top-left, top-right, bottom-right, bottom-left.
    tl, tr, br, bl = quadrilateral
    width = math.hypot(br[0] - bl[0], br[1] - bl[1]) * scale_x
    left_h = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
    right_h = math.hypot(tr[0] - br[0], tr[1] - br[1])
    height = (left_h + right_h) / 2.0 * scale_y
    font_size = max(MIN_FONT_SIZE, min(height * FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE))
    y_bottom = (bl[1] + br[1]) / 2.0 * scale_y
    return TextBox(
        text=text,
        x=origin_x + min(tl[0], bl[0]) * scale_x,
        y=top_y - y_bottom + _DESCENT_FRAC * font_size,
        width=width,
        height=height,
        confidence=result.confidence,
        font_size=font_size,
    )


def snap_baselines(boxes: list[TextBox]) -> None:
    """Align text boxes on the same visual line to the same y-coordinate.

    PDF text extractors (pdftotext, pdfgrep) use vertical position to
    group characters into lines.  When OCR boxes on the same visual line
    have slightly different y-coords (scan curvature), extractors split
    them into separate lines.

    Uses sequential clustering by y-proximity with cluster centroid
    comparison.  Each cluster is snapped to its median y value.
    """
    if not boxes:
        return

    # Sort by y descending (top of page = largest PDF y first)
    sorted_boxes = sorted(boxes, key=lambda b: -b.y)

    clusters: list[list[TextBox]] = []
    current_cluster: list[TextBox] = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        cluster_y = sum(b.y for b in current_cluster) / len(current_cluster)
        cluster_min_h = min(b.height for b in current_cluster)
        # 35% of the smaller height catches scan curvature but stays below
        # the gap between two lines of the same size.
        threshold = min(cluster_min_h, box.height) * 0.35
        if abs(cluster_y - box.y) <= threshold:
            current_cluster.append(box)
        else:
            clusters.append(current_cluster)
            current_cluster = [box]
    clusters.append(current_cluster)

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        ys = sorted(b.y for b in cluster)
        median_y = ys[len(ys) // 2]
        for b in cluster:
            b.y = median_y


def text_lines(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Group snapped boxes into lines, top of the page first, each left to right."""
    lines: dict[float, list[TextBox]] = defaultdict(list)
    for box in boxes:
        lines[box.y].append(box)
    return [sorted(line, key=lambda b: b.x) for _y, line in sorted(lines.items(), reverse=True)]


@dataclass(frozen=True)
class LineRun:
    """One invisible text run: the whole line drawn from its first box."""

    text: str
    x: float
    y: float
    font_size: float
    horizontal_scale: float


def line_run(line_boxes: list[TextBox], font_name: str) -> LineRun:
    """Join a line's boxes into one run stretched to the line's OCR width."""
    font_size = sum(b.font_size for b in line_boxes) / len(line_boxes)
    text = _line_text_with_spacing(line_boxes, font_name, font_size)
    width = (line_boxes[-1].x + line_boxes[-1].width) - line_boxes[0].x
    return LineRun(
        text=text,
        x=line_boxes[0].x,
        y=line_boxes[0].y,
        font_size=font_size,
        horizontal_scale=_line_horizontal_scale(text, width, font_name, font_size),
    )


class TextLayerRenderer:
    """Creates invisible text layers for searchable PDFs.

    The renderer creates PDF pages with transparent text positioned
    exactly where OCR detected text in the original image. The text
    is invisible (render mode 3) but selectable and searchable.

    Accepts an ``OCRConfig`` for font management and DPI-based
    coordinate conversion.
    """

    # Register fonts as class variable to avoid re-registration
    _registered_fonts: dict[str, TTFont] = {}
    _missing_cjk_logged = False

    def __init__(self, config: "OCRConfig") -> None:
        self.config = config
        self._font_name = self._register_font()

    def _register_font(self) -> str:
        """Register the configured font and return its ReportLab name.

        A missing font fails the job: Helvetica would drop every character
        outside Latin-1 from the text layer.
        """
        font_path = self.config.font_path
        if font_path is None:
            return "Helvetica"
        if not font_path.is_file():
            raise FileNotFoundError(f"Text-layer font not found: {font_path}")
        return self._register(font_path)

    @classmethod
    def _register(cls, font_path: Path) -> str:
        font_name = f"OCRFont_{font_path.stem}"
        if font_name not in cls._registered_fonts:
            font = TTFont(font_name, str(font_path))
            pdfmetrics.registerFont(font)
            cls._registered_fonts[font_name] = font
        return font_name

    def _font_for(self, text: str) -> str:
        """The main font, or the CJK font when it has glyphs the main one lacks."""
        if self.config.font_path is None:
            return self._font_name
        glyphs = self._registered_fonts[self._font_name].face.charToGlyph
        missing = {ord(char) for char in text if ord(char) not in glyphs and not char.isspace()}
        if not missing:
            return self._font_name

        cjk_path = self.config.cjk_font_path
        if not cjk_path.is_file():
            if not TextLayerRenderer._missing_cjk_logged:
                TextLayerRenderer._missing_cjk_logged = True
                logger.warning(
                    "CJK text-layer font not found: %s; Chinese and Japanese text "
                    "stays out of the text layer",
                    cjk_path,
                )
            return self._font_name
        cjk_name = self._register(cjk_path)
        if missing & self._registered_fonts[cjk_name].face.charToGlyph.keys():
            return cjk_name
        return self._font_name

    def create_text_layer(
        self,
        ocr_results: list["OCRResult"],
        image_width_px: int,
        image_height_px: int,
        page_size_pts: tuple[float, float] | None = None,
    ) -> PageTextLayer:
        """Convert OCR results to a text layer.

        Uses the full OCR quadrilateral for accurate height/width
        measurement and baseline positioning.  Text is kept horizontal
        (no rotation) so that PDF text extractors (pdfgrep, pdftotext,
        copy-paste) correctly reconstruct line order.

        Args:
            ocr_results: List of OCR results for one page.
            image_width_px: Image width in pixels.
            image_height_px: Image height in pixels.
            page_size_pts: If provided, use these page dimensions (width, height)
                in points for coordinate mapping instead of computing from DPI.
                Required for overlay mode where the image DPI may differ from
                the configured DPI.

        Returns:
            PageTextLayer with positioned text boxes.
        """
        if page_size_pts:
            width_pts, height_pts = page_size_pts
            px_to_pt_x = width_pts / image_width_px
            px_to_pt_y = height_pts / image_height_px
        else:
            dpi = float(self.config.dpi)
            px_to_pt = 72.0 / dpi
            px_to_pt_x = px_to_pt
            px_to_pt_y = px_to_pt
            width_pts = image_width_px * px_to_pt
            height_pts = image_height_px * px_to_pt

        layer = PageTextLayer(
            page_num=1,
            width_pts=width_pts,
            height_pts=height_pts,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )

        for result in ocr_results:
            box = place_text_box(result, px_to_pt_x, px_to_pt_y, 0.0, height_pts)
            if box is not None:
                layer.boxes.append(box)

        return layer

    def render(
        self,
        canvas: canvas.Canvas,
        ocr_results: list["OCRResult"],
        image_size: tuple[int, int],
        rotation: int = 0,
        page_size_pts: tuple[float, float] | None = None,
        image_offset: tuple[float, float] | None = None,
    ) -> int:
        """Render invisible text layer directly to an existing canvas.

        Args:
            canvas: ReportLab canvas to draw on.
            ocr_results: List of OCR results for one page.
            image_size: Tuple of ``(width, height)`` in pixels.
            rotation: Rotation angle in degrees (0, 90, 180, 270).
            page_size_pts: If provided, map pixel coordinates to this page
                size instead of using DPI-based calculation.
            image_offset: If provided, translate the text layer by (x, y)
                points so OCR text aligns with an image that does not start
                at the page origin.

        Returns:
            Number of text regions rendered.
        """
        layer = self.create_text_layer(
            ocr_results, image_size[0], image_size[1], page_size_pts=page_size_pts
        )

        # Snap baselines so text extractors group same-line boxes together
        snap_baselines(layer.boxes)

        canvas.saveState()

        # Shift text layer to the image position when the image does not
        # start at the page origin (e.g. centred or offset images).
        if image_offset:
            canvas.translate(image_offset[0], image_offset[1])

        self._apply_page_rotation(canvas, layer, rotation)

        # PDF text render mode 3 makes the text invisible without alpha states.
        count = self._render_text_lines(canvas, layer.boxes)
        canvas.restoreState()
        return count

    @staticmethod
    def _apply_page_rotation(
        canvas: canvas.Canvas,
        layer: PageTextLayer,
        rotation: int,
    ) -> None:
        if rotation == 90:
            canvas.translate(layer.height_pts, 0)
            canvas.rotate(90)
        elif rotation == 180:
            canvas.translate(layer.width_pts, layer.height_pts)
            canvas.rotate(180)
        elif rotation == 270:
            canvas.translate(0, layer.width_pts)
            canvas.rotate(270)

    def _render_text_lines(
        self,
        canvas: canvas.Canvas,
        boxes: list[TextBox],
    ) -> int:
        count = 0
        for line_boxes in text_lines(boxes):
            try:
                font_name = self._font_for("".join(box.text for box in line_boxes))
                run = line_run(line_boxes, font_name)
                text_obj = canvas.beginText()
                text_obj.setTextRenderMode(3)
                text_obj.setTextOrigin(run.x, run.y)
                text_obj.setFont(font_name, run.font_size)
                text_obj.setHorizScale(run.horizontal_scale)
                text_obj.textOut(run.text)
            except Exception as e:
                logger.debug(f"Failed to render line at y={line_boxes[0].y:.1f}: {e}")
                continue
            # Render mode is graphics state and outlives ET; q/Q keeps it local.
            canvas.saveState()
            canvas.drawText(text_obj)
            canvas.restoreState()
            count += len(line_boxes)
        return count
