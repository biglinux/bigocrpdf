"""PDF image position and pdfimages metadata analysis."""

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pikepdf

from bigocrpdf.utils.logger import logger


@dataclass
class ImagePosition:
    """Position and metadata of an image within a PDF page."""

    name: str  # Resource name (e.g. /Im1)
    page_num: int  # 1-based page number
    x: float  # X coordinate (PDF points)
    y: float  # Y coordinate (PDF points)
    width: float  # Display width (PDF points)
    height: float  # Display height (PDF points)


@dataclass(frozen=True)
class PdfResourceMetrics:
    """Lightweight dimensions needed to enforce OCR resource limits.

    ``image_dimensions`` entries are ``(page_number, width_px, height_px)``.
    Only PDF dictionaries are inspected; image streams are never decoded.
    """

    page_dimensions: tuple[tuple[float, float], ...]
    image_dimensions: tuple[tuple[int, int, int], ...]

    @property
    def total_pages(self) -> int:
        return len(self.page_dimensions)


def inspect_pdf_resource_metrics(
    pdf_path: Path | str,
    *,
    max_pages: int = 0,
) -> PdfResourceMetrics:
    """Inspect page and nested image dimensions without decoding image data."""
    page_dimensions: list[tuple[float, float]] = []
    image_dimensions: list[tuple[int, int, int]] = []

    with pikepdf.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if max_pages > 0 and total_pages > max_pages:
            raise ValueError(f"PDF has {total_pages} pages; configured limit is {max_pages}")

        for page_num, page in enumerate(pdf.pages, 1):
            mediabox = page.mediabox
            try:
                user_unit = float(page.get("/UserUnit", 1))
                width_pts = abs(float(mediabox[2]) - float(mediabox[0])) * user_unit
                height_pts = abs(float(mediabox[3]) - float(mediabox[1])) * user_unit
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Page {page_num} has invalid dimensions") from exc
            if (
                not math.isfinite(user_unit)
                or user_unit <= 0
                or not math.isfinite(width_pts)
                or not math.isfinite(height_pts)
                or width_pts <= 0
                or height_pts <= 0
            ):
                raise ValueError(f"Page {page_num} has invalid dimensions")
            page_dimensions.append((width_pts, height_pts))

            resources = page.get("/Resources")
            image_dimensions.extend(
                _resource_image_dimensions(resources, page_num, seen_objects=set(), depth=0)
            )

    return PdfResourceMetrics(tuple(page_dimensions), tuple(image_dimensions))


def _resource_image_dimensions(  # noqa: C901 - recursive PDF trust checks stay together
    resources,
    page_num: int,
    *,
    seen_objects: set[tuple[int, int]],
    depth: int,
) -> list[tuple[int, int, int]]:
    """Collect image dimensions through nested Form XObject resources."""
    if resources is None:
        return []
    if depth > 64:
        raise ValueError(f"Page {page_num} has excessively nested PDF resources")

    xobjects = resources.get("/XObject")
    if xobjects is None:
        return []

    dimensions: list[tuple[int, int, int]] = []
    for _name, xobject in xobjects.items():
        object_number, generation = xobject.objgen
        object_key = (int(object_number), int(generation))
        if object_number and object_key in seen_objects:
            continue
        if object_number:
            seen_objects.add(object_key)

        subtype = str(xobject.get("/Subtype", ""))
        if subtype == "/Image":
            try:
                width_px = int(xobject.get("/Width", 0))
                height_px = int(xobject.get("/Height", 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Page {page_num} has invalid image dimensions") from exc
            if width_px <= 0 or height_px <= 0:
                raise ValueError(
                    f"Page {page_num} has invalid image dimensions: {width_px}x{height_px}"
                )
            dimensions.append((page_num, width_px, height_px))
        elif subtype == "/Form":
            dimensions.extend(
                _resource_image_dimensions(
                    xobject.get("/Resources"),
                    page_num,
                    seen_objects=seen_objects,
                    depth=depth + 1,
                )
            )

    return dimensions


def _multiply_ctm(current: list[float], m: list[float]) -> list[float]:
    """Multiply two 3x3 affine matrices (stored as 6 elements)."""
    a1, b1, c1, d1, e1, f1 = current
    a2, b2, c2, d2, e2, f2 = m
    return [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    ]


def _get_page_xobjects(page) -> dict[str, dict]:
    """Return {name: {width, height, obj}} for image XObjects on a page.

    ``obj`` is the PDF object number, used to correlate each image with its
    ``pdfimages -list`` entry (whose "object" column carries the same number).
    """
    xobjects = {}
    if "/Resources" in page and "/XObject" in page.Resources:
        for name, xobj in page.Resources.XObject.items():
            if xobj.get("/Subtype") == "/Image":
                try:
                    obj_num = int(xobj.objgen[0])
                except Exception:
                    obj_num = 0
                xobjects[str(name)] = {
                    "width": int(xobj.get("/Width", 0)),
                    "height": int(xobj.get("/Height", 0)),
                    "obj": obj_num,
                }
    return xobjects


def _parse_page_images(commands, xobjects, page_num) -> list[ImagePosition]:
    """Parse content stream commands and return ImagePosition list."""
    identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ctm_stack: list[list[float]] = []
    ctm = list(identity)
    positions: list[ImagePosition] = []

    for operands, operator in commands:
        op = str(operator)
        if op == "q":
            ctm_stack.append(list(ctm))
        elif op == "Q":
            if ctm_stack:
                ctm = ctm_stack.pop()
        elif op == "cm" and len(operands) == 6:
            try:
                m = [float(x) for x in operands]
                ctm = _multiply_ctm(m, ctm)
            except (ValueError, TypeError):
                pass
        elif op == "Do" and len(operands) == 1:
            img_name = str(operands[0])
            if img_name in xobjects:
                a, b, c, d, e, f = ctm
                width = (a * a + b * b) ** 0.5
                height = (c * c + d * d) ** 0.5
                x = e
                y = f - height if d < 0 else f
                positions.append(
                    ImagePosition(
                        name=img_name,
                        page_num=page_num,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    )
                )
                logger.debug(
                    f"Found image {img_name} on page {page_num}: "
                    f"pos=({x:.1f}, {y:.1f}), size={width:.1f}x{height:.1f}"
                )

    return positions


def extract_image_positions(pdf_path: Path) -> dict[int, list[ImagePosition]]:
    """Extract positions and metadata of all images in a PDF."""
    positions: dict[int, list[ImagePosition]] = {}

    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            xobjects = _get_page_xobjects(page)
            if not xobjects:
                continue

            try:
                commands = pikepdf.parse_content_stream(page)
            except Exception as e:
                logger.debug(f"Failed to parse content stream for page {page_num}: {e}")
                continue

            page_positions = _parse_page_images(commands, xobjects, page_num)
            if page_positions:
                positions[page_num] = page_positions

    return positions


@dataclass
class PdfImageInfo:
    """Metadata for a single image entry from ``pdfimages -list``."""

    idx: int
    img_type: str
    width: int
    height: int
    comp_size: int  # compressed size in bytes (from pdfimages -list "size" column)
    object_id: int = 0  # PDF object number (from pdfimages -list "object" column)


def _parse_size_field(s: str) -> int:
    """Parse a size field like '249K', '5411B', '1.2M' into bytes."""
    s = s.strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    if s.endswith("G"):
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    if s.endswith("B"):
        return int(float(s[:-1]))
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_pdfimages_list(
    pdf_path: Path,
) -> tuple[dict[int, list[PdfImageInfo]], set[int]]:
    """Parse ``pdfimages -list`` and return per-page image info.

    Returns:
        A tuple of (mapping, masked_pages) where *mapping* is
        ``{page_num: [PdfImageInfo, …]}`` (masks/smasks excluded)
        and *masked_pages* is the set of pages that have JBIG2 mask
        entries (DjVu-like FG/BG layers).
    """
    try:
        result = subprocess.run(
            ["pdfimages", "-list", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return {}, set()
    except Exception as exc:
        logger.warning(f"pdfimages -list failed: {exc}")
        return {}, set()

    mapping: dict[int, list[PdfImageInfo]] = {}
    masked_pages: set[int] = set()
    for line in result.stdout.splitlines()[2:]:  # skip header lines
        parts = line.split()
        if len(parts) < 14:
            continue
        try:
            page_num = int(parts[0])
            img_idx = int(parts[1])
            img_type = parts[2]
            width = int(parts[3])
            height = int(parts[4])
            # Standard column layout (0-indexed):
            # page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
            # "object" (PDF object number) is at 10, "size" at 14.
            object_id = int(parts[10])
            comp_size = _parse_size_field(parts[14])
        except (ValueError, IndexError):
            continue
        if img_type in ("mask", "smask"):
            masked_pages.add(page_num)
            continue
        info = PdfImageInfo(
            idx=img_idx,
            img_type=img_type,
            width=width,
            height=height,
            comp_size=comp_size,
            object_id=object_id,
        )
        mapping.setdefault(page_num, []).append(info)
    return mapping, masked_pages


def match_positions_to_images(
    positions: list[ImagePosition],
    infos: list[PdfImageInfo],
    obj_by_name: dict[str, int],
    dims_by_name: dict[str, tuple[int, int]],
) -> list[tuple[ImagePosition, "PdfImageInfo | None"]]:
    """Pair each image position with the extracted image it actually belongs to.

    Each ``ImagePosition`` is the *display* rectangle of one image on the page;
    each ``PdfImageInfo`` describes one *extracted* image file.  They must be
    paired by identity, not by sort order — pairing the largest extracted image
    with whichever position comes first in the content stream crams a full-page
    scan's OCR text into, e.g., a 75 pt logo box, scrambling the text layer.

    Matching uses PDF object identity first, allowing one extracted image to be
    reused when the same XObject is drawn more than once. Dimension fallback is
    greedy and uses each remaining ``PdfImageInfo`` at most once.
    A position with no match yields ``None`` so the caller can fall back.
    """
    remaining = list(infos)
    result: list[tuple[ImagePosition, PdfImageInfo | None]] = []

    def _take(pred) -> "PdfImageInfo | None":
        for i, info in enumerate(remaining):
            if pred(info):
                return remaining.pop(i)
        return None

    for pos in positions:
        obj = obj_by_name.get(pos.name)
        match = None
        if obj:
            match = next((info for info in infos if info.object_id == obj), None)
            if match is not None:
                remaining = [info for info in remaining if info is not match]
        if match is None:
            dims = dims_by_name.get(pos.name)
            if dims:
                match = _take(lambda info, d=dims: (info.width, info.height) == d)
        result.append((pos, match))
    return result
