"""Page-level helpers for mixed-content PDF OCR."""

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    ImagePosition,
    PdfImageInfo,
    match_positions_to_images,
)
from bigocrpdf.services.rapidocr_service.pdf_image_analysis import _get_page_xobjects
from bigocrpdf.utils.logger import logger

_MULTI_BLANK_RE = re.compile(r"\n{4,}")


def _index_extracted_images(paths: Sequence[Path | str]) -> dict[int, Path]:
    """Map each pdfimages *global* image index to its extracted file path.

    ``pdfimages`` names files ``<prefix>-NNN.<ext>`` where ``NNN`` is the same
    global counter reported in the ``pdfimages -list`` "num" column.  Indexing
    by that number — instead of by position in the (filtered) file list — stays
    correct even when ``_extract_and_filter_images`` drops small-PNG masks and
    shifts later entries.
    """
    out: dict[int, Path] = {}
    for raw in paths:
        p = Path(raw)
        _, _, num = p.stem.rpartition("-")
        try:
            out[int(num)] = p
        except ValueError:
            continue
    return out


def _try_join_line(para: str, stripped_next: str) -> str | None:
    """Try joining next line to current paragraph via hyphen or mid-sentence.

    Returns the joined string, or None if lines should not be joined.
    """
    para_end = para.rstrip()
    if not para_end or not stripped_next:
        return None

    # Hyphenated word break
    if (
        para_end.endswith("-")
        and len(para_end) > 1
        and para_end[-2].isalpha()
        and stripped_next[0].islower()
    ):
        return para_end[:-1] + stripped_next

    # Mid-sentence continuation
    last_ch = para_end[-1]
    if (last_ch.isalpha() or last_ch == ",") and stripped_next[0].islower():
        return para_end + " " + stripped_next

    return None


def _reflow_text(text: str) -> str:
    """Conservative reflow: join only mid-sentence continuations."""
    lines = text.split("\n")
    reflowed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            reflowed.append("")
            i += 1
            continue

        para = line.rstrip()
        i += 1
        while i < len(lines):
            if not lines[i].strip():
                break
            joined = _try_join_line(para, lines[i].strip())
            if joined is None:
                break
            para = joined
            i += 1

        reflowed.append(para)

    return _MULTI_BLANK_RE.sub("\n\n\n", "\n".join(reflowed))


def _mixed_render_candidates(
    pdf_scan,
    image_positions: dict,
    pdfimages_map: dict[int, list[PdfImageInfo]],
    masked_pages: set[int],
) -> set[int]:
    render_candidates = (set(pdfimages_map.keys()) - set(image_positions.keys())) | (
        masked_pages & set(image_positions.keys())
    )
    for page_num, images in image_positions.items():
        if len(images) < 2 or page_num in render_candidates:
            continue
        if _has_overlapping_full_page_images(pdf_scan.pages[page_num - 1], images):
            render_candidates.add(page_num)
    return render_candidates


def _has_overlapping_full_page_images(page, images: list[ImagePosition]) -> bool:
    mediabox = page.mediabox
    page_area = (float(mediabox[2]) - float(mediabox[0])) * (
        float(mediabox[3]) - float(mediabox[1])
    )
    if page_area <= 0:
        return False
    return sum(1 for image in images if (image.width * image.height) / page_area > 0.5) >= 2


def _mixed_excluded_pages(page_modifications: list[dict] | None) -> set[int]:
    if not page_modifications:
        return set()
    return {
        page_num
        for mod in page_modifications
        if (page_num := mod.get("page_number"))
        and (mod.get("deleted") or not mod.get("included_for_ocr", True))
    }


def _mixed_progress_bands(
    render_pages: set[int],
    positioned_image_positions: dict,
) -> tuple[int, int]:
    n_render = len(render_pages) if render_pages else 0
    n_positioned = len(positioned_image_positions)
    n_total_ocr = n_render + n_positioned or 1
    render_band = int(80 * n_render / n_total_ocr)
    return render_band, 80 - render_band


def _pdf_page_size(page) -> tuple[float, float]:
    mediabox = page.mediabox
    return float(mediabox[2]) - float(mediabox[0]), float(mediabox[3]) - float(mediabox[1])


def _position_image_pairs(
    page,
    page_imgs: list[ImagePosition],
    page_img_infos: list[PdfImageInfo],
) -> list[tuple[ImagePosition, PdfImageInfo | None]]:
    xobjects = _get_page_xobjects(page)
    obj_by_name = {name: data["obj"] for name, data in xobjects.items()}
    dims_by_name = {name: (data["width"], data["height"]) for name, data in xobjects.items()}
    pairs = match_positions_to_images(page_imgs, page_img_infos, obj_by_name, dims_by_name)
    used = {id(info) for _, info in pairs if info is not None}
    leftover = sorted(
        (info for info in page_img_infos if id(info) not in used),
        key=lambda info: info.comp_size,
        reverse=True,
    )
    return [
        (image_pos, info if info is not None else (leftover.pop(0) if leftover else None))
        for image_pos, info in pairs
    ]


def _render_mixed_page_image(
    input_pdf: Path,
    render_dir: Path,
    page_num: int,
    dpi: int = 300,
) -> Path | None:
    out_prefix = str(render_dir / f"p{page_num}")
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(dpi),
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                str(input_pdf),
                out_prefix,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"pdftoppm failed for page {page_num}: {exc}")
        return None

    rendered_files = sorted(render_dir.glob(f"p{page_num}-*.ppm"))
    if rendered_files:
        return rendered_files[0]
    logger.warning(f"No rendered image for page {page_num}")
    return None
