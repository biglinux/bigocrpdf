"""
PDF assembly and merging utilities.

Functions for creating text layer commands, merging text layers with original PDFs,
overlaying OCR text on pages, and converting to PDF/A.
"""

import logging
from pathlib import Path

import pikepdf
from reportlab.pdfbase import pdfmetrics

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE, MIN_FONT_SIZE
from bigocrpdf.services.rapidocr_service.config import OCRResult
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_content_streams,
    merge_page_fonts,
)

logger = logging.getLogger(__name__)


def escape_pdf_text(text: str) -> str:
    """Escape text for inclusion in PDF literal string.

    Handles characters outside the WinAnsiEncoding range by replacing
    them with closest ASCII equivalents where possible.

    Args:
        text: Raw text to escape

    Returns:
        Escaped text safe for PDF string (latin-1 compatible)
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")

    # Replace common Unicode characters that fall outside latin-1
    _UNICODE_REPLACEMENTS = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "*",  # bullet
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2015": "-",  # horizontal bar
        "\u2032": "'",  # prime
        "\u2033": '"',  # double prime
        "\u2212": "-",  # minus sign
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\ufeff": "",  # BOM / zero-width no-break space
    }
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Filter remaining non-latin-1 characters
    result = []
    for ch in text:
        try:
            ch.encode("latin-1")
            result.append(ch)
        except UnicodeEncodeError:
            result.append("?")
    return "".join(result)


def _build_text_boxes(
    ocr_results: list[OCRResult],
    img_x: float,
    img_y: float,
    img_height: float,
    scale_x: float,
    scale_y: float,
) -> list[dict]:
    """Convert OCR results to PDF-coordinate text boxes."""
    boxes: list[dict] = []
    for r in ocr_results:
        text = r.text.strip()
        if not text:
            continue
        box = r.box
        min_x = min(p[0] for p in box)
        max_x = max(p[0] for p in box)
        min_y = min(p[1] for p in box)
        max_y = max(p[1] for p in box)

        width_pts = (max_x - min_x) * scale_x
        height_pts = (max_y - min_y) * scale_y
        font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, height_pts * FONT_SIZE_SCALE_FACTOR))

        descent_pts = 0.207 * font_size
        boxes.append({
            "text": text,
            "x": img_x + min_x * scale_x,
            "y": img_y + img_height - (max_y * scale_y) + descent_pts,
            "width": width_pts,
            "height": height_pts,
            "font_size": font_size,
        })
    return boxes


def _snap_baselines(boxes: list[dict]) -> None:
    """Cluster boxes by y-proximity and snap each cluster to its median y."""
    sorted_boxes = sorted(boxes, key=lambda b: -b["y"])
    clusters: list[list[dict]] = []
    current: list[dict] = [sorted_boxes[0]]

    for bx in sorted_boxes[1:]:
        cluster_y = sum(b["y"] for b in current) / len(current)
        cluster_min_h = min(b["height"] for b in current)
        threshold = min(cluster_min_h, bx["height"]) * 0.35
        if abs(cluster_y - bx["y"]) <= threshold:
            current.append(bx)
        else:
            clusters.append(current)
            current = [bx]
    clusters.append(current)

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        ys = sorted(b["y"] for b in cluster)
        median_y = ys[len(ys) // 2]
        for b in cluster:
            b["y"] = median_y


def _emit_line_commands(line_boxes: list[dict]) -> list[str]:
    """Build BT/ET PDF commands for one text line."""
    line_boxes.sort(key=lambda b: b["x"])
    line_font = sum(b["font_size"] for b in line_boxes) / len(line_boxes)

    space_w = pdfmetrics.stringWidth(" ", "Helvetica", line_font)
    if space_w <= 0:
        space_w = line_font * 0.25

    parts: list[str] = []
    for i, bx in enumerate(line_boxes):
        parts.append(bx["text"])
        if i < len(line_boxes) - 1:
            gap = line_boxes[i + 1]["x"] - (bx["x"] + bx["width"])
            num_spaces = max(1, round(gap / space_w)) if gap > 0 else 1
            parts.append(" " * num_spaces)

    line_text = "".join(parts)
    line_x = line_boxes[0]["x"]
    line_y = line_boxes[0]["y"]
    line_width = (line_boxes[-1]["x"] + line_boxes[-1]["width"]) - line_boxes[0]["x"]

    natural_w = pdfmetrics.stringWidth(line_text, "Helvetica", line_font)
    if natural_w > 0 and line_width > 0:
        h_scale = line_width / natural_w * 100.0
    else:
        h_scale = 100.0
    h_scale = max(30.0, min(300.0, h_scale))

    return [
        "BT",
        "3 Tr",
        f"1 0 0 1 {line_x:.2f} {line_y:.2f} Tm",
        f"/FOcr {line_font:.1f} Tf",
        f"{h_scale:.1f} Tz",
        f"({escape_pdf_text(line_text)}) Tj",
        "ET",
    ]


def create_text_layer_commands(
    ocr_results: list[OCRResult],
    img_x: float,
    img_y: float,
    img_width: float,
    img_height: float,
    scale_x: float,
    scale_y: float,
) -> list[str]:
    """Create PDF text layer commands for OCR results."""
    boxes = _build_text_boxes(ocr_results, img_x, img_y, img_height, scale_x, scale_y)
    if not boxes:
        return []

    _snap_baselines(boxes)

    from collections import defaultdict

    lines: dict[float, list[dict]] = defaultdict(list)
    for bx in boxes:
        lines[bx["y"]].append(bx)

    commands: list[str] = ["q"]
    for _y_val, line_boxes in sorted(lines.items(), reverse=True):
        commands.extend(_emit_line_commands(line_boxes))
    commands.append("Q")
    return commands


def append_text_to_page(pdf: pikepdf.Pdf, page: pikepdf.Page, text_commands: list[str]) -> None:
    """Append text layer commands to a PDF page.

    Args:
        pdf: pikepdf Pdf object (owner of the page)
        page: pikepdf Page object to modify
        text_commands: List of PDF content stream commands
    """
    if not text_commands:
        return

    # Ensure page has font resources
    if "/Resources" not in page:
        page["/Resources"] = pikepdf.Dictionary()
    if "/Font" not in page.Resources:
        page.Resources["/Font"] = pikepdf.Dictionary()

    # Add OCR font with unique name to avoid conflicts
    if "/FOcr" not in page.Resources.Font:
        page.Resources.Font["/FOcr"] = pikepdf.Dictionary({
            "/Type": pikepdf.Name("/Font"),
            "/Subtype": pikepdf.Name("/Type1"),
            "/BaseFont": pikepdf.Name("/Helvetica"),
            "/Encoding": pikepdf.Name("/WinAnsiEncoding"),
        })

    # Create new content stream with text layer
    text_content = "\n".join(text_commands)

    # Get existing content
    contents = page.get("/Contents")
    if contents is None:
        new_stream = pikepdf.Stream(pdf, text_content.encode("latin-1", errors="replace"))
        page["/Contents"] = new_stream
    elif isinstance(contents, pikepdf.Array):
        new_stream = pikepdf.Stream(pdf, text_content.encode("latin-1", errors="replace"))
        contents.append(new_stream)
    else:
        new_stream = pikepdf.Stream(pdf, text_content.encode("latin-1", errors="replace"))
        page["/Contents"] = pikepdf.Array([contents, new_stream])


def _collect_q_group(ops: list[tuple], start: int) -> tuple[list[tuple], int]:
    """Collect a full q...Q graphics state group from the ops list.

    Returns:
        (group_ops, end_index) where end_index is the position after the group.
    """
    group: list[tuple] = [ops[start]]
    depth = 1
    j = start + 1
    while j < len(ops) and depth > 0:
        group.append(ops[j])
        g_op = str(ops[j][1])
        if g_op == "q":
            depth += 1
        elif g_op == "Q":
            depth -= 1
        j += 1
    return group, j


def _is_invisible_text_group(group: list[tuple]) -> bool:
    """Check if a q/Q group contains only invisible text (render mode 3)."""
    has_invisible_bt = False
    has_image_or_visible = False
    in_bt = False
    bt_has_3tr = False

    for g_operands, g_operator in group:
        g_op = str(g_operator)
        if g_op == "BT":
            in_bt = True
            bt_has_3tr = False
        elif g_op == "ET":
            if bt_has_3tr:
                has_invisible_bt = True
            in_bt = False
        elif g_op == "Tr" and in_bt and g_operands:
            if int(g_operands[0]) == 3:
                bt_has_3tr = True
        elif g_op == "Do":
            has_image_or_visible = True

    return has_invisible_bt and not has_image_or_visible


def _collect_bt_block(ops: list[tuple], start: int) -> tuple[list[tuple], int]:
    """Collect a top-level BT...ET text block.

    Returns:
        (block_ops, end_index) where end_index is the position after the block.
    """
    group: list[tuple] = [ops[start]]
    j = start + 1
    while j < len(ops) and str(ops[j][1]) != "ET":
        group.append(ops[j])
        j += 1
    if j < len(ops):
        group.append(ops[j])
        j += 1
    return group, j


def _bt_block_is_invisible(group: list[tuple]) -> bool:
    """Check if a BT/ET block uses render mode 3 (invisible text)."""
    return any(str(g_op) == "Tr" and g_ops and int(g_ops[0]) == 3 for g_ops, g_op in group)


def strip_invisible_text(page: pikepdf.Page, pdf: pikepdf.Pdf) -> int:
    """Remove invisible text (render mode 3) from page content stream.

    Parses the content stream and removes q/Q groups or standalone
    BT/ET blocks whose only purpose is invisible text overlay (from
    a previous OCR pass).  Image display commands (cm + Do) and
    visible text are preserved.

    Args:
        page: PDF page to clean.
        pdf: Owner PDF (for creating new stream objects).

    Returns:
        Number of operator groups removed.
    """
    try:
        ops = list(pikepdf.parse_content_stream(page))
    except Exception:
        return 0

    if not ops:
        return 0

    filtered: list[tuple] = []
    removed = 0
    i = 0

    while i < len(ops):
        op = str(ops[i][1])

        if op == "q":
            group, j = _collect_q_group(ops, i)
            if _is_invisible_text_group(group):
                removed += 1
                i = j
                continue
            filtered.extend(group)
            i = j

        elif op == "BT":
            group, j = _collect_bt_block(ops, i)
            if _bt_block_is_invisible(group):
                removed += 1
                i = j
                continue
            filtered.extend(group)
            i = j

        else:
            filtered.append(ops[i])
            i += 1

    if removed > 0:
        new_content = pikepdf.unparse_content_stream(filtered)
        page["/Contents"] = pikepdf.Stream(pdf, new_content)

    return removed


def merge_single_page(
    orig_page: pikepdf.Page,
    text_page: pikepdf.Page,
    original_pdf: pikepdf.Pdf,
    page_num: int,
) -> bool:
    """Merge text layer content into a single original page.

    Directly prepends the text content stream to the original page's
    content streams, preserving all original XObjects (images) intact.
    Any existing invisible text (from a previous OCR pass) is stripped
    first to prevent duplicate overlapping text layers.

    Args:
        orig_page: Original PDF page
        text_page: Text layer PDF page
        original_pdf: Original PDF for copying foreign objects
        page_num: Page number (0-indexed) for logging

    Returns:
        True if merge succeeded, False otherwise
    """
    if "/Contents" not in text_page:
        return True  # No content to merge is not an error

    # Strip existing invisible OCR text from original page
    stripped = strip_invisible_text(orig_page, original_pdf)
    if stripped:
        logger.debug(f"Page {page_num + 1}: stripped {stripped} old OCR text blocks")

    # Extract text layer content streams and copy as foreign objects
    text_streams = extract_content_streams(text_page["/Contents"], original_pdf, copy_foreign=True)

    # Get original content streams (don't copy, they're already in this PDF)
    orig_streams = extract_content_streams(
        orig_page.get("/Contents", []), original_pdf, copy_foreign=False
    )

    # Prepend text layer streams (text goes underneath original content)
    orig_page["/Contents"] = pikepdf.Array(text_streams + orig_streams)

    # Copy fonts from text layer to original page resources
    if "/Resources" in text_page:
        merge_page_fonts(orig_page, text_page["/Resources"], original_pdf)

    logger.debug(
        f"Page {page_num + 1}: merged {len(text_streams)} text streams "
        f"with {len(orig_streams)} original streams"
    )
    return True


def overlay_text_on_original(
    original_pdf_path: Path,
    text_layer_pdf_path: Path,
    output_pdf_path: Path,
) -> None:
    """Overlay text layer PDF on original PDF, preserving everything from original.

    Uses pikepdf to merge the text layer as an underlay while keeping
    all original PDF content (images, metadata, rotation, etc.) intact.

    Args:
        original_pdf_path: Path to the original PDF
        text_layer_pdf_path: Path to the text layer PDF
        output_pdf_path: Path for the merged output PDF
    """
    logger.info("Merging text layer with original PDF...")

    with (
        pikepdf.open(original_pdf_path) as original,
        pikepdf.open(text_layer_pdf_path) as text_layer,
    ):
        if len(original.pages) != len(text_layer.pages):
            logger.warning(
                f"Page count mismatch: original={len(original.pages)}, "
                f"text_layer={len(text_layer.pages)}"
            )

        for page_num, (orig_page, text_page) in enumerate(
            zip(original.pages, text_layer.pages, strict=False)
        ):
            try:
                merge_single_page(orig_page, text_page, original, page_num)
            except Exception as e:
                logger.warning(f"Failed to merge text layer for page {page_num + 1}: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        original.save(
            output_pdf_path,
            object_stream_mode=pikepdf.ObjectStreamMode.preserve,
            stream_decode_level=pikepdf.StreamDecodeLevel.none,
            compress_streams=True,
        )

    logger.info(f"Merged PDF saved: {output_pdf_path}")


def smart_merge_pdfs(
    original_pdf_path: Path,
    text_layer_pdf_path: Path,
    output_pdf_path: Path,
    page_standalone_flags: list[bool],
) -> None:
    """Merge text layer with original PDF using per-page strategy.

    For pages that need standalone mode (geometry changes, appearance effects,
    format changes), the text layer page is used as-is (it contains the
    processed image + text).  For all other pages, the text layer is overlaid
    onto the original page, preserving original image quality.

    Args:
        original_pdf_path: Path to original PDF
        text_layer_pdf_path: Path to text layer PDF (may contain images on some pages)
        output_pdf_path: Path for merged output PDF
        page_standalone_flags: Per-page booleans — True means use standalone
    """
    logger.info("Smart merging text layer with per-page strategy...")

    with (
        pikepdf.open(original_pdf_path) as original,
        pikepdf.open(text_layer_pdf_path) as text_layer,
    ):
        page_count = min(len(original.pages), len(text_layer.pages))
        standalone_count = 0
        overlay_count = 0

        for i in range(page_count):
            needs_standalone = page_standalone_flags[i] if i < len(page_standalone_flags) else False

            if needs_standalone:
                original.pages[i] = text_layer.pages[i]
                standalone_count += 1
            else:
                try:
                    merge_single_page(original.pages[i], text_layer.pages[i], original, i)
                except Exception as e:
                    logger.warning(f"Failed to merge text layer for page {i + 1}: {e}")
                overlay_count += 1

        logger.info(
            f"Smart merge: {overlay_count} overlay pages, {standalone_count} standalone pages"
        )

        original.save(
            output_pdf_path,
            object_stream_mode=pikepdf.ObjectStreamMode.preserve,
            stream_decode_level=pikepdf.StreamDecodeLevel.none,
            compress_streams=True,
        )

    logger.info(f"Smart merged PDF saved: {output_pdf_path}")


def convert_to_pdfa(input_pdf: Path, output_pdf: Path) -> None:
    """Convert PDF to PDF/A-2b format using pikepdf metadata injection.

    Instead of re-rendering the entire PDF through Ghostscript (which
    re-encodes all images causing quality loss and slow processing),
    this approach injects the required PDF/A metadata directly:
      - XMP metadata declaring PDF/A-2b conformance
      - OutputIntents with embedded sRGB ICC profile
      - MarkInfo dictionary

    Images are preserved byte-for-byte, so there is zero quality loss
    and the operation is nearly instantaneous.

    Args:
        input_pdf: Path to source PDF
        output_pdf: Path for PDF/A output
    """
    import shutil

    logger.info("Converting to PDF/A-2b using pikepdf metadata injection...")

    srgb_icc = Path("/usr/share/color/icc/colord/sRGB.icc")
    if not srgb_icc.exists():
        # Fallback to Ghostscript location
        srgb_icc = Path("/usr/share/ghostscript/iccprofiles/srgb.icc")

    if not srgb_icc.exists():
        logger.warning("sRGB ICC profile not found, skipping PDF/A conversion")
        shutil.copy2(input_pdf, output_pdf)
        return

    try:
        icc_data = srgb_icc.read_bytes()

        with pikepdf.open(input_pdf) as pdf:
            # 1. Add MarkInfo (required for PDF/A-2)
            if pikepdf.Name.MarkInfo not in pdf.Root:
                pdf.Root.MarkInfo = pikepdf.Dictionary(Marked=True)
            else:
                pdf.Root.MarkInfo[pikepdf.Name.Marked] = True

            # 2. Create and embed sRGB ICC stream
            icc_stream = pdf.make_stream(icc_data)
            icc_stream[pikepdf.Name.N] = 3  # Number of color components (RGB)

            # 3. Build OutputIntent dictionary
            output_intent = pikepdf.Dictionary(
                Type=pikepdf.Name.OutputIntent,
                S=pikepdf.Name("/GTS_PDFA1"),
                OutputConditionIdentifier=pikepdf.String("sRGB IEC61966-2.1"),
                RegistryName=pikepdf.String("http://www.color.org"),
                Info=pikepdf.String("sRGB IEC61966-2.1"),
                DestOutputProfile=icc_stream,
            )

            # 4. Add OutputIntents array to document catalog
            pdf.Root.OutputIntents = pikepdf.Array([output_intent])

            # 5. Set XMP metadata with PDF/A-2b conformance
            with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                meta["pdfaid:part"] = "2"
                meta["pdfaid:conformance"] = "B"
                # Preserve existing title/author or set sensible defaults
                if "/Title" in pdf.docinfo:
                    meta["dc:title"] = str(pdf.docinfo["/Title"])
                if "/Author" in pdf.docinfo:
                    meta["dc:creator"] = [str(pdf.docinfo["/Author"])]
                meta["xmp:CreatorTool"] = "BigOCRPDF"
                meta["pdf:Producer"] = "BigOCRPDF (pikepdf)"

            # 6. Save preserving all streams as-is (no image re-encoding)
            #    force_version="1.7" sets PDF header to 1.7 (PDF/A-2 requires ≥1.7)
            pdf.save(
                output_pdf,
                object_stream_mode=pikepdf.ObjectStreamMode.preserve,
                stream_decode_level=pikepdf.StreamDecodeLevel.none,
                compress_streams=True,
                force_version="1.7",
            )

        logger.info(f"PDF/A-2b conversion successful: {output_pdf}")

    except Exception as e:
        logger.error(f"PDF/A conversion failed: {e}")
        shutil.copy2(input_pdf, output_pdf)
        logger.warning("Using non-PDF/A output due to conversion failure")
