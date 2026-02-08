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


def create_text_layer_commands(
    ocr_results: list[OCRResult],
    img_x: float,
    img_y: float,
    img_width: float,
    img_height: float,
    scale_x: float,
    scale_y: float,
) -> list[str]:
    """Create PDF text layer commands for OCR results.

    Args:
        ocr_results: List of OCR results with text and box coordinates
        img_x: X position of image in PDF points (from left)
        img_y: Y position of image in PDF points (from bottom)
        img_width: Width of image in PDF points
        img_height: Height of image in PDF points
        scale_x: Scale factor from image pixels to PDF points (X)
        scale_y: Scale factor from image pixels to PDF points (Y)

    Returns:
        List of PDF content stream commands to add invisible text
    """
    commands: list[str] = []
    commands.append("q")  # Save graphics state
    # Use render mode 3 for truly invisible text (searchable but not visible)
    commands.append("3 Tr")

    for result in ocr_results:
        if not result.text.strip():
            continue

        # Get bounding box (4 points)
        box = result.box
        min_x = min(p[0] for p in box)
        max_x = max(p[0] for p in box)
        min_y = min(p[1] for p in box)
        max_y = max(p[1] for p in box)

        # Calculate text dimensions in PDF points
        text_width_pts = (max_x - min_x) * scale_x
        text_height_pts = (max_y - min_y) * scale_y

        # Calculate font size based on height (85% to match TextLayerRenderer)
        font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, text_height_pts * FONT_SIZE_SCALE_FACTOR))

        # Calculate horizontal scale to make text fit the width
        try:
            actual_text_width = pdfmetrics.stringWidth(result.text, "Helvetica", font_size)
            if actual_text_width > 0:
                h_scale = (text_width_pts / actual_text_width) * 100
            else:
                h_scale = 100
        except Exception:
            # Fallback: estimate based on average character width
            estimated_char_width = font_size * 0.55
            estimated_text_width = len(result.text) * estimated_char_width
            if estimated_text_width > 0:
                h_scale = (text_width_pts / estimated_text_width) * 100
            else:
                h_scale = 100

        # Clamp horizontal scale to reasonable range (50% - 200%)
        h_scale = max(50, min(200, h_scale))

        # Transform to PDF coordinates
        pdf_x = img_x + min_x * scale_x

        # Position at bottom of OCR box with baseline adjustment
        baseline_offset = text_height_pts * 0.15
        pdf_y = img_y + img_height - (max_y * scale_y) + baseline_offset

        # Escape text for PDF
        escaped_text = escape_pdf_text(result.text)

        # Add text command with horizontal scaling
        commands.append("BT")
        commands.append(f"/FOcr {font_size:.1f} Tf")
        commands.append(f"{h_scale:.1f} Tz")
        commands.append(f"{pdf_x:.2f} {pdf_y:.2f} Td")
        commands.append(f"({escaped_text}) Tj")
        commands.append("ET")

    commands.append("Q")  # Restore graphics state
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
        page.Resources.Font["/FOcr"] = pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
                "/Encoding": pikepdf.Name("/WinAnsiEncoding"),
            }
        )

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


def merge_single_page(
    orig_page: pikepdf.Page,
    text_page: pikepdf.Page,
    original_pdf: pikepdf.Pdf,
    page_num: int,
) -> bool:
    """Merge text layer content into a single original page.

    Directly prepends the text content stream to the original page's
    content streams, preserving all original XObjects (images) intact.

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

    # Extract text layer content streams and copy as foreign objects
    text_streams = extract_content_streams(
        text_page["/Contents"], text_page, original_pdf, copy_foreign=True
    )

    # Get original content streams (don't copy, they're already in this PDF)
    orig_streams = extract_content_streams(
        orig_page.get("/Contents", []), orig_page, original_pdf, copy_foreign=False
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

    original = pikepdf.open(original_pdf_path)
    text_layer = pikepdf.open(text_layer_pdf_path)

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

            traceback.print_exc()

    original.save(
        output_pdf_path,
        object_stream_mode=pikepdf.ObjectStreamMode.preserve,
        stream_decode_level=pikepdf.StreamDecodeLevel.none,
        compress_streams=True,
    )
    original.close()
    text_layer.close()
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

    original = pikepdf.open(original_pdf_path)
    text_layer = pikepdf.open(text_layer_pdf_path)

    page_count = min(len(original.pages), len(text_layer.pages))
    standalone_count = 0
    overlay_count = 0

    for i in range(page_count):
        needs_standalone = page_standalone_flags[i] if i < len(page_standalone_flags) else False

        if needs_standalone:
            # Standalone: replace original page with text_layer page
            # (text_layer page has processed image + invisible text)
            original.pages[i] = text_layer.pages[i]
            standalone_count += 1
        else:
            # No changes needed: overlay text onto original page
            try:
                merge_single_page(original.pages[i], text_layer.pages[i], original, i)
            except Exception as e:
                logger.warning(f"Failed to merge text layer for page {i + 1}: {e}")
            overlay_count += 1

    logger.info(f"Smart merge: {overlay_count} overlay pages, {standalone_count} standalone pages")

    original.save(
        output_pdf_path,
        object_stream_mode=pikepdf.ObjectStreamMode.preserve,
        stream_decode_level=pikepdf.StreamDecodeLevel.none,
        compress_streams=True,
    )
    original.close()
    text_layer.close()
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
