#!/usr/bin/env python3
"""
BigOcrPdf CLI — comprehensive PDF toolbox from the terminal.

Usage:
    python -m bigocrpdf.cli <command> [options]

Commands:
    ocr         OCR processing (add searchable text layer)
    split       Split PDF by page count or file size
    merge       Merge multiple PDFs into one
    compress    Compress PDF (reduce file size)
    rotate      Rotate pages
    delete      Delete pages
    extract     Extract pages to a new PDF
    insert      Insert pages from another PDF
    reorder     Reorder or reverse pages
    info        Show PDF metadata and page count
    edit        Open interactive GUI editor

Examples:
    # Basic OCR
    bigocrpdf-cli ocr input.pdf -o output.pdf
    bigocrpdf-cli ocr input.pdf -o output.pdf --language ch

    # OCR with options
    bigocrpdf-cli ocr input.pdf -o out.pdf --replace-existing-ocr
    bigocrpdf-cli ocr input.pdf -o out.pdf --no-dewarp --no-scanner
    bigocrpdf-cli ocr input.pdf -o out.pdf --auto-contrast --denoise
    bigocrpdf-cli ocr input.pdf -o out.pdf --pdfa --image-format jpeg

    # Debug preprocessing
    bigocrpdf-cli ocr input.pdf -o out.pdf --dewarp-only --save-preprocessed /tmp/debug

    # Split
    bigocrpdf-cli split input.pdf -o parts/ --pages 5
    bigocrpdf-cli split input.pdf -o parts/ --size 10

    # Merge
    bigocrpdf-cli merge a.pdf b.pdf c.pdf -o merged.pdf

    # Compress
    bigocrpdf-cli compress input.pdf -o small.pdf --quality 50

    # Rotate
    bigocrpdf-cli rotate input.pdf -o out.pdf --angle 90 --pages 1,3,5

    # Info
    bigocrpdf-cli info document.pdf
"""

import argparse
import logging
import sys
import time
from pathlib import Path  # noqa: I001

from bigocrpdf.utils.i18n import _

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _setup_environment():
    """Minimal setup without GTK dependencies."""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from bigocrpdf.utils.python_compat import setup_python_compatibility

    setup_python_compatibility()

    import locale

    try:
        locale.setlocale(locale.LC_ALL, "")
        locale.setlocale(locale.LC_NUMERIC, "C")
    except locale.Error:
        locale.setlocale(locale.LC_ALL, "C")


# ---------------------------------------------------------------------------
# Page range parser (shared)
# ---------------------------------------------------------------------------


def _parse_page_list(text: str) -> list[int]:
    """Parse a page specification string into a sorted list of page numbers.

    Supports: "3", "1-5", "1,3,7", "1-3,7,10-12"

    Args:
        text: Page specification string.

    Returns:
        Sorted list of 1-indexed page numbers.
    """
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                s, e = int(start_s.strip()), int(end_s.strip())
                pages.update(range(s, e + 1))
            else:
                pages.add(int(part))
        except ValueError:
            raise ValueError(
                f"Invalid page specification '{part}'. "
                "Use numbers and ranges like '1-5' or '1,3,7'."
            ) from None
    return sorted(p for p in pages if p >= 1)


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """Parse a range specification into list of (start, end) tuples.

    Supports: "1-5,6-10,11-15"

    Args:
        text: Range specification string.

    Returns:
        List of (start, end) tuples, 1-indexed inclusive.
    """
    ranges: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                ranges.append((int(start_s.strip()), int(end_s.strip())))
            else:
                p = int(part.strip())
                ranges.append((p, p))
        except ValueError:
            raise ValueError(
                f"Invalid range specification '{part}'. "
                "Use numbers and ranges like '1-5' or '1,3,7'."
            ) from None
    return ranges


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    p = argparse.ArgumentParser(
        prog="bigocrpdf-cli",
        description="BigOcrPdf — comprehensive PDF toolbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help=_("Verbose logging (DEBUG)"))

    sub = p.add_subparsers(dest="command", help=_("Available commands"))

    # --- ocr ---
    ocr_p = sub.add_parser("ocr", help=_("OCR processing for PDF files"))
    ocr_p.add_argument("input", type=Path, help=_("Input PDF file"))
    ocr_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    ocr_p.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to process (e.g. '7', '3-10', '1,3,7'). Default: all."),
    )
    ocr_p.add_argument(
        "--language",
        type=str,
        default="latin",
        help=_("OCR language (latin, ch, japan, korean, etc.). Default: latin."),
    )
    ocr_p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=_("Parallel workers (0 = auto). Default: 0."),
    )
    ocr_p.add_argument("--dpi", type=int, default=300, help=_("DPI for extraction (default: 300)"))

    # OCR behavior
    ocr_b = ocr_p.add_argument_group(_("OCR behavior"))
    ocr_b.add_argument(
        "--replace-existing-ocr",
        action="store_true",
        default=False,
        help=_("Re-OCR pages that already have text (default: skip them)."),
    )
    ocr_b.add_argument(
        "--force-full-ocr",
        action="store_true",
        default=False,
        help=_("Process all pages as image-only (for editor-merged files)."),
    )
    ocr_b.add_argument(
        "--server-models",
        action="store_true",
        default=False,
        help=_("Use high-quality server models if available."),
    )

    # Geometric corrections
    ocr_g = ocr_p.add_argument_group(_("Geometric corrections"))
    ocr_g.add_argument("--no-dewarp", action="store_true", help=_("Disable curvature correction"))
    ocr_g.add_argument("--no-deskew", action="store_true", help=_("Disable skew correction"))
    ocr_g.add_argument(
        "--no-perspective", action="store_true", help=_("Disable perspective correction")
    )
    ocr_g.add_argument(
        "--no-orientation", action="store_true", help=_("Disable orientation detection")
    )

    # Image enhancements
    ocr_e = ocr_p.add_argument_group(_("Image enhancements"))
    ocr_e.add_argument(
        "--scanner",
        action="store_true",
        default=None,
        help=_("Enable scanner effect (whitens background)."),
    )
    ocr_e.add_argument("--no-scanner", action="store_true", help=_("Disable scanner effect."))
    ocr_e.add_argument("--auto-contrast", action="store_true", help=_("Enable CLAHE contrast"))
    ocr_e.add_argument("--auto-brightness", action="store_true", help=_("Enable auto brightness"))
    ocr_e.add_argument("--denoise", action="store_true", help=_("Enable denoising"))
    ocr_e.add_argument("--border-clean", action="store_true", help=_("Remove dark borders"))
    ocr_e.add_argument(
        "--vintage",
        action="store_true",
        help=_("Enable vintage look (sepia/BW)."),
    )
    ocr_e.add_argument("--vintage-bw", action="store_true", help=_("Vintage in black & white"))

    # Output options
    ocr_o = ocr_p.add_argument_group(_("Output options"))
    ocr_o.add_argument("--pdfa", action="store_true", help=_("Convert output to PDF/A-2b"))
    ocr_o.add_argument(
        "--image-format",
        type=str,
        default="original",
        choices=["original", "jpeg", "png", "webp"],
        help=_("Image format in output PDF. Default: original."),
    )
    ocr_o.add_argument(
        "--image-quality",
        type=int,
        default=85,
        metavar="Q",
        help=_("Image quality for JPEG/WebP (1-95, default: 85)."),
    )

    # Debugging
    ocr_dbg = ocr_p.add_argument_group(_("Debugging"))
    ocr_dbg.add_argument(
        "--save-preprocessed",
        type=Path,
        default=None,
        help=_("Save preprocessed images to this directory."),
    )
    ocr_dbg.add_argument(
        "--dewarp-only",
        action="store_true",
        help=_("Only run preprocessing (dewarp/deskew), save images, skip OCR."),
    )

    # --- split ---
    split_p = sub.add_parser("split", help=_("Split PDF into smaller files"))
    split_p.add_argument("input", type=Path, help=_("Input PDF file"))
    split_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output directory"))
    split_mode = split_p.add_mutually_exclusive_group(required=True)
    split_mode.add_argument(
        "--pages",
        type=int,
        metavar="N",
        help=_("Split every N pages"),
    )
    split_mode.add_argument(
        "--size",
        type=float,
        metavar="MB",
        help=_("Maximum file size per part in megabytes"),
    )
    split_mode.add_argument(
        "--ranges",
        type=str,
        metavar="RANGES",
        help=_("Explicit ranges (e.g. '1-5,6-10,11-15')"),
    )
    split_p.add_argument(
        "--prefix", type=str, default="", help=_("Filename prefix for output parts")
    )

    # --- merge ---
    merge_p = sub.add_parser("merge", help=_("Merge multiple PDFs into one"))
    merge_p.add_argument("inputs", nargs="+", type=Path, help=_("Input PDF files (in order)"))
    merge_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))

    # --- compress ---
    compress_p = sub.add_parser("compress", help=_("Compress PDF to reduce file size"))
    compress_p.add_argument("input", type=Path, help=_("Input PDF file"))
    compress_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    compress_p.add_argument(
        "--quality",
        type=int,
        default=60,
        metavar="Q",
        help=_("JPEG quality for images (1-95, default: 60)"),
    )
    compress_p.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="DPI",
        help=_("Target DPI for images (default: 150)"),
    )

    # --- rotate ---
    rotate_p = sub.add_parser("rotate", help=_("Rotate pages in a PDF"))
    rotate_p.add_argument("input", type=Path, help=_("Input PDF file"))
    rotate_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    rotate_p.add_argument(
        "--angle",
        type=int,
        required=True,
        choices=[90, 180, 270],
        help=_("Rotation angle in degrees (clockwise)"),
    )
    rotate_p.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to rotate (e.g. '1,3,5' or '1-5'). Default: all."),
    )

    # --- delete ---
    delete_p = sub.add_parser("delete", help=_("Remove pages from a PDF"))
    delete_p.add_argument("input", type=Path, help=_("Input PDF file"))
    delete_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    delete_p.add_argument(
        "--pages",
        type=str,
        required=True,
        help=_("Pages to delete (e.g. '3,5,7' or '2-4')"),
    )

    # --- extract ---
    extract_p = sub.add_parser("extract", help=_("Extract pages to a new PDF"))
    extract_p.add_argument("input", type=Path, help=_("Input PDF file"))
    extract_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    extract_p.add_argument(
        "--pages",
        type=str,
        required=True,
        help=_("Pages to extract (e.g. '3-5' or '1,3,7')"),
    )

    # --- insert ---
    insert_p = sub.add_parser("insert", help=_("Insert pages from another PDF"))
    insert_p.add_argument("input", type=Path, help=_("Target PDF file"))
    insert_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    insert_p.add_argument(
        "--from",
        dest="insert_from",
        type=Path,
        required=True,
        metavar="FILE",
        help=_("PDF file to insert pages from"),
    )
    insert_p.add_argument(
        "--at",
        type=int,
        default=0,
        metavar="POS",
        help=_("Insert at this page position (1-indexed, 0 = append). Default: append."),
    )
    insert_p.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to insert from the source (e.g. '1-3'). Default: all."),
    )

    # --- reorder ---
    reorder_p = sub.add_parser("reorder", help=_("Reorder or reverse pages"))
    reorder_p.add_argument("input", type=Path, help=_("Input PDF file"))
    reorder_p.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    reorder_grp = reorder_p.add_mutually_exclusive_group(required=True)
    reorder_grp.add_argument(
        "--order",
        type=str,
        metavar="ORDER",
        help=_("New page order (e.g. '3,1,2,5,4')"),
    )
    reorder_grp.add_argument(
        "--reverse",
        action="store_true",
        help=_("Reverse the page order"),
    )

    # --- info ---
    info_p = sub.add_parser("info", help=_("Show PDF metadata and page count"))
    info_p.add_argument("input", type=Path, help=_("Input PDF file"))

    # --- edit ---
    edit_p = sub.add_parser("edit", help=_("Open interactive GUI editor"))
    edit_p.add_argument("input", type=Path, help=_("PDF file to edit"))

    return p


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_ocr(args, logger) -> int:
    """Handle the 'ocr' command."""
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

    # Determine scanner state: explicit flag > default (True)
    scanner_enabled = True  # default
    if args.no_scanner:
        scanner_enabled = False
    elif args.scanner:
        scanner_enabled = True

    # Enable preprocessing master switch if any enhancement is requested
    enable_preprocessing = any([
        args.auto_contrast,
        args.auto_brightness,
        args.denoise,
    ])

    config = OCRConfig(
        language=args.language,
        dpi=args.dpi,
        workers=args.workers,
        use_server_models=args.server_models,
        # Geometric
        enable_baseline_dewarp=not args.no_dewarp,
        enable_deskew=not args.no_deskew,
        enable_perspective_correction=not args.no_perspective,
        enable_orientation_detection=not args.no_orientation,
        # Enhancements
        enable_scanner_effect=scanner_enabled,
        enable_preprocessing=enable_preprocessing,
        enable_auto_contrast=args.auto_contrast,
        enable_auto_brightness=args.auto_brightness,
        enable_denoise=args.denoise,
        enable_border_clean=args.border_clean,
        enable_vintage_look=args.vintage or args.vintage_bw,
        vintage_bw=args.vintage_bw,
        # Output
        convert_to_pdfa=args.pdfa,
        image_export_format=args.image_format,
        image_export_quality=args.image_quality,
        # Behavior
        replace_existing_ocr=args.replace_existing_ocr,
        force_full_ocr=args.force_full_ocr,
    )

    # Parse page range
    page_range = None
    page_set = None
    if args.pages:
        pages_list = _parse_page_list(args.pages)
        if len(pages_list) == 1:
            page_range = (pages_list[0], pages_list[0])
        elif pages_list:
            # Check if continuous range
            if pages_list == list(range(pages_list[0], pages_list[-1] + 1)):
                page_range = (pages_list[0], pages_list[-1])
            else:
                page_set = set(pages_list)
                page_range = (min(pages_list), max(pages_list))

    if args.dewarp_only:
        return _run_dewarp_only(args, config, page_range, logger, page_set=page_set)

    return _run_full_ocr(args, config, page_range, logger)


def _cmd_split(args, logger) -> int:
    """Handle the 'split' command."""
    from bigocrpdf.services.pdf_operations import split_by_pages, split_by_ranges, split_by_size

    if args.pages is not None:
        if args.pages < 1:
            print("Error: --pages must be at least 1", file=sys.stderr)
            return 1
        result = split_by_pages(args.input, args.output, args.pages, prefix=args.prefix)
    elif args.size is not None:
        if args.size <= 0:
            print("Error: --size must be greater than 0", file=sys.stderr)
            return 1
        result = split_by_size(args.input, args.output, args.size, prefix=args.prefix)
    elif args.ranges:
        ranges = _parse_ranges(args.ranges)
        result = split_by_ranges(args.input, args.output, ranges, prefix=args.prefix)
    else:
        print("Error: specify --pages, --size, or --ranges", file=sys.stderr)
        return 1

    print(f"Split into {result.parts} parts ({result.total_pages} total pages)")
    for f in result.output_files:
        print(f"  → {f}")
    return 0


def _cmd_merge(args, logger) -> int:
    """Handle the 'merge' command."""
    from bigocrpdf.services.pdf_operations import merge_pdfs

    for p in args.inputs:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1

    result = merge_pdfs([str(p) for p in args.inputs], str(args.output))
    if result.success:
        print(f"Merged: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_compress(args, logger) -> int:
    """Handle the 'compress' command."""
    from bigocrpdf.services.pdf_operations import compress_pdf

    result = compress_pdf(
        args.input,
        args.output,
        image_quality=args.quality,
        image_dpi=args.dpi,
    )
    if result.success:
        print(result.message)
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_rotate(args, logger) -> int:
    """Handle the 'rotate' command."""
    from bigocrpdf.services.pdf_operations import get_pdf_info, rotate_pages

    if args.pages:
        pages = _parse_page_list(args.pages)
    else:
        info = get_pdf_info(args.input)
        pages = list(range(1, info.page_count + 1))

    result = rotate_pages(args.input, args.output, pages, args.angle)
    if result.success:
        print(f"Rotated: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_delete(args, logger) -> int:
    """Handle the 'delete' command."""
    from bigocrpdf.services.pdf_operations import delete_pages

    pages = _parse_page_list(args.pages)
    result = delete_pages(args.input, args.output, pages)
    if result.success:
        print(f"Deleted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_extract(args, logger) -> int:
    """Handle the 'extract' command."""
    from bigocrpdf.services.pdf_operations import extract_pages

    pages = _parse_page_list(args.pages)
    result = extract_pages(args.input, args.output, pages)
    if result.success:
        print(f"Extracted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_insert(args, logger) -> int:
    """Handle the 'insert' command."""
    from bigocrpdf.services.pdf_operations import insert_pages

    if not args.insert_from.exists():
        print(f"Error: {args.insert_from} not found", file=sys.stderr)
        return 1

    source_pages = _parse_page_list(args.pages) if args.pages else None
    result = insert_pages(
        args.input,
        args.insert_from,
        args.output,
        at_page=args.at,
        source_pages=source_pages,
    )
    if result.success:
        print(f"Inserted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_reorder(args, logger) -> int:
    """Handle the 'reorder' command."""
    from bigocrpdf.services.pdf_operations import reorder_pages, reverse_pages

    if args.reverse:
        result = reverse_pages(args.input, args.output)
    else:
        order = [int(x.strip()) for x in args.order.split(",")]
        result = reorder_pages(args.input, args.output, order)

    if result.success:
        print(f"Reordered: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_info(args, _logger) -> int:
    """Handle the 'info' command."""
    from bigocrpdf.services.pdf_operations import get_pdf_info

    info = get_pdf_info(str(args.input))
    print(f"File:       {info.path}")
    print(f"Pages:      {info.page_count}")
    print(f"Size:       {info.file_size_mb:.2f} MB ({info.file_size_bytes:,} bytes)")
    print(f"Version:    PDF {info.pdf_version}")
    print(f"Encrypted:  {'Yes' if info.encrypted else 'No'}")
    if info.title:
        print(f"Title:      {info.title}")
    if info.author:
        print(f"Author:     {info.author}")
    if info.creator:
        print(f"Creator:    {info.creator}")
    return 0


def _cmd_edit(args, logger) -> int:
    """Handle the 'edit' command — launch GUI editor directly."""
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import Adw

        from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

        app = Adw.Application(application_id="com.biglinux.bigocrpdf.editor")

        def on_activate(_app):
            win = PDFEditorWindow(
                application=_app,
                pdf_path=str(args.input.resolve()),
                on_save_callback=lambda doc: _standalone_save(doc, args.input, logger),
            )
            win.present()

        app.connect("activate", on_activate)
        return app.run([])

    except ImportError as e:
        print(f"Error: GTK4/libadwaita required for GUI editor: {e}", file=sys.stderr)
        return 1


def _standalone_save(doc, original_path, logger):
    """Save callback for standalone editor mode."""
    import os
    import shutil
    import tempfile

    from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

    output = str(original_path.resolve())
    fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_edit_")
    os.close(fd)

    try:
        if apply_changes_to_pdf(doc, tmp):
            shutil.move(tmp, output)
            logger.info("Saved edited PDF: %s", output)
            print(f"Saved: {output}")
        else:
            print("Error: failed to save PDF", file=sys.stderr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# OCR helpers (moved from old cli.py)
# ---------------------------------------------------------------------------


def _run_dewarp_only(args, config, page_range, logger, page_set=None) -> int:
    """Run only preprocessing (dewarp/deskew) and save images."""
    import cv2
    import numpy as np

    save_dir = args.save_preprocessed or Path("/tmp/bigocrpdf_debug")
    save_dir.mkdir(parents=True, exist_ok=True)

    from bigocrpdf.services.rapidocr_service.pdf_extractor import PDFImageExtractor
    from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor

    extractor = PDFImageExtractor(config.dpi)
    preprocessor = ImagePreprocessor(config)

    import pikepdf

    page_rotations = []
    with pikepdf.open(args.input) as pdf:
        for page in pdf.pages:
            rot = int(page.get("/Rotate", 0)) % 360
            page_rotations.append(rot)

    effective_range = page_range
    if page_set:
        effective_range = (min(page_set), max(page_set))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger.info(f"Extracting images from {args.input}...")
        images = extractor.extract(args.input, tmp_path, page_range=effective_range)

        for i, img_path in enumerate(images):
            if img_path is None:
                continue
            page_num = (effective_range[0] if effective_range else 1) + i

            if page_set and page_num not in page_set:
                continue

            from PIL import Image as PILImage

            pil_img = PILImage.open(img_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if page_num <= len(page_rotations):
                rotation = page_rotations[page_num - 1]
                if rotation == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if rotation != 0:
                    logger.info(f"Page {page_num}: applied /Rotate={rotation}°")

            logger.info(f"Page {page_num}: {img.shape[1]}×{img.shape[0]} px")

            orig_path = save_dir / f"page_{page_num:02d}_original.png"
            cv2.imwrite(str(orig_path), img)
            logger.info(f"  Saved original: {orig_path}")

            t0 = time.perf_counter()
            processed = preprocessor.process(img)
            elapsed = time.perf_counter() - t0

            proc_path = save_dir / f"page_{page_num:02d}_processed.png"
            cv2.imwrite(str(proc_path), processed)
            logger.info(f"  Saved processed: {proc_path} ({elapsed:.2f}s)")

            if img.shape != processed.shape:
                logger.info(
                    f"  Geometry: {img.shape[1]}×{img.shape[0]} → "
                    f"{processed.shape[1]}×{processed.shape[0]}"
                )

    logger.info(f"All preprocessed images saved to {save_dir}")
    return 0


def _run_full_ocr(args, config, page_range, logger) -> int:
    """Run the full OCR pipeline."""
    from bigocrpdf.services.rapidocr_service.backend import ProfessionalPDFOCR

    if page_range:
        config.page_range = page_range

    ocr = ProfessionalPDFOCR(config)

    def progress_cb(current, total, message):
        print(f"\r[{current}/{total}] {message}", end="", flush=True)

    logger.info(f"Processing {args.input} → {args.output}")
    logger.info(
        f"Config: lang={config.language}, dpi={config.dpi}, workers={config.workers}, "
        f"dewarp={config.enable_baseline_dewarp}, deskew={config.enable_deskew}, "
        f"perspective={config.enable_perspective_correction}, "
        f"scanner={config.enable_scanner_effect}, "
        f"replace_ocr={config.replace_existing_ocr}, "
        f"force_full={config.force_full_ocr}"
    )

    t0 = time.perf_counter()
    try:
        stats = ocr.process(
            args.input,
            args.output,
            progress_callback=progress_cb,
        )
        elapsed = time.perf_counter() - t0
        print()  # newline after progress

        pages = stats.pages_processed
        confidence = stats.average_confidence
        logger.info(f"Done: {pages} pages, {confidence:.1f}% avg confidence, {elapsed:.1f}s total")

        if args.save_preprocessed:
            logger.info("Note: --save-preprocessed requires --dewarp-only mode")

        return 0

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print()
        logger.error(f"Fatal error after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _setup_environment()

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("bigocrpdf.cli")

    # Validate input file existence (except merge which has 'inputs')
    if hasattr(args, "input") and args.input and not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    # Dispatch to command handler
    handlers = {
        "ocr": _cmd_ocr,
        "split": _cmd_split,
        "merge": _cmd_merge,
        "compress": _cmd_compress,
        "rotate": _cmd_rotate,
        "delete": _cmd_delete,
        "extract": _cmd_extract,
        "insert": _cmd_insert,
        "reorder": _cmd_reorder,
        "info": _cmd_info,
        "edit": _cmd_edit,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args, logger)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
