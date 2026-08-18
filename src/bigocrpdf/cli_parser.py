"""Argument parser and page-range parsing for the BigOcrPdf CLI."""

import argparse
from pathlib import Path

from bigocrpdf.config import APP_VERSION
from bigocrpdf.services.rapidocr_service.config import DEFAULT_DETECTION_FULL_RESOLUTION
from bigocrpdf.utils.i18n import _


def _parse_page_list(text: str) -> list[int]:
    """Parse a page specification string into sorted 1-indexed numbers."""
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                s, e = int(start_s.strip()), int(end_s.strip())
                if s < 1 or e < s:
                    raise ValueError
                pages.update(range(s, e + 1))
            else:
                page = int(part)
                if page < 1:
                    raise ValueError
                pages.add(page)
        except ValueError:
            raise ValueError(
                f"Invalid page specification '{part}'. "
                "Use numbers and ranges like '1-5' or '1,3,7'."
            ) from None
    return sorted(pages)


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """Parse a range specification into 1-indexed inclusive tuples."""
    ranges: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s.strip()), int(end_s.strip())
            else:
                start = end = int(part.strip())
            if start < 1 or end < start:
                raise ValueError
            ranges.append((start, end))
        except ValueError:
            raise ValueError(
                f"Invalid range specification '{part}'. "
                "Use numbers and ranges like '1-5' or '1,3,7'."
            ) from None
    return ranges


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="bigocrpdf-cli",
        description="BigOcrPdf - comprehensive PDF toolbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
        help=_("Print version information and exit"),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help=_("Verbose logging (DEBUG)"))
    subparsers = parser.add_subparsers(dest="command", help=_("Available commands"))
    _add_ocr_parser(subparsers)
    _add_pdf_operation_parsers(subparsers)
    _add_export_parsers(subparsers)
    _add_editor_parser(subparsers)
    return parser


def _add_ocr_parser(subparsers: argparse._SubParsersAction) -> None:
    ocr_parser = subparsers.add_parser("ocr", help=_("OCR processing for PDF files"))
    ocr_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    ocr_parser.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))
    ocr_parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to process (e.g. '7', '3-10', '1,3,7'). Default: all."),
    )
    ocr_parser.add_argument(
        "--workers", type=int, default=0, help=_("Parallel workers (0 = auto). Default: 0.")
    )
    ocr_parser.add_argument(
        "--dpi", type=int, default=300, help=_("DPI for extraction (default: 300)")
    )
    _add_ocr_behavior_options(ocr_parser)
    _add_ocr_geometry_options(ocr_parser)
    _add_ocr_image_options(ocr_parser)
    _add_ocr_output_options(ocr_parser)
    _add_ocr_debug_options(ocr_parser)


def _add_ocr_behavior_options(ocr_parser: argparse.ArgumentParser) -> None:
    behavior = ocr_parser.add_argument_group(_("OCR behavior"))
    behavior.add_argument(
        "--replace-existing-ocr",
        action="store_true",
        default=False,
        help=_("Re-OCR pages that already have text (default: skip them)."),
    )
    behavior.add_argument(
        "--force-full-ocr",
        action="store_true",
        default=False,
        help=_("Process all pages as image-only (for editor-merged files)."),
    )
    _add_ocr_model_options(behavior)
    _add_ocr_gpu_options(behavior)


def _add_ocr_model_options(behavior: argparse._ArgumentGroup) -> None:
    behavior.add_argument(
        "--model-type", default="small", help=_("RapidOCR model size/type (default: small).")
    )
    behavior.add_argument(
        "--engine-type",
        default="openvino",
        choices=["openvino", "onnxruntime"],
        help=_("RapidOCR CPU inference engine (default: openvino)."),
    )
    behavior.add_argument(
        "--pdf-mode",
        default="auto",
        choices=["ocr", "geometric", "auto", "auto_verified"],
        help=_("PDF text handling mode (default: auto)."),
    )
    behavior.add_argument(
        "--rec-batch-num",
        type=int,
        default=1,
        help=_("RapidOCR recognition batch size (default: 1)."),
    )
    behavior.add_argument(
        "--use-textline-cls",
        action="store_true",
        default=False,
        help=_("Enable RapidOCR text-line orientation classifier."),
    )
    full_resolution = behavior.add_mutually_exclusive_group()
    full_resolution.add_argument(
        "--full-resolution",
        dest="full_resolution",
        action="store_true",
        default=DEFAULT_DETECTION_FULL_RESOLUTION,
        help=_("Detect text on the page as rendered, without letting RapidOCR downscale it."),
    )
    full_resolution.add_argument(
        "--no-full-resolution",
        dest="full_resolution",
        action="store_false",
        help=_("Let RapidOCR downscale the page before detection: faster, less memory."),
    )


def _add_ocr_gpu_options(behavior: argparse._ArgumentGroup) -> None:
    behavior.add_argument(
        "--gpu-backend",
        default="off",
        choices=["off", "auto", "paddle", "torch", "tensorrt", "onnxruntime_cuda_experimental"],
        help=_("Experimental GPU backend (default: off)."),
    )
    behavior.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help=_("GPU device id for experimental GPU backends."),
    )
    behavior.add_argument(
        "--gpu-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_("Use FP16 for supported experimental GPU backends."),
    )
    behavior.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help=_("Fail instead of falling back to CPU when GPU initialization fails."),
    )


def _add_ocr_geometry_options(ocr_parser: argparse.ArgumentParser) -> None:
    geometry = ocr_parser.add_argument_group(_("Geometric corrections"))
    geometry.add_argument(
        "--no-dewarp", action="store_true", help=_("Disable curvature correction")
    )
    geometry.add_argument("--no-deskew", action="store_true", help=_("Disable skew correction"))
    geometry.add_argument(
        "--no-perspective", action="store_true", help=_("Disable perspective correction")
    )
    geometry.add_argument(
        "--no-orientation", action="store_true", help=_("Disable orientation detection")
    )


def _add_ocr_image_options(ocr_parser: argparse.ArgumentParser) -> None:
    image = ocr_parser.add_argument_group(_("Image enhancements"))
    image.add_argument(
        "--scanner",
        action="store_true",
        default=None,
        help=_("Enable scanner effect (whitens background)."),
    )
    image.add_argument("--no-scanner", action="store_true", help=_("Disable scanner effect."))
    image.add_argument("--auto-contrast", action="store_true", help=_("Enable CLAHE contrast"))
    image.add_argument("--auto-brightness", action="store_true", help=_("Enable auto brightness"))
    image.add_argument("--denoise", action="store_true", help=_("Enable denoising"))
    image.add_argument("--border-clean", action="store_true", help=_("Remove dark borders"))
    image.add_argument("--vintage", action="store_true", help=_("Enable vintage look (sepia/BW)."))
    image.add_argument("--vintage-bw", action="store_true", help=_("Vintage in black & white"))


def _add_ocr_output_options(ocr_parser: argparse.ArgumentParser) -> None:
    output = ocr_parser.add_argument_group(_("Output options"))
    pdfa = output.add_mutually_exclusive_group()
    pdfa.add_argument(
        "--pdfa",
        dest="pdfa",
        action="store_true",
        default=True,
        help=_("Convert output to PDF/A-2b (default)"),
    )
    pdfa.add_argument(
        "--no-pdfa",
        dest="pdfa",
        action="store_false",
        help=_("Create a regular PDF instead of PDF/A-2b"),
    )
    output.add_argument(
        "--page-layout",
        type=str,
        default="default",
        choices=["default", "single", "continuous", "two_page"],
        help=_("Viewer page layout (/PageLayout) in output PDF. Default: default."),
    )
    output.add_argument(
        "--image-format",
        type=str,
        default="original",
        choices=["original", "jpeg", "png", "webp"],
        help=_("Image format in output PDF. Default: original."),
    )
    output.add_argument(
        "--image-quality",
        type=int,
        default=85,
        metavar="Q",
        help=_("Image quality for JPEG/WebP (1-95, default: 85)."),
    )
    output.add_argument(
        "--sidecar-json",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help=_(
            "Also write structured OCR data (text, word boxes, confidence, layout) "
            "as JSON. Nothing is written without this option. If FILE is omitted, "
            "the name is the output PDF with a .bigocr.json suffix."
        ),
    )


def _add_ocr_debug_options(ocr_parser: argparse.ArgumentParser) -> None:
    debug = ocr_parser.add_argument_group(_("Debugging"))
    debug.add_argument(
        "--save-preprocessed",
        type=Path,
        default=None,
        help=_("Save preprocessed images to this directory."),
    )
    debug.add_argument(
        "--dewarp-only",
        action="store_true",
        help=_("Only run preprocessing (dewarp/deskew), save images, skip OCR."),
    )


def _add_pdf_operation_parsers(subparsers: argparse._SubParsersAction) -> None:
    _add_split_parser(subparsers)
    _add_merge_parser(subparsers)
    _add_compress_parser(subparsers)
    _add_rotate_parser(subparsers)
    _add_delete_parser(subparsers)
    _add_extract_parser(subparsers)
    _add_insert_parser(subparsers)
    _add_reorder_parser(subparsers)
    info_parser = subparsers.add_parser("info", help=_("Show PDF metadata and page count"))
    info_parser.add_argument("input", type=Path, help=_("Input PDF file"))


def _add_split_parser(subparsers: argparse._SubParsersAction) -> None:
    split_parser = subparsers.add_parser("split", help=_("Split PDF into smaller files"))
    split_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    split_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output directory")
    )
    split_mode = split_parser.add_mutually_exclusive_group(required=True)
    split_mode.add_argument("--pages", type=int, metavar="N", help=_("Split every N pages"))
    split_mode.add_argument(
        "--size", type=float, metavar="MB", help=_("Maximum file size per part in megabytes")
    )
    split_mode.add_argument(
        "--ranges", type=str, metavar="RANGES", help=_("Explicit ranges (e.g. '1-5,6-10,11-15')")
    )
    split_parser.add_argument(
        "--prefix", type=str, default="", help=_("Filename prefix for output parts")
    )


def _add_merge_parser(subparsers: argparse._SubParsersAction) -> None:
    merge_parser = subparsers.add_parser("merge", help=_("Merge multiple PDFs into one"))
    merge_parser.add_argument("inputs", nargs="+", type=Path, help=_("Input PDF files (in order)"))
    merge_parser.add_argument("-o", "--output", type=Path, required=True, help=_("Output PDF file"))


def _add_compress_parser(subparsers: argparse._SubParsersAction) -> None:
    compress_parser = subparsers.add_parser("compress", help=_("Compress PDF to reduce file size"))
    compress_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    compress_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    compress_parser.add_argument(
        "--quality",
        type=int,
        default=60,
        metavar="Q",
        help=_("JPEG quality for images (1-95, default: 60)"),
    )
    compress_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="DPI",
        help=_("Target DPI for images (default: 150)"),
    )


def _add_rotate_parser(subparsers: argparse._SubParsersAction) -> None:
    rotate_parser = subparsers.add_parser("rotate", help=_("Rotate pages in a PDF"))
    rotate_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    rotate_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    rotate_parser.add_argument(
        "--angle",
        type=int,
        required=True,
        choices=[90, 180, 270],
        help=_("Rotation angle in degrees (clockwise)"),
    )
    rotate_parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to rotate (e.g. '1,3,5' or '1-5'). Default: all."),
    )


def _add_delete_parser(subparsers: argparse._SubParsersAction) -> None:
    delete_parser = subparsers.add_parser("delete", help=_("Remove pages from a PDF"))
    delete_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    delete_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    delete_parser.add_argument(
        "--pages", type=str, required=True, help=_("Pages to delete (e.g. '3,5,7' or '2-4')")
    )


def _add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    extract_parser = subparsers.add_parser("extract", help=_("Extract pages to a new PDF"))
    extract_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    extract_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    extract_parser.add_argument(
        "--pages", type=str, required=True, help=_("Pages to extract (e.g. '3-5' or '1,3,7')")
    )


def _add_insert_parser(subparsers: argparse._SubParsersAction) -> None:
    insert_parser = subparsers.add_parser("insert", help=_("Insert pages from another PDF"))
    insert_parser.add_argument("input", type=Path, help=_("Target PDF file"))
    insert_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    insert_parser.add_argument(
        "--from",
        dest="insert_from",
        type=Path,
        required=True,
        metavar="FILE",
        help=_("PDF file to insert pages from"),
    )
    insert_parser.add_argument(
        "--at",
        type=int,
        default=0,
        metavar="POS",
        help=_("Insert at this page position (1-indexed, 0 = append). Default: append."),
    )
    insert_parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help=_("Pages to insert from the source (e.g. '1-3'). Default: all."),
    )


def _add_reorder_parser(subparsers: argparse._SubParsersAction) -> None:
    reorder_parser = subparsers.add_parser("reorder", help=_("Reorder or reverse pages"))
    reorder_parser.add_argument("input", type=Path, help=_("Input PDF file"))
    reorder_parser.add_argument(
        "-o", "--output", type=Path, required=True, help=_("Output PDF file")
    )
    reorder_group = reorder_parser.add_mutually_exclusive_group(required=True)
    reorder_group.add_argument(
        "--order", type=str, metavar="ORDER", help=_("New page order (e.g. '3,1,2,5,4')")
    )
    reorder_group.add_argument("--reverse", action="store_true", help=_("Reverse the page order"))


def _add_export_parsers(subparsers: argparse._SubParsersAction) -> None:
    input_help = _("Input PDF file (must have text layer)")
    odf_parser = _add_export_output_parser(
        subparsers,
        command="export-odf",
        command_help=_("Export OCR'd PDF as ODF document"),
        input_help=input_help,
        output_help=_("Output ODF file (default: same name as input with .odt)"),
    )
    odf_parser.add_argument(
        "--preserve-text-layout",
        action="store_true",
        help=_("Preserve Text Layout"),
    )
    _add_export_output_parser(
        subparsers,
        command="export-txt",
        command_help=_("Export OCR'd PDF as formatted text"),
        input_help=input_help,
        output_help=_("Output text file (default: same name as input with .txt)"),
    )
    md_parser = _add_export_output_parser(
        subparsers,
        command="export-md",
        command_help=_("Export OCR'd PDF as Markdown"),
        input_help=input_help,
        output_help=_("Output Markdown file (default: same name as input with .md)"),
    )
    md_parser.add_argument(
        "--front-matter",
        action="store_true",
        help=_("Prepend YAML front-matter (title, source, page count, date)."),
    )


def _add_export_output_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    command_help: str,
    input_help: str,
    output_help: str,
) -> argparse.ArgumentParser:
    export_parser = subparsers.add_parser(command, help=command_help)
    export_parser.add_argument("input", type=Path, help=input_help)
    export_parser.add_argument("-o", "--output", type=Path, default=None, help=output_help)
    export_parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="FILE",
        help=_(
            "Export from structured OCR JSON written earlier by "
            "'ocr --sidecar-json', instead of reading the PDF text layer."
        ),
    )
    return export_parser


def _add_editor_parser(subparsers: argparse._SubParsersAction) -> None:
    edit_parser = subparsers.add_parser("edit", help=_("Open interactive GUI editor"))
    edit_parser.add_argument("input", type=Path, help=_("PDF file to edit"))
