#!/usr/bin/env python3
"""
CLI Script for testing ODF export quality.

Usage:
    python -m bigocrpdf.scripts.test_odf_export input.pdf output.odt [--verbose]

This script extracts OCR boxes from a PDF with text layer and exports to ODF,
allowing automated quality testing.
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

from bigocrpdf.services.rapidocr_service.config import OCRBoxData

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_pdf_page_dimensions(pdf_path: str) -> list[tuple[float, float]]:
    """Get dimensions (width, height) in points for each page using pdfinfo."""
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse "Page size:" line (format: "595.276 x 841.89 pts (A4)")
        for line in result.stdout.split("\n"):
            if "Page size:" in line:
                match = re.search(r"([\d.]+)\s*x\s*([\d.]+)\s*pts", line)
                if match:
                    width = float(match.group(1))
                    height = float(match.group(2))
                    # Return same dimensions for all pages (pdfinfo shows only first page)
                    page_count = count_pdf_pages(pdf_path)
                    return [(width, height)] * page_count

        # Default A4 if not found
        logger.warning("Could not get page size, using A4 default")
        return [(595.276, 841.89)] * count_pdf_pages(pdf_path)

    except Exception as e:
        logger.warning(f"Failed to get page dimensions: {e}")
        return [(595.276, 841.89)] * count_pdf_pages(pdf_path)


def extract_ocr_boxes_from_pdf(pdf_path: str) -> list[OCRBoxData]:
    """Extract text boxes from a PDF with OCR text layer using pdftotext -bbox.

    Args:
        pdf_path: Path to PDF file with OCR text layer

    Returns:
        List of OCRBoxData objects
    """
    boxes = []

    try:
        # Get page dimensions
        get_pdf_page_dimensions(pdf_path)

        # Run pdftotext with bbox to get XML output with coordinates
        result = subprocess.run(
            ["pdftotext", "-bbox-layout", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse HTML output
        # pdftotext -bbox-layout outputs HTML with word tags containing coordinates
        html_content = result.stdout

        # Parse each page
        page_pattern = r'<page\s+width="([\d.]+)"\s+height="([\d.]+)">(.*?)</page>'
        word_pattern = r'<word\s+xMin="([\d.]+)"\s+yMin="([\d.]+)"\s+xMax="([\d.]+)"\s+yMax="([\d.]+)">(.*?)</word>'

        for page_num, page_match in enumerate(
            re.finditer(page_pattern, html_content, re.DOTALL), start=1
        ):
            page_width = float(page_match.group(1))
            page_height = float(page_match.group(2))
            page_content = page_match.group(3)

            # Extract words with coordinates
            for word_match in re.finditer(word_pattern, page_content):
                x_min = float(word_match.group(1))
                y_min = float(word_match.group(2))
                x_max = float(word_match.group(3))
                y_max = float(word_match.group(4))
                text = word_match.group(5).strip()

                if not text:
                    continue

                # Convert to percentages
                x_percent = (x_min / page_width) * 100
                y_percent = (y_min / page_height) * 100
                width_percent = ((x_max - x_min) / page_width) * 100

                # Height in points
                height_pts = y_max - y_min

                boxes.append(
                    OCRBoxData(
                        text=text,
                        x=x_percent,
                        y=y_percent,
                        width=width_percent,
                        height=height_pts,
                        confidence=1.0,  # pdftotext doesn't provide confidence
                        page_num=page_num,
                    )
                )

        logger.info(f"Extracted {len(boxes)} text boxes from {pdf_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"pdftotext failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract boxes from PDF: {e}")
        raise

    return boxes


def count_pdf_pages(pdf_path: str) -> int:
    """Count pages in a PDF file using pdfinfo."""
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.split("\n"):
            if "Pages:" in line:
                return int(line.split(":")[1].strip())

        return 0
    except Exception as e:
        logger.error(f"Failed to count pages: {e}")
        return 0


def count_odf_pages(odf_path: str) -> int:
    """Estimate page count in an ODF file.

    This is an approximation based on page breaks and content.
    """
    from odf.opendocument import load
    from odf.text import P

    doc = load(odf_path)
    body = doc.body

    # Count explicit page breaks
    page_breaks = 0
    for elem in body.getElementsByType(P):
        style_attr = elem.getAttribute("stylename")
        if style_attr and "PageBreak" in str(style_attr):
            page_breaks += 1

    # Pages = page breaks + 1
    return page_breaks + 1


def export_to_odf(boxes: list[OCRBoxData], output_path: str) -> bool:
    """Export OCR boxes to ODF format.

    Args:
        boxes: List of OCR box data
        output_path: Output ODF file path

    Returns:
        True if successful
    """
    # Import exporter
    from bigocrpdf.utils.odf_exporter import ODFExporter

    # Convert to format expected by exporter
    exporter = ODFExporter()

    try:
        # Note: export_structured_data() takes (ocr_boxes, output_path)
        exporter.export_structured_data(boxes, output_path)
        logger.info(f"Exported to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


def print_quality_report(
    pdf_path: str,
    odf_path: str,
    boxes: list[OCRBoxData],
    pdf_pages: int,
    odf_pages: int,
) -> dict:
    """Print and return quality metrics.

    Args:
        pdf_path: Source PDF path
        odf_path: Output ODF path
        boxes: List of OCR boxes with data
        pdf_pages: Page count in source PDF
        odf_pages: Page count in output ODF

    Returns:
        Dict with quality metrics
    """
    import statistics

    num_boxes = len(boxes)
    page_ratio = odf_pages / pdf_pages if pdf_pages > 0 else 0
    page_diff = odf_pages - pdf_pages

    # Calculate height statistics
    heights = [b.height for b in boxes if b.height > 0]
    estimated_font_sizes = [min(72, max(4, h / 3.5)) for h in heights]

    # Calculate character statistics per page
    chars_per_page = {}
    for box in boxes:
        if box.page_num not in chars_per_page:
            chars_per_page[box.page_num] = 0
        chars_per_page[box.page_num] += len(box.text)

    total_chars = sum(len(b.text) for b in boxes)
    avg_chars_per_page = total_chars / pdf_pages if pdf_pages > 0 else 0

    print("\n" + "=" * 60)
    print("ODF EXPORT QUALITY REPORT")
    print("=" * 60)
    print(f"Source PDF:     {pdf_path}")
    print(f"Output ODF:     {odf_path}")
    print("-" * 60)
    print("CONTENT METRICS")
    print(f"  OCR Boxes:      {num_boxes}")
    print(f"  Total chars:    {total_chars}")
    print(f"  Avg chars/page: {avg_chars_per_page:.0f}")
    print("-" * 60)
    print("PAGE METRICS")
    print(f"  PDF Pages:      {pdf_pages}")
    print(f"  ODF Pages:      {odf_pages}")
    print(f"  Page Ratio:     {page_ratio:.2f}x")
    print(f"  Page Diff:      {page_diff:+d}")
    print("-" * 60)
    print("FONT METRICS (estimated from bbox heights)")
    if heights:
        print(f"  Min height:     {min(heights):.1f}pt -> ~{min(heights) / 3.5:.0f}pt font")
        print(f"  Max height:     {max(heights):.1f}pt -> ~{max(heights) / 3.5:.0f}pt font")
        print(
            f"  Median height:  {statistics.median(heights):.1f}pt -> ~{statistics.median(heights) / 3.5:.0f}pt font"
        )
        print(
            f"  Est. font range: {min(estimated_font_sizes):.0f}pt - {max(estimated_font_sizes):.0f}pt"
        )
    print("-" * 60)

    # Quality assessment
    if page_ratio == 1.0:
        quality = "EXCELLENT"
        print(f"Quality:        {quality} - Perfect page match!")
    elif 0.9 <= page_ratio <= 1.1:
        quality = "GOOD"
        print(f"Quality:        {quality} - Minor page difference")
    elif 0.75 <= page_ratio <= 1.25:
        quality = "ACCEPTABLE"
        print(f"Quality:        {quality} - Some page deviation")
    else:
        quality = "POOR"
        print(f"Quality:        {quality} - Significant page mismatch")

    print("=" * 60 + "\n")

    return {
        "pdf_path": pdf_path,
        "odf_path": odf_path,
        "num_boxes": num_boxes,
        "total_chars": total_chars,
        "avg_chars_per_page": avg_chars_per_page,
        "pdf_pages": pdf_pages,
        "odf_pages": odf_pages,
        "page_ratio": page_ratio,
        "page_diff": page_diff,
        "quality": quality,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test ODF export quality from OCR PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_pdf",
        type=str,
        help="Input PDF file with OCR text layer",
    )
    parser.add_argument(
        "output_odf",
        type=str,
        nargs="?",
        default=None,
        help="Output ODF file (default: input.odt)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--boxes-only",
        action="store_true",
        help="Only extract and print box statistics, don't export",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    input_path = Path(args.input_pdf)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Default output path
    output_path = args.output_odf
    if not output_path:
        output_path = str(input_path.with_suffix(".odt"))

    # Count PDF pages
    pdf_pages = count_pdf_pages(str(input_path))
    logger.info(f"PDF has {pdf_pages} pages")

    # Extract OCR boxes
    boxes = extract_ocr_boxes_from_pdf(str(input_path))

    # Print box statistics per page
    if args.verbose or args.boxes_only:
        page_counts = {}
        for box in boxes:
            page_counts[box.page_num] = page_counts.get(box.page_num, 0) + 1

        print("\nBoxes per page:")
        for page, count in sorted(page_counts.items()):
            print(f"  Page {page}: {count} boxes")
        print()

    if args.boxes_only:
        print(f"Total boxes: {len(boxes)}")
        sys.exit(0)

    # Export to ODF
    export_to_odf(boxes, output_path)

    # Count ODF pages
    odf_pages = count_odf_pages(output_path)

    # Generate report
    report = print_quality_report(
        str(input_path),
        output_path,
        boxes,
        pdf_pages,
        odf_pages,
    )

    if args.json:
        import json

        print(json.dumps(report, indent=2))

    # Exit with non-zero if quality is poor
    if report["quality"] == "POOR":
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
