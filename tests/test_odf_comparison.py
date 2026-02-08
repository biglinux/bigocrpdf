#!/usr/bin/env python3
"""
ODF Export Testing Script using LayoutAnalyzer

Tests the LayoutAnalyzer-based ODF exporter with OCR data from PDF files.

Usage:
    python -m bigocrpdf.scripts.test_odf_comparison input.pdf [output_dir] [--verbose]
"""

import argparse
import html
import json
import logging
import os
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


def get_pdf_page_count(pdf_path: str) -> int:
    """Get page count from PDF file."""
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
        logger.error(f"Failed to get page count: {e}")
        return 0


def extract_ocr_boxes(pdf_path: str) -> list[OCRBoxData]:
    """Extract text boxes from a PDF with OCR text layer.

    Uses pdftotext -bbox-layout for accurate word-level coordinate extraction.
    """
    boxes = []

    try:
        # Run pdftotext with bbox-layout for word coordinates
        result = subprocess.run(
            ["pdftotext", "-bbox-layout", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True,
        )

        html_content = result.stdout

        # Parse HTML output
        page_pattern = r'<page\s+width="([\d.]+)"\s+height="([\d.]+)">(.*?)</page>'
        word_pattern = r'<word\s+xMin="([\d.]+)"\s+yMin="([\d.]+)"\s+xMax="([\d.]+)"\s+yMax="([\d.]+)">(.*?)</word>'

        for page_num, page_match in enumerate(
            re.finditer(page_pattern, html_content, re.DOTALL), start=1
        ):
            page_width = float(page_match.group(1))
            page_height = float(page_match.group(2))
            page_content = page_match.group(3)

            for word_match in re.finditer(word_pattern, page_content):
                x_min = float(word_match.group(1))
                y_min = float(word_match.group(2))
                x_max = float(word_match.group(3))
                y_max = float(word_match.group(4))
                text = word_match.group(5).strip()

                if not text:
                    continue

                # Decode HTML entities (e.g., &lt; -> <, &gt; -> >)
                text = html.unescape(text)

                # Convert to percentages
                x_percent = (x_min / page_width) * 100
                y_percent = (y_min / page_height) * 100
                width_percent = ((x_max - x_min) / page_width) * 100

                # Height in points (bounding box height)
                height_pts = y_max - y_min

                boxes.append(
                    OCRBoxData(
                        text=text,
                        x=x_percent,
                        y=y_percent,
                        width=width_percent,
                        height=height_pts,
                        confidence=1.0,
                        page_num=page_num,
                    )
                )

        logger.info(f"Extracted {len(boxes)} text boxes from {pdf_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"pdftotext failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract boxes: {e}")
        raise

    return boxes


def count_odf_pages(odf_path: str) -> int:
    """Count pages in an ODF file based on page breaks."""
    try:
        from odf.opendocument import load
        from odf.text import P

        doc = load(odf_path)
        body = doc.body

        page_breaks = 0
        for elem in body.getElementsByType(P):
            style_attr = elem.getAttribute("stylename")
            if style_attr and "PageBreak" in str(style_attr):
                page_breaks += 1

        return page_breaks + 1
    except Exception as e:
        logger.warning(f"Could not count ODF pages: {e}")
        return 1


def get_file_size_kb(path: str) -> float:
    """Get file size in KB."""
    return os.path.getsize(path) / 1024


def export_with_layout_analyzer(boxes: list[OCRBoxData], output_path: str) -> bool:
    """Export using ODFExporter with LayoutAnalyzer."""
    try:
        from bigocrpdf.utils.odf_exporter import OCRTextData, ODFExporter

        # Convert to OCRTextData format
        ocr_data = [
            OCRTextData(
                text=b.text,
                x=b.x,
                y=b.y,
                width=b.width,
                height=b.height,
                confidence=b.confidence,
                page_num=b.page_num,
            )
            for b in boxes
        ]

        exporter = ODFExporter()
        return exporter.export_structured_data(ocr_data, output_path)
    except Exception as e:
        logger.error(f"ODFExporter failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def analyze_box_statistics(boxes: list[OCRBoxData]) -> dict:
    """Analyze OCR box statistics."""
    if not boxes:
        return {}

    import statistics

    heights = [b.height for b in boxes if b.height > 0]
    pages = {b.page_num for b in boxes}

    # Estimate font sizes (height / 3.5 based on bbox ratio)
    estimated_fonts = [min(72, max(4, h / 3.5)) for h in heights]

    # Characters per page
    chars_per_page = {}
    for box in boxes:
        if box.page_num not in chars_per_page:
            chars_per_page[box.page_num] = 0
        chars_per_page[box.page_num] += len(box.text)

    return {
        "total_boxes": len(boxes),
        "total_chars": sum(len(b.text) for b in boxes),
        "pages": len(pages),
        "avg_chars_per_page": statistics.mean(chars_per_page.values()) if chars_per_page else 0,
        "height_stats": {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "median": statistics.median(heights) if heights else 0,
            "mean": statistics.mean(heights) if heights else 0,
        },
        "font_estimate": {
            "min": min(estimated_fonts) if estimated_fonts else 0,
            "max": max(estimated_fonts) if estimated_fonts else 0,
            "median": statistics.median(estimated_fonts) if estimated_fonts else 0,
        },
    }


def print_comparison_report(
    pdf_path: str,
    boxes: list[OCRBoxData],
    pdf_pages: int,
    results: dict[str, dict],
) -> dict:
    """Print comparison report and return summary."""
    box_stats = analyze_box_statistics(boxes)

    print("\n" + "=" * 70)
    print("ODF EXPORT COMPARISON REPORT")
    print("=" * 70)
    print(f"Source PDF:     {pdf_path}")
    print(f"PDF Pages:      {pdf_pages}")
    print(f"OCR Boxes:      {box_stats.get('total_boxes', 0)}")
    print(f"Total chars:    {box_stats.get('total_chars', 0)}")

    print("\n" + "-" * 70)
    print("BOX HEIGHT STATISTICS (for font size calculation)")
    hs = box_stats.get("height_stats", {})
    fs = box_stats.get("font_estimate", {})
    print(f"  Height range: {hs.get('min', 0):.1f}pt - {hs.get('max', 0):.1f}pt")
    print(f"  Height median: {hs.get('median', 0):.1f}pt")
    print(f"  Est. font range: {fs.get('min', 0):.0f}pt - {fs.get('max', 0):.0f}pt")
    print(f"  Est. font median: {fs.get('median', 0):.0f}pt")

    print("\n" + "-" * 70)
    print("EXPORTER COMPARISON")
    print("-" * 70)
    print(f"{'Exporter':<25} {'Status':<10} {'Pages':<8} {'Size (KB)':<12} {'Page Ratio':<12}")
    print("-" * 70)

    summary = {"pdf_pages": pdf_pages, "box_stats": box_stats, "exporters": {}}

    for name, result in results.items():
        if result["success"]:
            odf_pages = result["odf_pages"]
            size_kb = result["size_kb"]
            ratio = odf_pages / pdf_pages if pdf_pages > 0 else 0

            status = "OK"
            if 0.9 <= ratio <= 1.1:
                quality = "GOOD"
            elif 0.75 <= ratio <= 1.25:
                quality = "ACCEPTABLE"
            else:
                quality = "POOR"

            print(
                f"{name:<25} {status:<10} {odf_pages:<8} {size_kb:<12.1f} {ratio:.2f}x ({quality})"
            )

            summary["exporters"][name] = {
                "success": True,
                "odf_pages": odf_pages,
                "size_kb": size_kb,
                "page_ratio": ratio,
                "quality": quality,
                "output_path": result["output_path"],
            }
        else:
            print(f"{name:<25} {'FAILED':<10} {'-':<8} {'-':<12} {'-':<12}")
            summary["exporters"][name] = {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }

    print("=" * 70)

    # Determine best exporter
    best = None
    best_ratio_diff = float("inf")
    for name, data in summary["exporters"].items():
        if data.get("success"):
            ratio_diff = abs(1.0 - data.get("page_ratio", 0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best = name

    if best:
        print(f"\nBest result: {best}")
        print(f"Output file: {summary['exporters'][best]['output_path']}")

    print("=" * 70 + "\n")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export OCR PDF to ODF using LayoutAnalyzer",
    )
    parser.add_argument(
        "input_pdf",
        type=str,
        help="Input PDF file with OCR text layer",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    # Get PDF info
    pdf_pages = get_pdf_page_count(str(input_path))
    logger.info(f"PDF has {pdf_pages} pages")

    # Extract OCR boxes
    boxes = extract_ocr_boxes(str(input_path))

    # Export using LayoutAnalyzer only
    base_name = input_path.stem
    output_path = output_dir / f"{base_name}_odf_layout_analyzer.odt"
    logger.info(f"Exporting to: {output_path}")

    try:
        success = export_with_layout_analyzer(boxes, str(output_path))
        if success and output_path.exists():
            result = {
                "success": True,
                "output_path": str(output_path),
                "odf_pages": count_odf_pages(str(output_path)),
                "size_kb": get_file_size_kb(str(output_path)),
            }
        else:
            result = {"success": False, "error": "Export returned False"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    # Generate report
    results = {"layout_analyzer": result}
    summary = print_comparison_report(str(input_path), boxes, pdf_pages, results)

    if args.json:
        print(json.dumps(summary, indent=2))

    if not result.get("success"):
        sys.exit(1)

    print(f"\nOutput: {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
