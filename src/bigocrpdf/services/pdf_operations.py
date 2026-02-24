"""
BigOcrPdf - PDF Operations Service

Pure-Python service for PDF manipulation operations.
No GTK dependencies - can be used from CLI, GUI, or scripts.

Supported operations:
  - Split by page count or file size
  - Merge multiple PDFs
  - Compress (reduce image quality/resolution)
  - Delete / extract pages
  - Rotate pages
  - Insert pages from another PDF
  - Page count and metadata info
"""

import io
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import pikepdf

from bigocrpdf.constants import MIN_IMAGE_DIMENSION_PX
from bigocrpdf.utils.i18n import _

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Error classification for PDF operations."""

    NONE = auto()
    FILE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    CORRUPT_PDF = auto()
    PASSWORD_PROTECTED = auto()
    DISK_FULL = auto()
    UNKNOWN = auto()


def _classify_error(e: Exception) -> ErrorCode:
    """Classify an exception into an ErrorCode."""
    if isinstance(e, FileNotFoundError):
        return ErrorCode.FILE_NOT_FOUND
    if isinstance(e, PermissionError):
        return ErrorCode.PERMISSION_DENIED
    if isinstance(e, pikepdf.PasswordError):
        return ErrorCode.PASSWORD_PROTECTED
    if isinstance(e, pikepdf.PdfError):
        return ErrorCode.CORRUPT_PDF
    if isinstance(e, OSError) and e.errno == 28:
        return ErrorCode.DISK_FULL
    return ErrorCode.UNKNOWN


def _friendly_error(e: Exception) -> str:
    """Map common exceptions to user-friendly messages (Nielsen's heuristic #9)."""
    if isinstance(e, FileNotFoundError):
        return _("Could not find the file. Was it moved or deleted?")
    if isinstance(e, PermissionError):
        return _("Cannot write to this folder. Choose a different location.")
    if isinstance(e, pikepdf.PasswordError):
        return _("This PDF is password-protected. Remove the password first.")
    if isinstance(e, pikepdf.PdfError):
        return _("The PDF file appears to be damaged or invalid: {error}").format(error=e)
    return str(e)


def _fail(e: Exception) -> "OperationResult":
    """Create a failed OperationResult from an exception."""
    return OperationResult(
        success=False,
        message=_friendly_error(e),
        error_code=_classify_error(e),
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PDFInfo:
    """Basic information about a PDF file."""

    path: str
    page_count: int
    file_size_bytes: int
    title: str = ""
    author: str = ""
    creator: str = ""
    encrypted: bool = False
    pdf_version: str = ""

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


@dataclass
class SplitResult:
    """Result of a split operation."""

    output_files: list[str] = field(default_factory=list)
    total_pages: int = 0
    parts: int = 0


@dataclass
class OperationResult:
    """Generic result for PDF operations."""

    success: bool
    message: str = ""
    output_path: str = ""
    pages_affected: int = 0
    error_code: ErrorCode = ErrorCode.NONE


# ---------------------------------------------------------------------------
# Info / Inspection
# ---------------------------------------------------------------------------


def get_pdf_info(pdf_path: str | Path) -> PDFInfo:
    """Get basic information about a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        PDFInfo with metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        pikepdf.PdfError: If the file is not a valid PDF.
    """
    pdf_path = str(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    file_size = os.path.getsize(pdf_path)

    with pikepdf.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        info = PDFInfo(
            path=pdf_path,
            page_count=page_count,
            file_size_bytes=file_size,
            pdf_version=str(pdf.pdf_version),
            encrypted=pdf.is_encrypted,
        )

        # Extract metadata if available
        with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
            info.title = str(meta.get("dc:title", ""))
            info.author = str(meta.get("dc:creator", ""))

        if "/Creator" in pdf.docinfo:
            info.creator = str(pdf.docinfo["/Creator"])

    return info


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_by_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    pages_per_file: int,
    *,
    prefix: str = "",
) -> SplitResult:
    """Split a PDF into multiple files with a fixed number of pages each.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where output files will be created.
        pages_per_file: Maximum number of pages per output file.
        prefix: Optional filename prefix (default: source filename stem).

    Returns:
        SplitResult with list of created files.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pages_per_file < 1:
        raise ValueError("pages_per_file must be >= 1")

    if not prefix:
        prefix = pdf_path.stem

    result = SplitResult()

    with pikepdf.open(pdf_path) as src:
        total = len(src.pages)
        result.total_pages = total
        parts = math.ceil(total / pages_per_file)
        result.parts = parts

        for part_idx in range(parts):
            start = part_idx * pages_per_file
            end = min(start + pages_per_file, total)

            dst = pikepdf.Pdf.new()
            for page_num in range(start, end):
                dst.pages.append(src.pages[page_num])

            out_name = f"{prefix}_part{part_idx + 1:03d}.pdf"
            out_path = str(output_dir / out_name)
            dst.save(out_path)
            dst.close()

            result.output_files.append(out_path)
            logger.info(
                "Split part %d/%d: pages %d-%d → %s",
                part_idx + 1,
                parts,
                start + 1,
                end,
                out_name,
            )

    return result


def split_by_size(
    pdf_path: str | Path,
    output_dir: str | Path,
    max_size_mb: float,
    *,
    prefix: str = "",
) -> SplitResult:
    """Split a PDF so each output file is at most *max_size_mb* megabytes.

    Uses a greedy approach: adds pages one at a time and flushes to a new
    part whenever the accumulated size exceeds the threshold.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory for output files.
        max_size_mb: Maximum file size per part in megabytes.
        prefix: Optional filename prefix.

    Returns:
        SplitResult with created files.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_size_mb <= 0:
        raise ValueError("max_size_mb must be > 0")

    max_bytes = int(max_size_mb * 1024 * 1024)

    if not prefix:
        prefix = pdf_path.stem

    result = SplitResult()

    with pikepdf.open(pdf_path) as src:
        total = len(src.pages)
        result.total_pages = total

        part_pages: list[int] = []
        part_idx = 0

        def _flush(pages: list[int]) -> None:
            nonlocal part_idx
            if not pages:
                return
            part_idx += 1
            dst = pikepdf.Pdf.new()
            for pn in pages:
                dst.pages.append(src.pages[pn])

            out_name = f"{prefix}_part{part_idx:03d}.pdf"
            out_path = str(output_dir / out_name)
            dst.save(out_path)
            dst.close()

            actual_mb = os.path.getsize(out_path) / (1024 * 1024)
            result.output_files.append(out_path)
            logger.info(
                "Split part %d: %d pages, %.2f MB → %s",
                part_idx,
                len(pages),
                actual_mb,
                out_name,
            )

        # Pre-compute per-page size estimates (O(n) total)
        page_sizes: list[int] = []
        for i in range(total):
            single = pikepdf.Pdf.new()
            single.pages.append(src.pages[i])
            buf = io.BytesIO()
            single.save(buf)
            page_sizes.append(buf.tell())
            single.close()

        for page_num in range(total):
            part_pages.append(page_num)

            # Estimate size by summing per-page sizes (O(n) total).
            # Individual page sizes overestimate the combined PDF
            # (shared fonts/images counted per page), so splits are
            # conservative — safe but may produce slightly more parts.
            est_size = sum(page_sizes[pn] for pn in part_pages)

            # If we exceed the limit and have more than one page,
            # flush all pages EXCEPT the last one.
            if est_size > max_bytes and len(part_pages) > 1:
                _flush(part_pages[:-1])
                part_pages = [page_num]

        # Flush remaining pages
        _flush(part_pages)
        result.parts = part_idx

    return result


def split_by_ranges(
    pdf_path: str | Path,
    output_dir: str | Path,
    ranges: list[tuple[int, int]],
    *,
    prefix: str = "",
) -> SplitResult:
    """Split a PDF by explicit page ranges.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory for output files.
        ranges: List of (start, end) tuples, 1-indexed inclusive.
        prefix: Optional filename prefix.

    Returns:
        SplitResult with created files.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prefix:
        prefix = pdf_path.stem

    result = SplitResult()

    with pikepdf.open(pdf_path) as src:
        total = len(src.pages)
        result.total_pages = total

        for _idx, (start, end) in enumerate(ranges, 1):
            start_0 = max(0, start - 1)
            end_0 = min(end, total)

            if start_0 >= total or start_0 >= end_0:
                logger.warning("Skipping invalid range (%d, %d)", start, end)
                continue

            dst = pikepdf.Pdf.new()
            for pn in range(start_0, end_0):
                dst.pages.append(src.pages[pn])

            out_name = f"{prefix}_pages{start}-{end}.pdf"
            out_path = str(output_dir / out_name)
            dst.save(out_path)
            dst.close()

            result.output_files.append(out_path)
            result.parts += 1
            logger.info("Extracted pages %d-%d → %s", start, end, out_name)

    return result


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_pdfs(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> OperationResult:
    """Merge multiple PDF files into one.

    Args:
        input_paths: List of PDF file paths to merge (in order).
        output_path: Path for the merged output PDF.

    Returns:
        OperationResult.
    """
    if not input_paths:
        return OperationResult(success=False, message=_("No input files provided."))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dst = pikepdf.Pdf.new()
    total_pages = 0
    open_sources: list[pikepdf.Pdf] = []

    try:
        for path in input_paths:
            path = Path(path)
            if not path.is_file():
                logger.warning("Skipping missing file: %s", path)
                continue

            src = pikepdf.open(path)
            open_sources.append(src)
            count = len(src.pages)
            for page in src.pages:
                dst.pages.append(page)
            total_pages += count
            logger.info("Merged %d pages from %s", count, path.name)

        dst.save(str(output_path))
        logger.info("Merged PDF saved: %s (%d pages)", output_path, total_pages)

        return OperationResult(
            success=True,
            message=f"Merged {len(input_paths)} files → {total_pages} pages",
            output_path=str(output_path),
            pages_affected=total_pages,
        )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Merge failed: %s", e)
        return _fail(e)
    finally:
        for src in open_sources:
            src.close()
        dst.close()


# ---------------------------------------------------------------------------
# Extract pages
# ---------------------------------------------------------------------------


def extract_pages(
    pdf_path: str | Path,
    output_path: str | Path,
    pages: list[int],
) -> OperationResult:
    """Extract specific pages into a new PDF.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.
        pages: List of 1-indexed page numbers to extract.

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pikepdf.open(pdf_path) as src:
            total = len(src.pages)
            dst = pikepdf.Pdf.new()

            valid = [p for p in pages if 1 <= p <= total]
            if not valid:
                return OperationResult(
                    success=False,
                    message=f"No valid pages in {pages} (document has {total} pages)",
                )

            for p in valid:
                dst.pages.append(src.pages[p - 1])

            dst.save(str(output_path))
            dst.close()

            logger.info("Extracted pages %s → %s", valid, output_path)
            return OperationResult(
                success=True,
                message=f"Extracted {len(valid)} pages",
                output_path=str(output_path),
                pages_affected=len(valid),
            )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Extract failed: %s", e)
        return _fail(e)


# ---------------------------------------------------------------------------
# Delete pages
# ---------------------------------------------------------------------------


def delete_pages(
    pdf_path: str | Path,
    output_path: str | Path,
    pages: list[int],
) -> OperationResult:
    """Remove specific pages from a PDF.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.
        pages: List of 1-indexed page numbers to remove.

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pikepdf.open(pdf_path) as src:
            total = len(src.pages)
            pages_to_delete = {p for p in pages if 1 <= p <= total}

            if not pages_to_delete:
                return OperationResult(
                    success=False,
                    message=f"No valid pages to delete (document has {total} pages)",
                )

            if len(pages_to_delete) >= total:
                return OperationResult(
                    success=False,
                    message="Cannot delete all pages from the document",
                )

            dst = pikepdf.Pdf.new()
            kept = 0
            for i in range(total):
                if (i + 1) not in pages_to_delete:
                    dst.pages.append(src.pages[i])
                    kept += 1

            dst.save(str(output_path))
            dst.close()

            logger.info(
                "Deleted pages %s, kept %d pages → %s",
                sorted(pages_to_delete),
                kept,
                output_path,
            )
            return OperationResult(
                success=True,
                message=f"Deleted {len(pages_to_delete)} pages, {kept} remaining",
                output_path=str(output_path),
                pages_affected=len(pages_to_delete),
            )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Delete pages failed: %s", e)
        return _fail(e)


# ---------------------------------------------------------------------------
# Insert pages
# ---------------------------------------------------------------------------


def insert_pages(
    pdf_path: str | Path,
    insert_path: str | Path,
    output_path: str | Path,
    at_page: int = 0,
    *,
    source_pages: list[int] | None = None,
) -> OperationResult:
    """Insert pages from another PDF into the target document.

    Args:
        pdf_path: Target PDF path.
        insert_path: PDF whose pages will be inserted.
        output_path: Output PDF path.
        at_page: 1-indexed position where pages are inserted (0 = append at end).
        source_pages: Optional list of 1-indexed pages from insert_path to use.
                      If None, all pages are inserted.

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    insert_path = Path(insert_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pikepdf.open(pdf_path) as target, pikepdf.open(insert_path) as source:
            target_total = len(target.pages)
            source_total = len(source.pages)

            # Determine which pages to insert
            if source_pages:
                insert_indices = [p - 1 for p in source_pages if 1 <= p <= source_total]
            else:
                insert_indices = list(range(source_total))

            if not insert_indices:
                return OperationResult(
                    success=False,
                    message="No valid pages to insert",
                )

            # Determine insert position (0-indexed)
            if at_page <= 0 or at_page > target_total:
                insert_pos = target_total  # Append
            else:
                insert_pos = at_page - 1

            dst = pikepdf.Pdf.new()

            # Copy pages before insert point
            for i in range(insert_pos):
                dst.pages.append(target.pages[i])

            # Insert pages
            for idx in insert_indices:
                dst.pages.append(source.pages[idx])

            # Copy pages after insert point
            for i in range(insert_pos, target_total):
                dst.pages.append(target.pages[i])

            dst.save(str(output_path))
            dst.close()

            logger.info(
                "Inserted %d pages at position %d → %s",
                len(insert_indices),
                insert_pos + 1,
                output_path,
            )
            return OperationResult(
                success=True,
                message=f"Inserted {len(insert_indices)} pages at position {insert_pos + 1}",
                output_path=str(output_path),
                pages_affected=len(insert_indices),
            )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Insert pages failed: %s", e)
        return _fail(e)


# ---------------------------------------------------------------------------
# Rotate pages
# ---------------------------------------------------------------------------


def rotate_pages(
    pdf_path: str | Path,
    output_path: str | Path,
    pages: list[int],
    angle: int,
) -> OperationResult:
    """Rotate specific pages in a PDF.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.
        pages: List of 1-indexed page numbers to rotate.
        angle: Rotation angle in degrees (90, 180, 270 / -90).

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    angle = angle % 360
    if angle not in (0, 90, 180, 270):
        return OperationResult(success=False, message=f"Invalid angle: {angle}")

    if angle == 0:
        return OperationResult(success=True, message="No rotation needed", pages_affected=0)

    try:
        with pikepdf.open(pdf_path) as pdf:
            total = len(pdf.pages)
            rotated = 0

            for p in pages:
                if 1 <= p <= total:
                    page = pdf.pages[p - 1]
                    current = int(page.get("/Rotate", 0))
                    page.Rotate = (current + angle) % 360
                    rotated += 1

            if rotated == 0:
                return OperationResult(
                    success=False,
                    message=f"No valid pages to rotate (document has {total} pages)",
                )

            pdf.save(str(output_path))

            logger.info("Rotated %d pages by %d° → %s", rotated, angle, output_path)
            return OperationResult(
                success=True,
                message=f"Rotated {rotated} pages by {angle}°",
                output_path=str(output_path),
                pages_affected=rotated,
            )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Rotate failed: %s", e)
        return _fail(e)


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------


def compress_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    image_quality: int = 60,
    image_dpi: int = 150,
) -> OperationResult:
    """Compress a PDF by reducing image quality and applying stream compression.

    Uses PIL to re-encode images at lower quality/resolution, and pikepdf's
    stream compression for all other streams.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.
        image_quality: JPEG quality for image re-encoding (1-95, default 60).
        image_dpi: Target DPI for images (default 150).

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_size = os.path.getsize(pdf_path)

    try:
        with pikepdf.open(pdf_path) as pdf:
            images_compressed = 0
            seen_objgen: set = set()

            for page in pdf.pages:
                images_compressed += _compress_page_images(
                    pdf,
                    page,
                    image_quality,
                    image_dpi,
                    seen_objgen,
                )

            # Remove unreferenced objects and enable stream compression
            pdf.remove_unreferenced_resources()
            pdf.save(
                str(output_path),
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )

        new_size = os.path.getsize(str(output_path))
        ratio = (1 - new_size / original_size) * 100 if original_size > 0 else 0

        logger.info(
            "Compressed: %.2f MB → %.2f MB (%.1f%% reduction, %d images re-encoded)",
            original_size / 1048576,
            new_size / 1048576,
            ratio,
            images_compressed,
        )
        return OperationResult(
            success=True,
            message=(
                f"Compressed {original_size / 1048576:.2f} MB → "
                f"{new_size / 1048576:.2f} MB ({ratio:.1f}% reduction)"
            ),
            output_path=str(output_path),
            pages_affected=images_compressed,
        )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Compress failed: %s", e)
        return _fail(e)


def _compress_page_images(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    quality: int,
    target_dpi: int,
    seen_objgen: set | None = None,
) -> int:
    """Compress images in a single page.

    Args:
        seen_objgen: Set of (objgen) tuples already processed. Shared across pages
                     to avoid re-processing the same XObject stream reference.

    Returns:
        Number of images compressed.
    """
    if seen_objgen is None:
        seen_objgen = set()
    import io

    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available — skipping image compression")
        return 0

    count = 0

    try:
        resources = page.get("/Resources", {})
        xobjects = resources.get("/XObject", {})
    except (AttributeError, TypeError):
        return 0

    for key in list(xobjects.keys()):
        try:
            obj = xobjects[key]
            if not isinstance(obj, pikepdf.Stream):
                continue
            if obj.get("/Subtype") != pikepdf.Name.Image:
                continue

            # Skip already-processed shared XObjects
            objgen = obj.objgen
            if objgen in seen_objgen:
                continue
            seen_objgen.add(objgen)

            width = int(obj.get("/Width", 0))
            height = int(obj.get("/Height", 0))

            if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
                continue

            # Use pikepdf.PdfImage for reliable decoding
            try:
                pdf_image = pikepdf.PdfImage(obj)
                pil_img = pdf_image.as_pil_image()
            except (pikepdf.PdfError, OSError, ValueError):
                continue

            # Determine original data size for comparison
            try:
                original_size = len(obj.read_raw_bytes())
            except (pikepdf.PdfError, OSError):
                original_size = width * height * 3  # rough estimate

            # Determine current effective DPI from page media box
            try:
                mbox = page.MediaBox
                page_width_pt = float(mbox[2]) - float(mbox[0])
                page_height_pt = float(mbox[3]) - float(mbox[1])
                if page_width_pt > 0 and page_height_pt > 0:
                    current_dpi = max(
                        width / (page_width_pt / 72),
                        height / (page_height_pt / 72),
                    )
                else:
                    current_dpi = 300
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                current_dpi = 300

            # Downsample if current DPI is significantly higher than target
            if current_dpi > target_dpi * 1.2:
                scale = target_dpi / current_dpi
                new_w = max(MIN_IMAGE_DIMENSION_PX, int(width * scale))
                new_h = max(MIN_IMAGE_DIMENSION_PX, int(height * scale))
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

            # Re-encode as JPEG
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img = pil_img.convert("RGB")
            elif pil_img.mode == "L":
                pass  # Keep grayscale

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
            jpeg_data = buf.getvalue()

            # Only replace if the new size is actually smaller
            if len(jpeg_data) < original_size * 0.90:
                new_img = pikepdf.Stream(pdf, jpeg_data)
                new_img["/Type"] = pikepdf.Name.XObject
                new_img["/Subtype"] = pikepdf.Name.Image
                new_img["/Width"] = pil_img.width
                new_img["/Height"] = pil_img.height
                if pil_img.mode == "L":
                    new_img["/ColorSpace"] = pikepdf.Name.DeviceGray
                else:
                    new_img["/ColorSpace"] = pikepdf.Name.DeviceRGB
                new_img["/BitsPerComponent"] = 8
                new_img["/Filter"] = pikepdf.Name.DCTDecode

                xobjects[key] = new_img
                count += 1

        except (pikepdf.PdfError, OSError, ValueError) as e:
            logger.debug("Skipping image %s: %s", key, e)
            continue

    return count


# ---------------------------------------------------------------------------
# Reorder pages
# ---------------------------------------------------------------------------


def reorder_pages(
    pdf_path: str | Path,
    output_path: str | Path,
    new_order: list[int],
) -> OperationResult:
    """Reorder pages in a PDF.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.
        new_order: List of 1-indexed page numbers in the desired order.
                   Can include duplicates and omit pages.

    Returns:
        OperationResult.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pikepdf.open(pdf_path) as src:
            total = len(src.pages)
            valid = [p for p in new_order if 1 <= p <= total]

            if not valid:
                return OperationResult(
                    success=False,
                    message=f"No valid pages in order list (document has {total} pages)",
                )

            dst = pikepdf.Pdf.new()
            for p in valid:
                dst.pages.append(src.pages[p - 1])

            dst.save(str(output_path))
            dst.close()

            logger.info("Reordered %d pages → %s", len(valid), output_path)
            return OperationResult(
                success=True,
                message=f"Reordered {len(valid)} pages",
                output_path=str(output_path),
                pages_affected=len(valid),
            )
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Reorder failed: %s", e)
        return _fail(e)


# ---------------------------------------------------------------------------
# Reverse page order
# ---------------------------------------------------------------------------


def reverse_pages(
    pdf_path: str | Path,
    output_path: str | Path,
) -> OperationResult:
    """Reverse the page order of a PDF.

    Args:
        pdf_path: Source PDF path.
        output_path: Output PDF path.

    Returns:
        OperationResult.
    """
    try:
        with pikepdf.open(pdf_path) as src:
            total = len(src.pages)
            order = list(range(total, 0, -1))
        return reorder_pages(pdf_path, output_path, order)
    except (OSError, pikepdf.PdfError, ValueError) as e:
        logger.error("Reverse failed: %s", e)
        return _fail(e)
