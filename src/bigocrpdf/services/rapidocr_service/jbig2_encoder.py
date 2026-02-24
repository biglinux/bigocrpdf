"""JBIG2 and CCITT Group 4 encoding for PDF image compression.

Provides encoding functions for bi-level (1-bit) images using:
- jbig2enc (preferred): better compression via symbol matching
- CCITT Group 4 via Pillow (fallback): universal, no external deps

Both produce raw streams suitable for embedding in PDF via pikepdf.
"""

import io
import logging
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from bigocrpdf.constants import JBIG2_ENCODER_TIMEOUT_SECS

logger = logging.getLogger(__name__)


def jbig2enc_available() -> bool:
    """Check if jbig2enc is installed on the system."""
    return shutil.which("jbig2") is not None


def encode_jbig2(img: np.ndarray) -> bytes | None:
    """Encode a binary image to JBIG2 format for PDF embedding.

    Uses jbig2enc in generic mode (no symbol matching) for safe
    lossless compression. The output is a raw JBIG2 stream
    suitable for direct embedding via pikepdf.

    Args:
        img: Single-channel binary image (0 and 255 values, uint8).

    Returns:
        JBIG2 encoded bytes, or None on failure.
    """
    if img is None or img.size == 0:
        return None

    with tempfile.TemporaryDirectory(prefix="jbig2_") as tmpdir:
        input_path = Path(tmpdir) / "input.png"

        # Ensure 1-bit image saved as PNG for jbig2enc
        pil_img = Image.fromarray(img).convert("1")
        pil_img.save(input_path, "PNG")

        try:
            # Generic mode: -p outputs PDF-ready JBIG2 stream to stdout
            result = subprocess.run(
                ["jbig2", "-p", str(input_path)],
                capture_output=True,
                timeout=JBIG2_ENCODER_TIMEOUT_SECS,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if result.returncode != 0:
            logger.debug("jbig2enc failed: %s", result.stderr.decode(errors="replace"))
            return None

        return result.stdout if result.stdout else None


def encode_jbig2_with_globals(img: np.ndarray) -> tuple[bytes, bytes] | None:
    """Encode a binary image to JBIG2 with separate globals stream.

    Uses symbol mode (-s) for better compression on text documents.
    Returns both the page data and the globals (symbol dictionary),
    which are needed for proper JBIG2 embedding in PDF.

    Args:
        img: Single-channel binary image (0 and 255 values, uint8).

    Returns:
        Tuple of (page_data, globals_data), or None on failure.
    """
    if img is None or img.size == 0:
        return None

    with tempfile.TemporaryDirectory(prefix="jbig2_") as tmpdir:
        input_path = Path(tmpdir) / "input.png"

        pil_img = Image.fromarray(img).convert("1")
        pil_img.save(input_path, "PNG")

        try:
            # Symbol mode: -s -p creates output.sym + output.0000 files
            result = subprocess.run(
                ["jbig2", "-s", "-p", "-b", str(Path(tmpdir) / "output"), str(input_path)],
                capture_output=True,
                cwd=tmpdir,
                timeout=JBIG2_ENCODER_TIMEOUT_SECS,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if result.returncode != 0:
            return None

        page_file = Path(tmpdir) / "output.0000"
        sym_file = Path(tmpdir) / "output.sym"

        if not page_file.exists():
            return None

        page_data = page_file.read_bytes()
        globals_data = sym_file.read_bytes() if sym_file.exists() else b""

        if not page_data:
            return None

        return page_data, globals_data


def encode_ccitt_g4(img: np.ndarray) -> tuple[bytes, int, int] | None:
    """Encode a binary image as CCITT Group 4 via Pillow/libtiff.

    Saves to a TIFF with Group 4 compression, then extracts the raw
    CCITT stream data from the TIFF container.

    Args:
        img: Single-channel binary image (0 and 255 values, uint8).

    Returns:
        Tuple of (ccitt_data, width, height), or None on failure.
    """
    if img is None or img.size == 0:
        return None

    pil_img = Image.fromarray(img).convert("1")
    width, height = pil_img.size

    buf = io.BytesIO()
    try:
        # Single strip produces slightly smaller output (~1% less overhead)
        tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
        tiffinfo[278] = height  # RowsPerStrip = full image
        pil_img.save(buf, format="TIFF", compression="group4", tiffinfo=tiffinfo)
    except Exception as e:
        logger.debug("CCITT G4 encoding failed: %s", e)
        return None

    ccitt_data = _extract_ccitt_from_tiff(buf.getvalue())
    if not ccitt_data:
        return None

    return ccitt_data, width, height


def _read_ifd_values(tiff_bytes: bytes, endian: str, entry_offset: int) -> list[int]:
    """Read an array of integer values from a TIFF IFD entry."""
    type_id = struct.unpack_from(f"{endian}H", tiff_bytes, entry_offset + 2)[0]
    count = struct.unpack_from(f"{endian}I", tiff_bytes, entry_offset + 4)[0]

    type_sizes = {3: 2, 4: 4}  # SHORT=2 bytes, LONG=4 bytes
    fmt_chars = {3: "H", 4: "I"}
    item_size = type_sizes.get(type_id, 4)
    fmt_char = fmt_chars.get(type_id, "I")
    total_size = item_size * count

    # Values fit inline (<=4 bytes) or are at an offset
    if total_size <= 4:
        data_offset = entry_offset + 8
    else:
        data_offset = struct.unpack_from(f"{endian}I", tiff_bytes, entry_offset + 8)[0]

    return [
        struct.unpack_from(f"{endian}{fmt_char}", tiff_bytes, data_offset + i * item_size)[0]
        for i in range(count)
    ]


def _extract_ccitt_from_tiff(tiff_bytes: bytes) -> bytes | None:
    """Extract raw CCITT data from a TIFF file byte stream.

    Parses TIFF IFD to find strip offsets and byte counts,
    then concatenates all strips into a single raw data block.
    """
    if len(tiff_bytes) < 8:
        return None

    byte_order = tiff_bytes[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None

    ifd_offset = struct.unpack_from(f"{endian}I", tiff_bytes, 4)[0]
    if ifd_offset >= len(tiff_bytes):
        return None

    num_entries = struct.unpack_from(f"{endian}H", tiff_bytes, ifd_offset)[0]

    strip_offsets = None
    strip_byte_counts = None

    for i in range(num_entries):
        entry_offset = ifd_offset + 2 + i * 12
        if entry_offset + 12 > len(tiff_bytes):
            break

        tag = struct.unpack_from(f"{endian}H", tiff_bytes, entry_offset)[0]

        if tag == 273:  # StripOffsets
            strip_offsets = _read_ifd_values(tiff_bytes, endian, entry_offset)
        elif tag == 279:  # StripByteCounts
            strip_byte_counts = _read_ifd_values(tiff_bytes, endian, entry_offset)

    if not strip_offsets or not strip_byte_counts:
        return None
    if len(strip_offsets) != len(strip_byte_counts):
        return None

    parts = []
    for offset, length in zip(strip_offsets, strip_byte_counts):
        end = offset + length
        if end > len(tiff_bytes):
            return None
        parts.append(tiff_bytes[offset:end])

    return b"".join(parts) if parts else None


def encode_bilevel(img: np.ndarray) -> tuple[str, bytes, bytes | None] | None:
    """Encode a binary image using the best available method.

    Tries JBIG2 first (better compression), falls back to CCITT G4.

    Args:
        img: Single-channel binary image (0 and 255 values, uint8).

    Returns:
        Tuple of (encoding_name, data, globals_or_none):
        - ("jbig2", page_data, globals_data) for JBIG2
        - ("ccitt", ccitt_data, None) for CCITT G4
        - None on total failure
    """
    if jbig2enc_available():
        result = encode_jbig2_with_globals(img)
        if result is not None:
            page_data, globals_data = result
            return "jbig2", page_data, globals_data or None

    ccitt_result = encode_ccitt_g4(img)
    if ccitt_result is not None:
        ccitt_data, width, height = ccitt_result
        return "ccitt", ccitt_data, None

    return None
