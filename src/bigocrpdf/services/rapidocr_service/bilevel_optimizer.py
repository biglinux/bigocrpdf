"""Post-processing optimizer for bilevel (1-bit) PDF images.

After the pipeline assembles the output PDF (via ReportLab + merge),
this module re-encodes suitable images with JBIG2 or CCITT Group 4
for dramatically smaller file sizes on scanned document pages.

Typical savings: 50-75% on text-heavy scanned pages vs JPEG/PNG.
"""

import logging
from pathlib import Path

import numpy as np
import pikepdf
from pikepdf import Dictionary, Name, Stream

from bigocrpdf.services.rapidocr_service.bilevel_analysis import (
    binarize,
)
from bigocrpdf.services.rapidocr_service.jbig2_encoder import (
    encode_ccitt_g4,
    encode_jbig2_with_globals,
    jbig2enc_available,
)

logger = logging.getLogger(__name__)


def optimize_bilevel_images(
    pdf_path: Path,
    page_encodings: dict[int, str],
    force_bilevel: bool = False,
) -> int:
    """Re-encode suitable page images as JBIG2 or CCITT G4.

    Opens the PDF, inspects each page's images, and replaces
    JPEG/PNG streams with JBIG2 (preferred) or CCITT G4 (fallback)
    when the image is suitable for bilevel compression.

    Args:
        pdf_path: Path to PDF to optimize (modified in-place).
        page_encodings: {page_num: original_encoding} from input PDF.
        force_bilevel: Convert all images to bilevel.

    Returns:
        Number of images optimized.
    """
    if not pdf_path.exists():
        return 0

    has_jbig2 = jbig2enc_available()
    if not has_jbig2:
        logger.info(
            "jbig2enc not found — using CCITT G4 fallback (install jbig2enc for better compression)"
        )

    optimized = 0

    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                xobjects = _get_page_xobjects(page)
                if not xobjects:
                    continue

                for key in list(xobjects.keys()):
                    result = _try_optimize_image(
                        pdf,
                        xobjects,
                        key,
                        page_num,
                        page_encodings,
                        force_bilevel,
                        has_jbig2,
                    )
                    if result:
                        optimized += 1

            if optimized > 0:
                pdf.save(
                    str(pdf_path),
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.generate,
                )
                logger.info(f"Optimized {optimized} bilevel images in {pdf_path.name}")

    except Exception as e:
        logger.error(f"Bilevel optimization failed: {e}")
        return 0

    return optimized


def _get_page_xobjects(page: pikepdf.Page) -> pikepdf.Dictionary | None:
    """Extract XObject dictionary from a page, or None.

    Returns the actual pikepdf Dictionary reference so mutations
    apply to the PDF in memory.
    """
    try:
        resources = page.get("/Resources")
        if resources is None:
            return None
        xobjects = resources.get("/XObject")
        return xobjects if isinstance(xobjects, pikepdf.Dictionary) and xobjects else None
    except Exception:
        return None


def _try_optimize_image(
    pdf: pikepdf.Pdf,
    xobjects: pikepdf.Dictionary,
    key: str,
    page_num: int,
    page_encodings: dict[int, str],
    force_bilevel: bool,
    has_jbig2: bool,
) -> bool:
    """Try to optimize a single image XObject to bilevel encoding.

    Returns True if image was successfully optimized.
    """
    try:
        obj = _candidate_image_stream(xobjects, key)
        if obj is None:
            return False
        if not _should_optimize_image(obj, page_num, page_encodings, force_bilevel):
            return False

        binary = _extract_binary_image(obj)
        return _embed_best_bilevel_encoding(pdf, xobjects, key, page_num, binary, has_jbig2)

    except Exception as e:
        logger.debug(f"Could not optimize image {key} on page {page_num}: {e}")

    return False


def _candidate_image_stream(
    xobjects: pikepdf.Dictionary,
    key: str,
) -> pikepdf.Stream | None:
    """Return an image XObject stream eligible for inspection."""
    obj = xobjects[key]
    if not isinstance(obj, pikepdf.Stream):
        return None
    if obj.get("/Subtype") != Name.Image:
        return None
    if obj.get("/Filter") in (Name.JBIG2Decode, Name.CCITTFaxDecode):
        return None
    return obj


def _should_optimize_image(
    obj: pikepdf.Stream,
    page_num: int,
    page_encodings: dict[int, str],
    force_bilevel: bool,
) -> bool:
    """Return whether an image should be converted to bilevel output."""
    from bigocrpdf.constants import MIN_IMAGE_DIMENSION_PX

    width = int(obj.get("/Width", 0))
    height = int(obj.get("/Height", 0))
    if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
        return False

    orig_enc = page_encodings.get(page_num, "")
    return force_bilevel or orig_enc in ("jbig2", "ccitt")


def _extract_binary_image(obj: pikepdf.Stream) -> np.ndarray | None:
    """Extract and binarize a PDF image stream."""
    pil_img = _extract_pil_image(obj)
    if pil_img is None:
        return None
    gray = np.array(pil_img.convert("L"))
    return binarize(gray)


def _embed_best_bilevel_encoding(
    pdf: pikepdf.Pdf,
    xobjects: pikepdf.Dictionary,
    key: str,
    page_num: int,
    binary: np.ndarray | None,
    has_jbig2: bool,
) -> bool:
    """Embed a binary image using JBIG2 when available, otherwise CCITT G4."""
    if binary is None:
        return False
    h, w = binary.shape

    if has_jbig2:
        result = encode_jbig2_with_globals(binary)
        if result is not None:
            page_data, globals_data = result
            _embed_jbig2(pdf, xobjects, key, page_data, globals_data, w, h)
            logger.debug(
                f"Page {page_num} image {key}: JBIG2 {len(page_data) + len(globals_data)} bytes"
            )
            return True

    ccitt_result = encode_ccitt_g4(binary)
    if ccitt_result is None:
        return False
    ccitt_data, cw, ch = ccitt_result
    _embed_ccitt(pdf, xobjects, key, ccitt_data, cw, ch)
    logger.debug(f"Page {page_num} image {key}: CCITT G4 {len(ccitt_data)} bytes")
    return True


def _extract_pil_image(obj: pikepdf.Stream):
    """Extract a PIL Image from a pikepdf image stream."""
    try:
        pdf_image = pikepdf.PdfImage(obj)
        return pdf_image.as_pil_image()
    except Exception:
        return None


def _embed_jbig2(
    pdf: pikepdf.Pdf,
    xobjects: pikepdf.Dictionary,
    key: str,
    page_data: bytes,
    globals_data: bytes,
    width: int,
    height: int,
) -> None:
    """Replace an image XObject with JBIG2-encoded data."""
    new_img = Stream(pdf, page_data)
    new_img["/Type"] = Name.XObject
    new_img["/Subtype"] = Name.Image
    new_img["/Width"] = width
    new_img["/Height"] = height
    new_img["/BitsPerComponent"] = 1
    new_img["/ColorSpace"] = Name.DeviceGray
    new_img["/Filter"] = Name.JBIG2Decode

    if globals_data:
        globals_stream = Stream(pdf, globals_data)
        new_img["/DecodeParms"] = Dictionary({"/JBIG2Globals": pdf.make_indirect(globals_stream)})

    xobjects[key] = new_img


def _embed_ccitt(
    pdf: pikepdf.Pdf,
    xobjects: pikepdf.Dictionary,
    key: str,
    ccitt_data: bytes,
    width: int,
    height: int,
) -> None:
    """Replace an image XObject with CCITT Group 4 encoded data."""
    new_img = Stream(pdf, ccitt_data)
    new_img["/Type"] = Name.XObject
    new_img["/Subtype"] = Name.Image
    new_img["/Width"] = width
    new_img["/Height"] = height
    new_img["/BitsPerComponent"] = 1
    new_img["/ColorSpace"] = Name.DeviceGray
    new_img["/Filter"] = Name.CCITTFaxDecode
    new_img["/DecodeParms"] = Dictionary(
        {
            "/K": -1,
            "/Columns": width,
            "/Rows": height,
        }
    )

    xobjects[key] = new_img
