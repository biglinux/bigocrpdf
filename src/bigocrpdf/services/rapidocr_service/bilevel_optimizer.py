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

    return optimized


def _get_page_xobjects(page: pikepdf.Page):
    """Extract XObject dictionary from a page, or None.

    Returns the actual pikepdf Dictionary reference so mutations
    apply to the PDF in memory.
    """
    try:
        resources = page.get("/Resources")
        if resources is None:
            return None
        xobjects = resources.get("/XObject")
        return xobjects if xobjects else None
    except Exception:
        return None


def _try_optimize_image(
    pdf: pikepdf.Pdf,
    xobjects: dict,
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
        obj = xobjects[key]
        if not isinstance(obj, pikepdf.Stream):
            return False
        if obj.get("/Subtype") != Name.Image:
            return False

        # Skip if already JBIG2 or CCITT
        current_filter = obj.get("/Filter")
        if current_filter in (Name.JBIG2Decode, Name.CCITTFaxDecode):
            return False

        from bigocrpdf.constants import MIN_IMAGE_DIMENSION_PX

        width = int(obj.get("/Width", 0))
        height = int(obj.get("/Height", 0))
        if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
            return False

        # Determine if this page should be optimized
        orig_enc = page_encodings.get(page_num, "")
        was_bilevel = orig_enc in ("jbig2", "ccitt")

        if not was_bilevel and not force_bilevel:
            # Original was not bilevel (e.g. JPEG scan) — skip auto-conversion.
            # The bilevel candidate check runs on the PROCESSED image which may
            # look bilevel after auto-contrast/brightness preprocessing even
            # though the original was a continuous-tone scan.  Converting such
            # pages to 1-bit loses quality with no user intent to do so.
            return False

        pil_img = _extract_pil_image(obj)
        if pil_img is None:
            return False

        # Binarize the image
        gray = np.array(pil_img.convert("L"))
        binary = binarize(gray)
        h, w = binary.shape

        # Encode with best available method
        if has_jbig2:
            result = encode_jbig2_with_globals(binary)
            if result is not None:
                page_data, globals_data = result
                _embed_jbig2(pdf, xobjects, key, page_data, globals_data, w, h)
                logger.debug(
                    f"Page {page_num} image {key}: JBIG2 {len(page_data) + len(globals_data)} bytes"
                )
                return True

        # Fallback to CCITT G4
        ccitt_result = encode_ccitt_g4(binary)
        if ccitt_result is not None:
            ccitt_data, cw, ch = ccitt_result
            _embed_ccitt(pdf, xobjects, key, ccitt_data, cw, ch)
            logger.debug(f"Page {page_num} image {key}: CCITT G4 {len(ccitt_data)} bytes")
            return True

    except Exception as e:
        logger.debug(f"Could not optimize image {key} on page {page_num}: {e}")

    return False


def _extract_pil_image(obj: pikepdf.Stream):
    """Extract a PIL Image from a pikepdf image stream."""
    try:
        pdf_image = pikepdf.PdfImage(obj)
        return pdf_image.as_pil_image()
    except Exception:
        return None


def _embed_jbig2(
    pdf: pikepdf.Pdf,
    xobjects: dict,
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
    xobjects: dict,
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
