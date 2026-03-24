"""
BigOcrPdf - Unified Rotation Module

Centralized handling of all PDF page rotation logic.
This module provides a single source of truth for:
- Extracting rotation info from PDFs
- Applying editor rotation modifications
- Rotating images for OCR
- Transforming OCR coordinates back to PDF space
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pikepdf

logger = logging.getLogger(__name__)


@dataclass
class PageRotation:
    """Unified rotation state for a single PDF page.

    Attributes:
        page_number: 1-indexed page number
        original_pdf_rotation: /Rotate value from original PDF (0, 90, 180, 270)
        editor_rotation: Rotation applied in editor (0, 90, 180, 270)
        deleted: Whether page is marked for deletion
        mediabox: PDF MediaBox as [x0, y0, x1, y1] or None
    """

    page_number: int
    original_pdf_rotation: int = 0
    editor_rotation: int = 0
    deleted: bool = False
    included_for_ocr: bool = True
    mediabox: list[float] | None = None

    @property
    def effective_rotation(self) -> int:
        """Combined rotation = original + editor (mod 360).

        This is the final rotation that should be applied to the PDF page.
        """
        return (self.original_pdf_rotation + self.editor_rotation) % 360

    @property
    def ocr_image_rotation(self) -> int:
        """Rotation needed to make extracted image upright for OCR.

        When we extract an image from a PDF, the image data is stored
        without rotation - the rotation is only metadata. So if the PDF
        says /Rotate 90, the image pixels are sideways and we need to
        rotate it for OCR to work correctly.

        Returns the rotation in degrees (0, 90, 180, 270) to apply to
        the extracted image before OCR.
        """
        return self.original_pdf_rotation

    @property
    def pdf_dimensions(self) -> tuple[float, float]:
        """Get PDF page dimensions from mediabox.

        Returns (width, height) in PDF points.
        """
        if self.mediabox:
            return (self.mediabox[2] - self.mediabox[0], self.mediabox[3] - self.mediabox[1])
        return (595.0, 842.0)  # Default A4


def extract_page_rotations(pdf_path: Path) -> list[PageRotation]:
    """Extract rotation info from all pages in a PDF.

    Resolves both direct and inherited /Rotate values by walking
    the page tree upward when needed.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PageRotation for each page (index = page_number - 1)
    """
    rotations = []

    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Get rotation - check direct first, then walk parent tree
            rotation = 0
            if "/Rotate" in page:
                rotation = int(page["/Rotate"])
            else:
                # Walk up the page tree to find inherited /Rotate
                node = page.get("/Parent")
                while node is not None:
                    if "/Rotate" in node:
                        rotation = int(node["/Rotate"])
                        break
                    node = node.get("/Parent")

            # Normalize to valid values
            rotation = rotation % 360
            if rotation not in (0, 90, 180, 270):
                rotation = round(rotation / 90) * 90 % 360

            # Get MediaBox (use property for inheritance)
            mediabox = None
            if hasattr(page, "mediabox") and page.mediabox:
                mediabox = [float(x) for x in page.mediabox]

            rotations.append(
                PageRotation(
                    page_number=page_num,
                    original_pdf_rotation=rotation,
                    mediabox=mediabox,
                )
            )

    return rotations


def apply_editor_modifications(
    rotations: list[PageRotation],
    page_modifications: list[dict] | None,
) -> list[PageRotation]:
    """Apply editor modifications to rotation list.

    Args:
        rotations: Base rotation info from PDF
        page_modifications: List of modifications from editor UI, e.g.:
            [{"page_number": 2, "rotation": 90, "deleted": False}, ...]

    Returns:
        Updated list of PageRotation with editor changes applied
    """
    if not page_modifications:
        return rotations

    # Create lookup for quick access
    mod_lookup = {m["page_number"]: m for m in page_modifications}

    for rot in rotations:
        if rot.page_number in mod_lookup:
            mod = mod_lookup[rot.page_number]
            rot.editor_rotation = mod.get("rotation", 0)
            rot.deleted = mod.get("deleted", False)
            rot.included_for_ocr = mod.get("included_for_ocr", True)

    return rotations


def apply_final_rotation_to_pdf(
    pdf_path: Path,
    rotations: list[PageRotation],
    start_page: int = 1,
) -> None:
    """Apply final rotation and deletions to PDF file.

    This is the last step - modifies the PDF in place to apply:
    1. Editor rotations (added to existing /Rotate)
    2. Page deletions

    Args:
        pdf_path: Path to PDF to modify
        rotations: List of PageRotation with editor modifications
        start_page: Starting page number for index mapping (for page range processing)
    """
    logger.info("Applying final rotations and deletions...")

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        # Collect pages to delete (process in reverse to preserve indices)
        pages_to_delete = []

        for i in range(len(pdf.pages)):
            page_num = start_page + i

            # Find rotation info for this page
            rot_info = None
            for r in rotations:
                if r.page_number == page_num:
                    rot_info = r
                    break

            if rot_info is None:
                continue

            # Apply editor rotation
            if rot_info.editor_rotation != 0:
                current_rot = int(pdf.pages[i].get("/Rotate", 0))
                new_rot = (current_rot + rot_info.editor_rotation) % 360
                pdf.pages[i]["/Rotate"] = new_rot
                logger.info(
                    f"Page {page_num}: {current_rot}° + {rot_info.editor_rotation}° = {new_rot}°"
                )

            # Mark for deletion (deleted or excluded from OCR)
            if rot_info.deleted or not rot_info.included_for_ocr:
                pages_to_delete.append(i)
                reason = "deleted" if rot_info.deleted else "excluded from OCR"
                logger.info(f"Page {page_num}: marked for removal ({reason})")

        # Delete in reverse order
        for idx in reversed(pages_to_delete):
            del pdf.pages[idx]
            logger.info(f"Deleted page at index {idx}")

        pdf.save(pdf_path)

    logger.info(f"Applied modifications to: {pdf_path}")
