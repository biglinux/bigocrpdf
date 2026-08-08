"""Small file and page helpers for the chunked OCR pipeline."""

from __future__ import annotations

from pathlib import Path


def store_rendered_ocr_size(result: dict, rendered_ocr: str) -> None:
    """Record rendered OCR image dimensions on a chunk result."""
    import cv2

    rendered_img = cv2.imread(rendered_ocr)
    if rendered_img is not None:
        result["ocr_img_h"], result["ocr_img_w"] = rendered_img.shape[:2]
        del rendered_img


def remove_rendered_chunk_ocr(rendered_ocr: str | None) -> None:
    if rendered_ocr:
        Path(rendered_ocr).unlink(missing_ok=True)


def clean_chunk_images_dir(images_dir: Path) -> None:
    for file_path in images_dir.glob("*"):
        try:
            file_path.unlink()
        except OSError:
            pass


def chunk_skip_pages(page_rotations: list) -> set[int]:
    return {
        rotation.page_number
        for rotation in page_rotations
        if rotation.deleted or not rotation.included_for_ocr
    }
