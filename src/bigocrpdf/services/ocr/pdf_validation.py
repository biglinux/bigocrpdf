"""Validation helpers for searchable PDF text extraction."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PdfOcrValidation:
    """Result of validating extracted searchable PDF text."""

    extracted_chars: int
    suspicious_ratio: float
    ok: bool
    reason: str = ""


@dataclass
class TextLayerQuality:
    """Trust classification for existing native PDF text."""

    status: Literal["absent", "rejected", "trusted"]
    chars: int = 0
    fffd_ratio: float = 0.0
    nonprint_ratio: float = 0.0
    suspicious_ratio: float = 0.0
    reason: str = ""


def classify_text_layer(text: str, page_area_pts: float = 0.0) -> TextLayerQuality:
    stripped = text.strip()
    if not stripped:
        return TextLayerQuality("absent", reason="empty")

    chars = len(stripped)
    if chars < 10:
        return TextLayerQuality("rejected", chars=chars, reason="too_few_chars")

    fffd_ratio = stripped.count("�") / chars
    nonprint_ratio = sum(1 for char in stripped if ord(char) < 32 and char not in "\n\r\t") / chars
    suspicious_ratio = _suspicious_glyph_ratio(stripped)
    if fffd_ratio > 0.01:
        return TextLayerQuality(
            "rejected",
            chars=chars,
            fffd_ratio=fffd_ratio,
            nonprint_ratio=nonprint_ratio,
            suspicious_ratio=suspicious_ratio,
            reason="fffd_ratio",
        )
    if nonprint_ratio > 0.02:
        return TextLayerQuality(
            "rejected",
            chars=chars,
            fffd_ratio=fffd_ratio,
            nonprint_ratio=nonprint_ratio,
            suspicious_ratio=suspicious_ratio,
            reason="nonprint_ratio",
        )
    if suspicious_ratio > 0.03:
        return TextLayerQuality(
            "rejected",
            chars=chars,
            fffd_ratio=fffd_ratio,
            nonprint_ratio=nonprint_ratio,
            suspicious_ratio=suspicious_ratio,
            reason="suspicious_glyph_loss",
        )

    if page_area_pts > 0:
        chars_per_1000_pts = chars / max(page_area_pts / 1000.0, 1.0)
        if chars_per_1000_pts > 15:
            return TextLayerQuality(
                "rejected",
                chars=chars,
                fffd_ratio=fffd_ratio,
                nonprint_ratio=nonprint_ratio,
                suspicious_ratio=suspicious_ratio,
                reason="implausible_text_density",
            )

    return TextLayerQuality(
        "trusted",
        chars=chars,
        fffd_ratio=fffd_ratio,
        nonprint_ratio=nonprint_ratio,
        suspicious_ratio=suspicious_ratio,
    )


def validate_searchable_pdf_text(text: str) -> PdfOcrValidation:
    """Validate that extracted PDF text is present and not mostly glyph loss."""
    total = len(text.strip())
    if total == 0:
        return PdfOcrValidation(0, 1.0, False, "empty extracted text")

    ratio = _suspicious_glyph_ratio(text)
    if ratio > 0.03:
        return PdfOcrValidation(total, ratio, False, "suspicious glyph loss")
    return PdfOcrValidation(total, ratio, True)


def _suspicious_glyph_ratio(text: str) -> float:
    suspicious = sum(1 for char in text if char in "�□")
    question_runs = text.count("???")
    return (suspicious + question_runs * 3) / max(len(text.strip()), 1)
