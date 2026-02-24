"""Backward-compatible re-exports for contour analysis.

The implementation has been split into:
- contour_spans.py  — Contour detection and span assembly infrastructure
- contour_dewarp.py — Dewarp algorithms and skew detection
"""

from bigocrpdf.services.contour_dewarp import (
    detect_skew_from_contours,
    dewarp_3d,
    dewarp_baseline,
)

__all__ = ["detect_skew_from_contours", "dewarp_3d", "dewarp_baseline"]
