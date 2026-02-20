"""Contour detection and span assembly for text line analysis.

Shared infrastructure used by dewarp algorithms and skew detection:
- Adaptive threshold + morphological ops detect text contours
- PCA-based contour orientation (blob mean and tangent via SVD)
- Greedy graph matching assembles contours into horizontal text spans
- Keypoints sampled along spans at dense intervals
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Algorithm constants (from page-dewarp defaults) ──────────────────────────

ADAPTIVE_WINSZ: int = 55

TEXT_MIN_WIDTH: int = 15
TEXT_MIN_HEIGHT: int = 2
TEXT_MIN_ASPECT: float = 1.5
TEXT_MAX_THICKNESS: int = 15

SPAN_MIN_WIDTH: int = 30
SPAN_PX_PER_STEP: int = 20

EDGE_MAX_OVERLAP: float = 1.0
EDGE_MAX_LENGTH: float = 160.0
EDGE_ANGLE_COST: float = 10.0
EDGE_MAX_ANGLE: float = 7.5


# ── Contour info ─────────────────────────────────────────────────────────────


class _ContourInfo:
    """Geometric and orientation data about a single text contour."""

    __slots__ = (
        "contour",
        "rect",
        "mask",
        "center",
        "tangent",
        "angle",
        "local_xrng",
        "point0",
        "point1",
        "pred",
        "succ",
    )

    def __init__(
        self,
        contour: np.ndarray,
        rect: tuple[int, int, int, int],
        mask: np.ndarray,
        center: np.ndarray,
        tangent: np.ndarray,
    ) -> None:
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center = center
        self.tangent = tangent
        self.angle: float = float(np.arctan2(tangent[1], tangent[0]))

        # Project contour points onto tangent axis to find extent
        clx = [float(np.dot(self.tangent, pt.flatten() - self.center)) for pt in contour]
        lxmin, lxmax = min(clx), max(clx)
        self.local_xrng: tuple[float, float] = (lxmin, lxmax)
        self.point0: np.ndarray = self.center + self.tangent * lxmin
        self.point1: np.ndarray = self.center + self.tangent * lxmax
        self.pred: _ContourInfo | None = None
        self.succ: _ContourInfo | None = None

    def proj_x(self, point: np.ndarray) -> float:
        """Scalar projection of a point onto this contour's tangent axis."""
        return float(np.dot(self.tangent, point.flatten() - self.center))

    def local_overlap(self, other: _ContourInfo) -> float:
        """Measure horizontal overlap in local tangent coordinates."""
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return min(self.local_xrng[1], xmax) - max(self.local_xrng[0], xmin)


# ── Contour detection ────────────────────────────────────────────────────────


def _blob_mean_and_tangent(
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute centroid and principal orientation of a contour using moments + SVD."""
    moments = cv2.moments(contour)
    area = moments["m00"]
    if not area:
        return None
    mean_x = moments["m10"] / area
    mean_y = moments["m01"] / area
    cov = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]]) / area
    _, svd_u, _ = cv2.SVDecomp(cov)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()
    return center, tangent


def _make_tight_mask(
    contour: np.ndarray, xmin: int, ymin: int, width: int, height: int
) -> np.ndarray:
    """Create a tight binary mask of a contour within its bounding box."""
    mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    cv2.drawContours(mask, [tight_contour], contourIdx=0, color=1, thickness=-1)
    return mask


def _detect_text_contours(
    small: np.ndarray,
    pagemask: np.ndarray,
    text: bool = True,
) -> list[_ContourInfo]:
    """Detect text contours using adaptive threshold + morphological ops.

    Args:
        small: Downscaled BGR image.
        pagemask: Binary mask of the page region (excluding margins).
        text: If True, detect text blobs (dilate horizontally);
              if False, detect lines (erode to remove thin elements).

    Returns:
        List of _ContourInfo objects for valid text contours.
    """
    sgray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        src=sgray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_WINSZ,
        C=25 if text else 7,
    )

    if text:
        # Dilate horizontally to connect characters into word blobs
        kernel = np.ones((1, 9), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    else:
        # Erode to remove thin lines, keep thick table borders
        kernel = np.ones((7, 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)

    # AND with page mask to exclude margins
    mask = np.minimum(mask, pagemask)

    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cinfo_list: list[_ContourInfo] = []

    for contour in contours:
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if width < TEXT_MIN_WIDTH or height < TEXT_MIN_HEIGHT or width < TEXT_MIN_ASPECT * height:
            continue

        tight_mask = _make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        result = _blob_mean_and_tangent(contour)
        if result is None:
            continue

        center, tangent = result
        cinfo_list.append(_ContourInfo(contour, rect, tight_mask, center, tangent))

    return cinfo_list


# ── Span assembly ────────────────────────────────────────────────────────────


def _angle_dist(angle_b: float, angle_a: float) -> float:
    """Compute angular distance between two angles."""
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return abs(diff)


def _generate_candidate_edge(
    cinfo_a: _ContourInfo, cinfo_b: _ContourInfo
) -> tuple[float, _ContourInfo, _ContourInfo] | None:
    """Generate a candidate edge between two contours, scored by proximity + angle.

    Returns (score, left_contour, right_contour) or None if the pair is invalid.
    """
    # Ensure a is to the left of b
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        cinfo_a, cinfo_b = cinfo_b, cinfo_a

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = float(np.arctan2(overall_tangent[1], overall_tangent[0]))

    delta_angle = (
        max(
            _angle_dist(cinfo_a.angle, overall_angle),
            _angle_dist(cinfo_b.angle, overall_angle),
        )
        * 180
        / np.pi
    )

    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = float(np.linalg.norm(cinfo_b.point0 - cinfo_a.point1))

    if dist > EDGE_MAX_LENGTH or x_overlap > EDGE_MAX_OVERLAP or delta_angle > EDGE_MAX_ANGLE:
        return None

    score = dist + delta_angle * EDGE_ANGLE_COST
    return (score, cinfo_a, cinfo_b)


def _assemble_spans(cinfo_list: list[_ContourInfo]) -> list[list[_ContourInfo]]:
    """Assemble contours into horizontal text spans using greedy graph matching.

    A 'span' is a left-to-right chain of contours forming a text line or
    partial text line.
    """
    cinfo_list = sorted(cinfo_list, key=lambda ci: ci.rect[1])

    # Generate all candidate edges
    candidate_edges: list[tuple[float, _ContourInfo, _ContourInfo]] = []
    for i, ci_i in enumerate(cinfo_list):
        for j in range(i):
            edge = _generate_candidate_edge(ci_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # Sort by score (lower is better)
    candidate_edges.sort(key=lambda e: e[0])

    # Link contours: each contour can have at most one predecessor and one successor
    for _, cinfo_a, cinfo_b in candidate_edges:
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # Build spans by walking from each head (no predecessor) to tail
    spans: list[list[_ContourInfo]] = []
    remaining = list(cinfo_list)

    while remaining:
        cinfo: _ContourInfo | None = remaining[0]
        # Walk to head of chain
        while cinfo is not None and cinfo.pred:
            cinfo = cinfo.pred

        cur_span: list[_ContourInfo] = []
        width = 0.0

        while cinfo:
            if cinfo in remaining:
                remaining.remove(cinfo)
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            cinfo = cinfo.succ

        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    return spans


# ── Span sampling ────────────────────────────────────────────────────────────


def _sample_spans(shape: tuple[int, ...], spans: list[list[_ContourInfo]]) -> list[np.ndarray]:
    """Sample keypoints along spans at regular intervals.

    Within each contour's bounding rectangle, measures the vertical centroid
    of the mask at horizontal steps.

    Returns:
        List of arrays, each containing sampled points for one span
        in pixel coordinates, shape (N, 2) where columns are (x, y).
    """
    span_points: list[np.ndarray] = []
    for span in spans:
        contour_points: list[tuple[float, float]] = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            col_sums = cinfo.mask.sum(axis=0)
            # Skip columns with no contour pixels
            valid = col_sums > 0
            if not np.any(valid):
                continue
            means = np.zeros_like(totals, dtype=float)
            means[valid] = totals[valid] / col_sums[valid]

            xmin, ymin = cinfo.rect[:2]
            step = SPAN_PX_PER_STEP
            start = int(np.floor_divide(np.mod(len(means) - 1, step), 2))
            contour_points.extend(
                (x + xmin, means[x] + ymin) for x in range(start, len(means), step) if valid[x]
            )

        if len(contour_points) >= 2:
            pts = np.array(contour_points, dtype=np.float64)
            span_points.append(pts)

    return span_points
