"""Measuring how much of a known distortion a corrector removed.

Not a test module -- imported, not collected.

Correctors legitimately change the image size: a four-point transform reshapes
the page, and border trimming crops it. So comparing pixels against the
original is invalid, and the metrics here are built to be invariant to
resampling and cropping:

* ``registration_residual`` resizes both to a common height and recovers the
  transform that still separates them, reporting rotation, scale and a corner
  error normalised by width.
* ``baseline_curvature_px`` measures an image on its own terms, which is the
  only workable oracle for dewarp -- dewarp changes global geometry by design,
  so registration alone cannot judge it. It is monotonic in the true bow rather
  than equal to it, so assertions on it should be relative.
* ``text_pixel_retention`` guards against a "correction" that improves every
  other number by cropping the text away. Use it as a floor only: resampling
  antialiases glyph edges and inflates the ink count well above 1.0, so an
  increase means nothing while a decrease means lost text.

SSIM is implemented here rather than taken from ``cv2.quality``: that module is
part of opencv-contrib, which the project does not depend on. Where it happens
to be available, ``test_geometry_recovery`` cross-checks the two.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Common working height for registration. Large enough for ECC to converge on
# text, small enough to keep the whole suite in seconds.
_REGISTRATION_HEIGHT = 600


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Mean structural similarity over an 11x11 Gaussian window."""
    left = _gray(first).astype(np.float64)
    right = _gray(second).astype(np.float64)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = (11, 11)
    mu_left = cv2.GaussianBlur(left, kernel, 1.5)
    mu_right = cv2.GaussianBlur(right, kernel, 1.5)
    mu_left_sq, mu_right_sq = mu_left**2, mu_right**2
    mu_cross = mu_left * mu_right

    sigma_left = cv2.GaussianBlur(left * left, kernel, 1.5) - mu_left_sq
    sigma_right = cv2.GaussianBlur(right * right, kernel, 1.5) - mu_right_sq
    sigma_cross = cv2.GaussianBlur(left * right, kernel, 1.5) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    return float(np.mean(numerator / denominator))


@dataclass(frozen=True)
class RegistrationResult:
    converged: bool
    residual_rotation_deg: float
    residual_scale: float
    rms_corner_error_px: float


def _resize_to_common_height(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    scale = _REGISTRATION_HEIGHT / gray.shape[0]
    return cv2.resize(
        gray,
        (max(1, int(round(gray.shape[1] * scale))), _REGISTRATION_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def registration_residual(corrected: np.ndarray, reference: np.ndarray) -> RegistrationResult:
    """How far ``corrected`` still is from ``reference`` geometrically.

    Both are scaled to a common height first, so a corrector that changed the
    output size is not punished for it. A failure to converge is itself a
    signal and is reported rather than raised.
    """
    moving = _resize_to_common_height(corrected)
    fixed = _resize_to_common_height(reference)
    if moving.shape != fixed.shape:
        canvas = np.full(fixed.shape, 255, dtype=moving.dtype)
        rows = min(fixed.shape[0], moving.shape[0])
        cols = min(fixed.shape[1], moving.shape[1])
        canvas[:rows, :cols] = moving[:rows, :cols]
        moving = canvas

    warp = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            fixed.astype(np.float32),
            moving.astype(np.float32),
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            5,
        )
        converged = True
    except cv2.error:
        converged = False

    rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    scale = float(np.hypot(warp[0, 0], warp[1, 0]))

    height, width = fixed.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, warp).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((mapped - corners.reshape(-1, 2)) ** 2, axis=1))))

    return RegistrationResult(
        converged=converged,
        residual_rotation_deg=rotation,
        residual_scale=scale,
        rms_corner_error_px=rms / width,
    )


def _ink_row_profile(binary: np.ndarray) -> np.ndarray:
    """Ink per row: the vertical signature of the text lines in a strip."""
    return binary.sum(axis=1).astype(np.float64)


def _profile_shift(profile: np.ndarray, reference: np.ndarray, limit: int) -> float:
    """Vertical offset, in rows, that best aligns ``profile`` with ``reference``."""
    best_shift, best_score = 0, -np.inf
    for shift in range(-limit, limit + 1):
        shifted = np.roll(profile, shift)
        if shift > 0:
            shifted[:shift] = 0
        elif shift < 0:
            shifted[shift:] = 0
        score = float(np.dot(shifted, reference))
        if score > best_score:
            best_score, best_shift = score, shift
    return float(best_shift)


def baseline_curvature_px(image: np.ndarray, strips: int = 9) -> float:
    """How far text rows bow across the page, in pixels.

    The primary dewarp oracle, and defined the same way ``dewarp_probmap``
    defines curvature: the maximum vertical deviation of a baseline from
    straight.

    Measured by splitting the page into vertical strips, taking each strip's
    ink-per-row profile, and finding the shift that best aligns it with the
    centre strip. On a straight page every shift is zero; on a bowed page the
    shifts trace the bow. Working on whole strips rather than on individual
    glyphs avoids the row-grouping problem that makes per-glyph line fitting
    unreliable on exactly the curved pages it would need to measure.
    """
    gray = _gray(image)
    binary = (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0).astype(
        np.uint8
    )
    height, width = binary.shape[:2]
    if height < 32 or width < strips * 8:
        return 0.0

    edges = np.linspace(0, width, strips + 1, dtype=int)
    profiles = [_ink_row_profile(binary[:, edges[i] : edges[i + 1]]) for i in range(strips)]
    reference = profiles[strips // 2]
    if reference.sum() == 0:
        return 0.0

    limit = max(4, int(height * 0.08))
    shifts = [
        _profile_shift(profile, reference, limit)
        for profile in profiles
        if profile.sum() > reference.sum() * 0.05
    ]
    if len(shifts) < 3:
        return 0.0
    return float(max(shifts) - min(shifts)) / 2.0


def text_block_taper(image: np.ndarray) -> float:
    """Ratio of the text block's width at the top to its width at the bottom.

    The perspective oracle. A keystoned page has systematically wider text at
    one end; correcting it drives this to 1.0. Unlike image registration it
    needs no correspondence between two images, so it stays meaningful when the
    corrector changed the output size.
    """
    gray = _gray(image)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    height = binary.shape[0]

    def band_width(start: float, end: float) -> float:
        band = binary[int(height * start) : int(height * end)]
        columns = np.flatnonzero(band.any(axis=0))
        return float(columns[-1] - columns[0]) if columns.size >= 2 else 0.0

    top = band_width(0.15, 0.35)
    bottom = band_width(0.65, 0.85)
    if top <= 0 or bottom <= 0:
        return 1.0
    return top / bottom


def text_pixel_retention(corrected: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of the reference's ink still present after correction.

    A correction that crops text away can improve every geometric metric while
    destroying the document, and nothing else here would notice.
    """
    reference_ink = int(
        np.count_nonzero(
            cv2.threshold(_gray(reference), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        )
    )
    if reference_ink == 0:
        return 1.0
    corrected_ink = int(
        np.count_nonzero(
            cv2.threshold(_gray(corrected), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        )
    )
    return corrected_ink / reference_ink
