"""Recovering vertical text that the recogniser read upside-down.

RapidOCR crops each detected region with ``get_rotate_crop_image``, which turns
a tall crop into a wide one with ``np.rot90`` -- always counter-clockwise. Text
set bottom-to-top therefore arrives correctly; text set top-to-bottom, which is
the common case for the side captions on Brazilian registry and certificate
pages, arrives rotated 180 degrees and comes back as noise.

The text-line classifier is the mechanism meant to catch that, and does not:
its labels are 0 and 180 degrees, but on these narrow captions it is not
confident enough to flip them. Measured on a real certificate, a caption read
as ``'se---/..-e: z2'`` at score 0.52 becomes
``'seguir: https://assinador-web.onr.org.br/docs/...'`` at score 1.00 once the
crop is turned the other way.

So the rotation is decided from the result instead of trusted in advance:
regions that are tall and came back with weak text are recognised a second time
the other way up, and the better reading wins. Only recognition re-runs --
detection already found the region.
"""

from collections.abc import Sequence
from typing import Any

# RapidOCR rotates a crop when its height is at least this many times its
# width, so this is exactly the set of regions whose orientation it guessed.
VERTICAL_ASPECT_RATIO = 1.5

# Above this the reading is already good and a second pass would only cost
# time. Correct vertical captions in the sample scored 0.95 and above, while
# the mis-rotated ones sat between 0.5 and 0.85.
CONFIDENT_SCORE = 0.90

# Require a clear improvement before replacing a reading: recognition scores
# fluctuate slightly between runs, and swapping on noise would make the output
# depend on which way the coin landed.
MIN_SCORE_GAIN = 0.05


def is_vertical_box(box: Sequence[Sequence[float]]) -> bool:
    """Whether the region is tall enough that RapidOCR rotated its crop."""
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return False
    if not xs or not ys:
        return False
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0:
        return False
    return height >= width * VERTICAL_ASPECT_RATIO


def needs_reorientation(box: Sequence[Sequence[float]], score: float) -> bool:
    """Whether a region is worth recognising again the other way up.

    Both conditions matter: a wide region was never rotated, so its orientation
    is not in question, and a confident tall one is already right.
    """
    return is_vertical_box(box) and float(score) < CONFIDENT_SCORE


def choose_better_reading(
    original_text: str,
    original_score: float,
    rotated_text: str,
    rotated_score: float,
) -> tuple[str, float, bool]:
    """Pick between the two orientations, returning (text, score, replaced)."""
    if not rotated_text.strip():
        return original_text, original_score, False
    if float(rotated_score) < float(original_score) + MIN_SCORE_GAIN:
        return original_text, original_score, False
    return rotated_text, float(rotated_score), True


def vertical_candidates(ocr_raw: dict[str, Any]) -> list[int]:
    """Indices of the regions that should be re-recognised."""
    boxes = ocr_raw.get("boxes") or []
    scores = ocr_raw.get("scores") or []
    return [
        index
        for index, box in enumerate(boxes)
        if index < len(scores) and needs_reorientation(box, scores[index])
    ]
