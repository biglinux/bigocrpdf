#!/usr/bin/env python3
"""Small OCR benchmark metrics shared by tooling scripts."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence


def levenshtein(left: str | list[str], right: str | list[str]) -> int:
    """Return Levenshtein distance for strings or token lists."""
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def normalize_for_ocr_metric(text: str) -> str:
    """Normalize whitespace while preserving accents and script identity."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def char_error_rate(predicted: str, ground_truth: str) -> float:
    """Return character error rate without accent stripping."""
    normalized_predicted = normalize_for_ocr_metric(predicted)
    normalized_ground_truth = normalize_for_ocr_metric(ground_truth)
    return levenshtein(normalized_predicted, normalized_ground_truth) / max(
        len(normalized_ground_truth),
        1,
    )


def word_error_rate(predicted: str, ground_truth: str) -> float:
    """Return word error rate."""
    predicted_words = normalize_for_ocr_metric(predicted).split()
    ground_truth_words = normalize_for_ocr_metric(ground_truth).split()
    return levenshtein(predicted_words, ground_truth_words) / max(len(ground_truth_words), 1)


def levenshtein_ratio(predicted: str, ground_truth: str) -> float:
    """Return similarity ratio in the 0..1 range."""
    normalized_predicted = normalize_for_ocr_metric(predicted)
    normalized_ground_truth = normalize_for_ocr_metric(ground_truth)
    max_len = max(len(normalized_predicted), len(normalized_ground_truth), 1)
    return 1.0 - levenshtein(normalized_predicted, normalized_ground_truth) / max_len


def sorted_line_char_error_rate(predicted: str, ground_truth: str) -> float:
    """Character error rate that ignores the order of whole lines.

    A two-column page has genuinely ambiguous reading order: a correct read
    that interleaves the columns differently from the ground truth is not an
    OCR error, but plain CER punishes it as one. Sorting both line lists before
    comparing removes the order while keeping every character exact, so these
    samples can be gated without loosening the threshold.

    Report the ordinary CER alongside it: a real column-order regression should
    still be visible, just not fatal.
    """
    predicted_lines = sorted(normalize_for_ocr_metric(predicted).splitlines())
    ground_truth_lines = sorted(normalize_for_ocr_metric(ground_truth).splitlines())
    return char_error_rate("\n".join(predicted_lines), "\n".join(ground_truth_lines))


def micro_char_error_rate(pairs: Sequence[tuple[str, str]]) -> float:
    """Corpus-level CER: total edit distance over total ground-truth length.

    Averaging per-sample CER weights a one-line receipt the same as an
    eighteen-page contract, which is how a regression on long documents hides
    behind a corpus of short ones.
    """
    total_distance = 0
    total_length = 0
    for predicted, ground_truth in pairs:
        normalized_predicted = normalize_for_ocr_metric(predicted)
        normalized_ground_truth = normalize_for_ocr_metric(ground_truth)
        total_distance += levenshtein(normalized_predicted, normalized_ground_truth)
        total_length += len(normalized_ground_truth)
    return total_distance / max(total_length, 1)


def aggregate_confidence(values: Sequence[float]) -> dict[str, float | None]:
    """Summarise per-region confidences.

    ``p10`` and ``min`` are the load-bearing ones: a mean of 0.95 says nothing
    about the page that came back at 0.30, and that page is the failure a user
    would actually notice.
    """
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"mean": None, "median": None, "p10": None, "min": None, "count": 0}
    return {
        "mean": sum(numbers) / len(numbers),
        "median": _percentile(numbers, 50.0),
        "p10": _percentile(numbers, 10.0),
        "min": numbers[0],
        "count": len(numbers),
    }


def _percentile(sorted_values: list[float], percent: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percent / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def quad_to_bbox(quad: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of an OCR quadrilateral, as (x0, y0, x1, y1)."""
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Intersection over union of two axis-aligned boxes.

    Axis-aligned only. Rotated-polygon IoU needs a clipping algorithm, is a
    steady source of subtle bugs, and buys nothing here: the ground truth this
    is measured against is itself axis-aligned.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    predicted: Sequence[tuple[float, float, float, float]],
    ground_truth: Sequence[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedily pair predicted boxes to ground-truth boxes by descending IoU.

    Greedy rather than optimal assignment: it is the convention for OCR
    detection evaluation, it is deterministic, and it is simple enough to
    verify by hand -- all of which matter more here than the last fraction of a
    percent an optimal matcher would recover.

    Returns (predicted index, ground-truth index, IoU) for each accepted pair.
    """
    candidates = sorted(
        (
            (box_iou(predicted_box, truth_box), predicted_index, truth_index)
            for predicted_index, predicted_box in enumerate(predicted)
            for truth_index, truth_box in enumerate(ground_truth)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for iou, predicted_index, truth_index in candidates:
        if iou < iou_threshold:
            break
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        matches.append((predicted_index, truth_index, iou))
    return matches


def detection_scores(
    predicted: Sequence[tuple[float, float, float, float]],
    ground_truth: Sequence[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall and F1 for text detection.

    Text that is found but misplaced reads as correct to a CER metric only if
    the extractor happens to order it the same way; these catch the detector
    drifting without waiting for that coincidence to break.
    """
    matches = match_boxes(predicted, ground_truth, iou_threshold)
    matched = len(matches)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": float(matched),
        "false_positives": float(len(predicted) - matched),
        "false_negatives": float(len(ground_truth) - matched),
    }
