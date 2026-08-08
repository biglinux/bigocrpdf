#!/usr/bin/env python3
"""Small OCR benchmark metrics shared by tooling scripts."""

from __future__ import annotations

import re
import unicodedata


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
