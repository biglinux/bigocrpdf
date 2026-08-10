"""Deciding when a tall OCR region was read upside-down.

RapidOCR turns every tall crop the same way (``np.rot90``, counter-clockwise),
so vertical captions that run top-to-bottom come back rotated 180 degrees and
recognise as noise. The text-line classifier is supposed to catch that and, on
these narrow captions, does not.

Measured on a real certificate page: a caption read as ``'se---/..-e: z2'`` at
score 0.52 became ``'seguir: https://assinador-web.onr.org.br/docs/...'`` at
1.00 once the crop was turned the other way. Across that page the mean score of
vertical regions went from 0.749 to 0.991.

This file covers the decision; the image work lives in the OCR worker, where
the engine is.
"""

import pytest

from bigocrpdf.services.rapidocr_service.vertical_text import (
    CONFIDENT_SCORE,
    MIN_SCORE_GAIN,
    VERTICAL_ASPECT_RATIO,
    choose_better_reading,
    is_vertical_box,
    needs_reorientation,
    vertical_candidates,
)


def _box(width: float, height: float, left: float = 10.0, top: float = 20.0):
    return [
        [left, top],
        [left + width, top],
        [left + width, top + height],
        [left, top + height],
    ]


class TestIsVerticalBox:
    def test_a_text_line_is_not_vertical(self):
        assert is_vertical_box(_box(400, 30)) is False

    def test_a_tall_caption_is_vertical(self):
        """The 58x1440 side caption from the sample certificate."""
        assert is_vertical_box(_box(58, 1440)) is True

    @pytest.mark.parametrize("ratio,expected", [(1.4, False), (1.5, True), (1.6, True)])
    def test_the_threshold_matches_rapidocr(self, ratio, expected):
        """RapidOCR rotates at h/w >= 1.5, so that is exactly the set of
        regions whose orientation it guessed."""
        assert is_vertical_box(_box(100, 100 * ratio)) is expected
        assert VERTICAL_ASPECT_RATIO == 1.5

    def test_a_square_region_is_not_vertical(self):
        assert is_vertical_box(_box(100, 100)) is False

    def test_a_zero_width_box_is_rejected_rather_than_dividing(self):
        assert is_vertical_box([[10, 20], [10, 20], [10, 900], [10, 900]]) is False

    @pytest.mark.parametrize(
        "box",
        [
            pytest.param([], id="empty"),
            pytest.param([[1, 2], [3]], id="short-point"),
            pytest.param([["a", "b"], [1, 2], [3, 4], [5, 6]], id="non-numeric"),
            pytest.param(None, id="none"),
        ],
    )
    def test_malformed_boxes_are_rejected(self, box):
        assert is_vertical_box(box) is False

    def test_a_rotated_quadrilateral_uses_its_extent(self):
        skewed = [[100, 100], [130, 105], [125, 900], [95, 895]]

        assert is_vertical_box(skewed) is True


class TestNeedsReorientation:
    def test_a_weak_tall_region_qualifies(self):
        assert needs_reorientation(_box(58, 1440), 0.52) is True

    def test_a_confident_tall_region_is_left_alone(self):
        """Correct vertical captions scored 0.95 and above in the sample."""
        assert needs_reorientation(_box(58, 1440), 0.98) is False

    def test_a_weak_wide_region_is_left_alone(self):
        """A wide crop was never rotated, so its orientation is not in doubt.

        Low confidence there means hard text, and re-reading it upside-down
        would only invite a wrong answer.
        """
        assert needs_reorientation(_box(400, 30), 0.40) is False

    def test_the_threshold_sits_between_the_two_populations(self):
        assert 0.85 < CONFIDENT_SCORE < 0.95


class TestChooseBetterReading:
    def test_a_clearly_better_reading_wins(self):
        text, score, replaced = choose_better_reading("se---/..-e: z2", 0.52, "seguir: https", 1.0)

        assert (text, score, replaced) == ("seguir: https", 1.0, True)

    def test_a_worse_reading_is_discarded(self):
        text, score, replaced = choose_better_reading("assinado", 0.95, "opeuisse", 0.40)

        assert (text, score, replaced) == ("assinado", 0.95, False)

    def test_a_marginally_better_reading_is_not_worth_a_coin_flip(self):
        """Scores wobble between runs; swapping on noise makes output random."""
        text, _score, replaced = choose_better_reading("original", 0.80, "alternativa", 0.82)

        assert (text, replaced) == ("original", False)
        assert MIN_SCORE_GAIN > 0.02

    def test_an_empty_rotated_reading_never_replaces(self):
        text, score, replaced = choose_better_reading("algum texto", 0.60, "   ", 0.99)

        assert (text, score, replaced) == ("algum texto", 0.60, False)


class TestVerticalCandidates:
    def test_only_weak_tall_regions_are_selected(self):
        ocr_raw = {
            "boxes": [_box(400, 30), _box(58, 1440), _box(58, 1440), _box(400, 30)],
            "txts": ["linha", "ruim", "boa", "outra"],
            "scores": [0.99, 0.52, 0.98, 0.40],
        }

        assert vertical_candidates(ocr_raw) == [1]

    def test_a_page_without_vertical_text_costs_nothing(self):
        ocr_raw = {"boxes": [_box(400, 30)] * 20, "txts": ["a"] * 20, "scores": [0.5] * 20}

        assert vertical_candidates(ocr_raw) == []

    def test_missing_keys_are_tolerated(self):
        assert vertical_candidates({}) == []
        assert vertical_candidates({"boxes": [_box(58, 1440)]}) == []

    def test_scores_shorter_than_boxes_do_not_raise(self):
        ocr_raw = {"boxes": [_box(58, 1440), _box(58, 1440)], "scores": [0.5]}

        assert vertical_candidates(ocr_raw) == [0]
