"""The measuring instrument for every OCR quality claim in this project.

``ocr_metrics`` decides whether a change improved or degraded OCR, and it had
no tests of its own -- so a defect in it would silently move every threshold
that depends on it.
"""

import math

import pytest

from benchmarks.ocr_metrics import (
    aggregate_confidence,
    box_iou,
    char_error_rate,
    detection_scores,
    levenshtein,
    levenshtein_ratio,
    match_boxes,
    micro_char_error_rate,
    normalize_for_ocr_metric,
    quad_to_bbox,
    sorted_line_char_error_rate,
    word_error_rate,
)


class TestLevenshtein:
    @pytest.mark.parametrize(
        "left,right,expected",
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("abc", "", 3),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
        ],
    )
    def test_string_distance(self, left, right, expected):
        assert levenshtein(left, right) == expected

    def test_token_lists_measure_whole_words(self):
        """The same function doubles as the word-level distance."""
        assert levenshtein(["um", "dois", "tres"], ["um", "tres"]) == 1

    def test_it_is_symmetric(self):
        assert levenshtein("relatorio", "relatoria") == levenshtein("relatoria", "relatorio")


class TestNormalisation:
    def test_decomposed_accents_compare_equal_to_composed(self):
        """pdftotext may emit NFD where the ground truth is NFC."""
        assert normalize_for_ocr_metric("inspeção") == normalize_for_ocr_metric("inspeção")

    def test_accents_are_preserved(self):
        """Stripping them would hide the errors this project most cares about."""
        assert "ç" in normalize_for_ocr_metric("inspeção")
        assert normalize_for_ocr_metric("inspeção") != normalize_for_ocr_metric("inspecao")

    def test_case_is_preserved(self):
        assert normalize_for_ocr_metric("CAIXA") != normalize_for_ocr_metric("caixa")

    def test_runs_of_spaces_and_tabs_collapse(self):
        assert normalize_for_ocr_metric("a  \t b") == "a b"

    def test_a_non_breaking_space_becomes_a_space(self):
        assert normalize_for_ocr_metric("a b") == "a b"

    def test_blank_line_runs_collapse_to_one(self):
        assert normalize_for_ocr_metric("a\n\n\n\n\nb") == "a\n\nb"

    def test_surrounding_whitespace_is_dropped(self):
        assert normalize_for_ocr_metric("  a  ") == "a"


class TestErrorRates:
    def test_a_perfect_read_scores_zero(self):
        assert char_error_rate("Relatorio tecnico", "Relatorio tecnico") == 0.0
        assert word_error_rate("Relatorio tecnico", "Relatorio tecnico") == 0.0

    def test_an_empty_ground_truth_does_not_divide_by_zero(self):
        assert char_error_rate("spurious", "") == pytest.approx(8.0)

    def test_a_missed_page_scores_one(self):
        """The shape of the real failure: nothing extracted at all."""
        assert char_error_rate("", "texto original") == pytest.approx(1.0)

    def test_pure_insertion_can_exceed_one(self):
        """Uncapped on purpose -- a hallucinating engine should look worse."""
        assert char_error_rate("abcabcabc", "abc") > 1.0

    def test_word_rate_counts_words_not_characters(self):
        assert word_error_rate("um dois quatro", "um dois tres") == pytest.approx(1 / 3)

    def test_ratio_is_the_complement_of_relative_distance(self):
        assert levenshtein_ratio("abc", "abc") == 1.0
        assert levenshtein_ratio("", "abc") == 0.0
        assert 0.0 < levenshtein_ratio("abcd", "abce") < 1.0


class TestSortedLineErrorRate:
    def test_reordered_lines_are_forgiven(self):
        """Two-column reading order is ambiguous; character accuracy is not."""
        ground_truth = "coluna esquerda\ncoluna direita"
        swapped = "coluna direita\ncoluna esquerda"

        assert sorted_line_char_error_rate(swapped, ground_truth) == 0.0
        assert char_error_rate(swapped, ground_truth) > 0.0

    def test_real_character_errors_still_count(self):
        assert sorted_line_char_error_rate("coluna esqverda", "coluna esquerda") > 0.0


class TestMicroErrorRate:
    def test_it_weights_by_length_not_by_sample(self):
        """One long document must not be outvoted by many short ones.

        A wholly missed 100-character document among twenty perfect two-letter
        ones: 100 edits over 140 ground-truth characters. Averaging per sample
        buries the same failure at under 5%.
        """
        pairs = [("", "x" * 100)] + [("ok", "ok")] * 20

        micro = micro_char_error_rate(pairs)
        macro = sum(char_error_rate(p, g) for p, g in pairs) / len(pairs)

        assert micro == pytest.approx(100 / 140)
        assert macro < 0.05

    def test_an_empty_corpus_is_zero(self):
        assert micro_char_error_rate([]) == 0.0


class TestAggregateConfidence:
    def test_an_empty_page_reports_nothing_rather_than_zero(self):
        """Zero confidence and no measurement are different facts."""
        assert aggregate_confidence([]) == {
            "mean": None,
            "median": None,
            "p10": None,
            "min": None,
            "count": 0,
        }

    def test_the_worst_region_survives_a_high_mean(self):
        """A single bad region is invisible in the mean but not in the minimum."""
        summary = aggregate_confidence([0.99] * 19 + [0.30])

        assert summary["mean"] > 0.95
        assert summary["min"] == pytest.approx(0.30)

    def test_p10_reports_the_bottom_tenth(self):
        """Nearest-rank, so it only moves once a tenth of the page is poor.

        That is the intended sensitivity: ``min`` catches one bad region, p10
        catches a systematically bad page.
        """
        summary = aggregate_confidence([0.99] * 18 + [0.30, 0.31])

        assert summary["p10"] == pytest.approx(0.31)
        assert summary["min"] == pytest.approx(0.30)

    def test_the_median_is_the_middle_value(self):
        assert aggregate_confidence([0.1, 0.5, 0.9])["median"] == pytest.approx(0.5)


class TestBoxGeometry:
    def test_a_quad_reduces_to_its_bounds(self):
        quad = [[10, 20], [110, 25], [108, 60], [8, 55]]

        assert quad_to_bbox(quad) == (8.0, 20.0, 110.0, 60.0)

    @pytest.mark.parametrize(
        "first,second,expected",
        [
            ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
            ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),
            ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),
            ((0, 0, 10, 10), (0, 0, 5, 10), 0.5),
        ],
    )
    def test_iou(self, first, second, expected):
        assert box_iou(first, second) == pytest.approx(expected)

    def test_a_zero_area_box_does_not_divide_by_zero(self):
        assert box_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_a_nested_box_scores_its_area_ratio(self):
        assert box_iou((0, 0, 10, 10), (2, 2, 8, 8)) == pytest.approx(36 / 100)


class TestMatching:
    def test_each_box_is_used_at_most_once(self):
        predicted = [(0, 0, 10, 10), (1, 1, 11, 11)]
        ground_truth = [(0, 0, 10, 10)]

        matches = match_boxes(predicted, ground_truth)

        assert len(matches) == 1
        assert matches[0][0] == 0

    def test_pairs_below_the_threshold_are_rejected(self):
        assert match_boxes([(0, 0, 10, 10)], [(9, 9, 19, 19)]) == []

    def test_matching_is_deterministic(self):
        predicted = [(0, 0, 10, 10), (0, 0, 10, 10)]
        ground_truth = [(0, 0, 10, 10), (0, 0, 10, 10)]

        assert match_boxes(predicted, ground_truth) == match_boxes(predicted, ground_truth)


class TestDetectionScores:
    def test_a_hand_computed_case(self):
        """Three predictions, two of which are right, against two truths."""
        predicted = [(0, 0, 10, 10), (20, 20, 30, 30), (50, 50, 60, 60)]
        ground_truth = [(0, 0, 10, 10), (20, 20, 30, 30)]

        scores = detection_scores(predicted, ground_truth)

        assert scores["precision"] == pytest.approx(2 / 3)
        assert scores["recall"] == pytest.approx(1.0)
        assert scores["f1"] == pytest.approx(0.8)
        assert scores["false_positives"] == 1.0
        assert scores["false_negatives"] == 0.0

    def test_detecting_nothing_scores_zero_without_raising(self):
        scores = detection_scores([], [(0, 0, 10, 10)])

        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0
        assert scores["false_negatives"] == 1.0

    def test_every_score_is_finite(self):
        for scores in (
            detection_scores([], []),
            detection_scores([(0, 0, 1, 1)], []),
            detection_scores([], [(0, 0, 1, 1)]),
        ):
            assert all(math.isfinite(value) for value in scores.values())
