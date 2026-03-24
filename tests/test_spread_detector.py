"""Tests for spread detection and splitting."""

from bigocrpdf.services.rapidocr_service.config import OCRBoxData
from bigocrpdf.utils.spread_detector import (
    _find_center_gap,
    detect_and_split_spreads,
    split_text_by_spreads,
)


def _make_box(x, width, page_num=1, y=0, text=""):
    """Create an OCRBoxData for testing."""
    return OCRBoxData(x=x, width=width, page_num=page_num, y=y, text=text, height=10)


class TestFindCenterGap:
    """Tests for _find_center_gap."""

    def test_clear_spread(self):
        # Two clusters: left (10-40%) and right (60-90%) with gap at ~50%
        boxes = [
            _make_box(10, 20),  # center=20
            _make_box(15, 15),  # center=22.5
            _make_box(60, 20),  # center=70
            _make_box(65, 15),  # center=72.5
        ]
        gap = _find_center_gap(boxes)
        assert gap is not None
        assert 40 < gap < 60

    def test_no_spread_single_column(self):
        boxes = [
            _make_box(10, 80),  # center=50
            _make_box(12, 76),  # center=50
            _make_box(15, 70),  # center=50
            _make_box(10, 80),  # center=50
        ]
        gap = _find_center_gap(boxes)
        assert gap is None

    def test_too_few_boxes(self):
        boxes = [_make_box(10, 10), _make_box(60, 10)]
        gap = _find_center_gap(boxes)
        assert gap is None

    def test_gap_too_small(self):
        # Boxes very close together — gap between consecutive centers < _MIN_GAP_PCT (4%)
        boxes = [
            _make_box(45, 2),  # center=46
            _make_box(47, 2),  # center=48
            _make_box(49, 2),  # center=50
            _make_box(51, 2),  # center=52
        ]
        gap = _find_center_gap(boxes)
        assert gap is None

    def test_gap_off_center(self):
        # Gap at ~20% — too far from center
        boxes = [
            _make_box(2, 5),  # center=4.5
            _make_box(5, 5),  # center=7.5
            _make_box(30, 30),  # center=45
            _make_box(60, 30),  # center=75
        ]
        gap = _find_center_gap(boxes)
        # Gap between 7.5 and 45 is at ~26, which is < center_lo (35)
        assert gap is None


class TestDetectAndSplitSpreads:
    """Tests for detect_and_split_spreads."""

    def test_empty_input(self):
        boxes, split_map = detect_and_split_spreads([])
        assert boxes == []
        assert split_map == {}

    def test_no_spreads_returns_unchanged(self):
        boxes = [
            _make_box(10, 80, page_num=1),
            _make_box(12, 76, page_num=1),
            _make_box(15, 70, page_num=1),
            _make_box(10, 80, page_num=1),
        ]
        result_boxes, split_map = detect_and_split_spreads(boxes)
        assert split_map == {}

    def test_spread_detected(self):
        boxes = [
            _make_box(5, 15, page_num=1),  # left: center=12.5
            _make_box(8, 12, page_num=1),  # left: center=14
            _make_box(55, 15, page_num=1),  # right: center=62.5
            _make_box(60, 12, page_num=1),  # right: center=66
        ]
        result_boxes, split_map = detect_and_split_spreads(boxes)
        assert 1 in split_map
        left_num, right_num = split_map[1]
        assert left_num < right_num

    def test_multiple_pages_some_spread(self):
        page1 = [  # spread
            _make_box(5, 15, page_num=1),
            _make_box(8, 12, page_num=1),
            _make_box(55, 15, page_num=1),
            _make_box(60, 12, page_num=1),
        ]
        page2 = [  # not spread
            _make_box(10, 80, page_num=2),
            _make_box(12, 76, page_num=2),
            _make_box(15, 70, page_num=2),
            _make_box(10, 80, page_num=2),
        ]
        boxes = page1 + page2
        result_boxes, split_map = detect_and_split_spreads(boxes)
        assert 1 in split_map
        assert 2 not in split_map


class TestSplitTextBySpreads:
    """Tests for split_text_by_spreads."""

    def test_empty_split_map(self):
        text = "Hello World"
        result = split_text_by_spreads(text, [], {})
        assert result == text

    def test_empty_boxes(self):
        text = "Hello World"
        result = split_text_by_spreads(text, [], {1: (1, 2)})
        assert result == text

    def test_spread_text_split(self):
        boxes = [
            _make_box(5, 15, page_num=1, y=10, text="Left line 1"),
            _make_box(8, 12, page_num=1, y=20, text="Left line 2"),
            _make_box(55, 15, page_num=1, y=10, text="Right line 1"),
            _make_box(60, 12, page_num=1, y=20, text="Right line 2"),
        ]
        split_map = {1: (1, 2)}
        result = split_text_by_spreads("", boxes, split_map)
        assert "Left line 1" in result
        assert "Right line 1" in result
