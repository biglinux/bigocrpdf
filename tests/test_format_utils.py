"""Tests for format_utils module."""

from bigocrpdf.utils.format_utils import format_elapsed_time


class TestFormatElapsedTime:
    def test_zero_seconds(self):
        assert format_elapsed_time(0) == "0s"

    def test_seconds_only(self):
        assert format_elapsed_time(45) == "45s"

    def test_one_minute(self):
        assert format_elapsed_time(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert format_elapsed_time(125) == "2m 5s"

    def test_one_hour(self):
        assert format_elapsed_time(3600) == "1h 0m 0s"

    def test_hours_minutes_seconds(self):
        assert format_elapsed_time(3661) == "1h 1m 1s"

    def test_negative_treated_as_zero(self):
        assert format_elapsed_time(-5) == "0s"

    def test_boolean_treated_as_zero(self):
        assert format_elapsed_time(True) == "0s"
        assert format_elapsed_time(False) == "0s"
