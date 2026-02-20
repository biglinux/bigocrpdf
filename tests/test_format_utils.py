"""Tests for format_utils module."""

from bigocrpdf.utils.format_utils import format_elapsed_time, format_file_size


class TestFormatFileSize:
    def test_zero_bytes(self):
        assert format_file_size(0) == "0 B"

    def test_bytes(self):
        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(1536) == "1.50 KB"

    def test_megabytes(self):
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(15 * 1024 * 1024) == "15.0 MB"

    def test_gigabytes(self):
        assert format_file_size(1024**3) == "1.00 GB"

    def test_large_values_no_decimals(self):
        assert format_file_size(200 * 1024 * 1024) == "200 MB"

    def test_negative_returns_zero(self):
        assert format_file_size(-1) == "0 B"


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
