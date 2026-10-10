"""Tests for text_utils module."""

import os
import tempfile
from unittest.mock import patch

import pytest

from bigocrpdf.utils.text_utils import read_text_from_sidecar


class TestReadTextFromSidecar:
    def test_empty_path_returns_none(self):
        assert read_text_from_sidecar("") is None

    def test_nonexistent_file_returns_none(self):
        assert read_text_from_sidecar("/nonexistent/file.txt") is None

    def test_symlink_is_not_read(self, tmp_path):
        protected = tmp_path / "protected.txt"
        protected.write_text("private", encoding="utf-8")
        sidecar = tmp_path / "document.txt"
        sidecar.symlink_to(protected)

        assert read_text_from_sidecar(str(sidecar)) is None

    def test_fifo_is_rejected_without_blocking(self, tmp_path):
        sidecar = tmp_path / "document.txt"
        os.mkfifo(sidecar)

        assert read_text_from_sidecar(str(sidecar)) is None

    def test_reads_utf8_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            path = f.name
        try:
            assert read_text_from_sidecar(path) == "Hello world"
        finally:
            os.unlink(path)

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            assert read_text_from_sidecar(path) is None
        finally:
            os.unlink(path)

    def test_whitespace_only_file_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  \n\t")
            path = f.name
        try:
            assert read_text_from_sidecar(path) is None
        finally:
            os.unlink(path)

    def test_unexpected_read_error_is_not_hidden(self):
        with patch(
            "bigocrpdf.utils.text_utils.read_regular_file_bytes",
            side_effect=RuntimeError("broken reader"),
        ):
            with pytest.raises(RuntimeError, match="broken reader"):
                read_text_from_sidecar("sidecar.txt")

    def test_reads_latin1_fallback(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"\xe9\xe0\xfc")  # Latin-1 chars that are not valid UTF-8
            path = f.name
        try:
            result = read_text_from_sidecar(path)
            assert result is not None
            assert "é" in result
        finally:
            os.unlink(path)
