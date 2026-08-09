"""Tests for pdftotext TSV parsing."""

from subprocess import CompletedProcess
from unittest.mock import patch

from bigocrpdf.utils.tsv_parser import parse_tsv_pages


def test_empty_pdftotext_output_returns_no_pages() -> None:
    result = CompletedProcess(
        args=["pdftotext"],
        returncode=0,
        stdout="",
        stderr="",
    )

    with patch("bigocrpdf.utils.tsv_parser.subprocess.run", return_value=result):
        assert parse_tsv_pages("empty.pdf") == {}


def test_malformed_word_row_does_not_discard_later_words() -> None:
    result = CompletedProcess(
        args=["pdftotext"],
        returncode=0,
        stdout=(
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
            "5\tinvalid\t1\t1\t1\t1\t10\t20\t30\t10\t100\tbroken\n"
            "5\t2\t1\t1\t1\t1\t10\t20\t30\t10\t100\tvalid\n"
        ),
        stderr="",
    )

    with patch("bigocrpdf.utils.tsv_parser.subprocess.run", return_value=result):
        pages = parse_tsv_pages("malformed.pdf")

    assert [word.text for word in pages[2]] == ["valid"]
