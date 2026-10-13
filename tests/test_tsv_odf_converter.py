"""Paragraphs rebuilt from pdftotext word boxes."""

import pytest

from bigocrpdf.utils.tsv_odf_converter import process_page
from bigocrpdf.utils.tsv_parser import Word

# Word height and line pitch pdftotext reports for the text layer of 12 pt
# single-spaced body text.
HEIGHT = 10.0
PITCH = 12.0


@pytest.mark.parametrize(
    ("pitches", "paragraph_sizes"),
    [
        ([1, 1, 2, 1, 1], [3, 3]),  # single-spaced, one paragraph gap
        ([2, 2, 2, 2, 2], [6]),  # double-spaced, no paragraph gap
        ([2, 2, 4, 2, 2], [3, 3]),  # double-spaced, one paragraph gap
    ],
)
def test_paragraph_gap_relative_to_line_spacing(pitches, paragraph_sizes):
    lines = [f"line {index} of flowing body text that fills the column" for index in range(6)]
    words, top = [], 100.0
    for index, text in enumerate(lines):
        words.append(Word(text=text, left=72.0, top=top, width=468.0, height=HEIGHT))
        if index < len(pitches):
            top += pitches[index] * PITCH

    expected, start = [], 0
    for size in paragraph_sizes:
        expected.append(" ".join(lines[start : start + size]))
        start += size
    assert [element.text for element in process_page(words)] == expected
