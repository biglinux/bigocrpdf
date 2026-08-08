from bigocrpdf.utils.column_detector import _find_valley, detect_table_region
from bigocrpdf.utils.tsv_parser import TextLine, Word


def test_find_valley_requires_one_contiguous_gap():
    coverage = [10] * 20
    coverage[5:15] = [0, 0, 10, 10, 10, 10, 10, 10, 0, 0]

    assert _find_valley(coverage, len(coverage)) is None


def test_table_region_stops_before_non_table_line():
    lines = [
        TextLine(
            [
                Word("A", 0, 0, 10, 10),
                Word("B", 50, 0, 10, 10),
                Word("C", 100, 0, 10, 10),
            ],
            0,
        ),
        TextLine(
            [Word("D", 0, 20, 10, 10), Word("E", 300, 20, 10, 10)],
            20,
        ),
    ]

    assert detect_table_region(lines, 0) == ([], 0, 0)
