"""When the space between two text lines starts a new paragraph."""

from statistics import median


def paragraph_gap_threshold(gap_ratios: list[float]) -> float:
    """Gap, in line heights, beyond which the next line starts a paragraph.

    ``gap_ratios`` are the page's (or column's) gaps between consecutive lines,
    each over a line height. The threshold sits above their median, so double
    spacing alone never breaks: 0.4 line heights wider, or half the usual gap
    when that is more. Fewer than three gaps say nothing about the spacing, so
    single spacing is assumed.
    """
    usual = median(gap_ratios) if len(gap_ratios) >= 3 else 0.2
    return usual + max(0.4, usual / 2)
