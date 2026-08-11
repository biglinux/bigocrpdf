"""Geometry of the fixed-layout ODT, read back out of the file it writes.

This is the layer that would have caught the defect that prompted it: on a real
six-page export, 302 of 303 consecutive frames overlapped the next one, because
each frame was 1.8 line-heights tall while lines sit about one line-height
apart. Nothing measured the output, so nothing noticed.

It needs no office suite -- it parses ``content.xml`` -- so unlike the
round-trip test it cannot be silently skipped on a machine without LibreOffice.
"""

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest
from reportlab.pdfbase import pdfmetrics

from bigocrpdf.constants import MAX_FONT_SIZE
from bigocrpdf.utils.odf_builder import (
    LIBERATION_FIRST_BASELINE_EM,
    create_positioned_text_odf,
    positioned_font_size,
)
from bigocrpdf.utils.tsv_parser import (
    TSV_BASELINE_FRACTION,
    TSV_BOX_HEIGHT_EM,
    TextLine,
    Word,
)

NS = {
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
}

A4 = (595.28, 841.89, 21.0, 29.7)
CM_PER_POINT = 2.54 / 72.0
# Frames may reach a little into the line below without harm: they have no fill
# and run through, so only text colliding with text is visible. This is the
# descender's worth of room, not licence for the 0.24cm overlap that shipped.
OVERLAP_ALLOWANCE_CM = 0.05


class Frame:
    """One emitted frame, in the units the file states."""

    def __init__(self, element, styles):
        self.x = _cm(element.get(f"{{{NS['svg']}}}x"))
        self.y = _cm(element.get(f"{{{NS['svg']}}}y"))
        self.width = _cm(element.get(f"{{{NS['svg']}}}width"))
        self.height = _cm(element.get(f"{{{NS['svg']}}}height"))
        self.z = int(element.get(f"{{{NS['draw']}}}z-index"))
        self.page = int(element.get(f"{{{NS['text']}}}anchor-page-number"))
        paragraph = element.find(f".//{{{NS['text']}}}p")
        self.text = "".join(paragraph.itertext())
        self.font_size = styles[paragraph.get(f"{{{NS['text']}}}style-name")]

    @property
    def bottom(self) -> float:
        return self.y + self.height


def _cm(value: str) -> float:
    assert value.endswith("cm"), value
    return float(value[:-2])


def _paragraph_font_sizes(root) -> dict[str, float]:
    sizes = {}
    for style in root.iter(f"{{{NS['style']}}}style"):
        properties = style.find(f"{{{NS['style']}}}text-properties")
        if properties is None:
            continue
        size = properties.get(f"{{{NS['fo']}}}font-size")
        if size and size.endswith("pt"):
            sizes[style.get(f"{{{NS['style']}}}name")] = float(size[:-2])
    return sizes


def frames_of(path: Path) -> list[Frame]:
    root = ElementTree.fromstring(zipfile.ZipFile(path).read("content.xml"))
    styles = _paragraph_font_sizes(root)
    return [Frame(element, styles) for element in root.iter(f"{{{NS['draw']}}}frame")]


def graphic_properties(path: Path) -> dict[str, str]:
    root = ElementTree.fromstring(zipfile.ZipFile(path).read("content.xml"))
    for style in root.iter(f"{{{NS['style']}}}style"):
        properties = style.find(f"{{{NS['style']}}}graphic-properties")
        if properties is not None:
            return {key.rsplit("}", 1)[-1]: value for key, value in properties.attrib.items()}
    raise AssertionError("no graphic properties emitted")


def word(text: str, left: float, top: float, size: float) -> Word:
    """A word as poppler would report it, for a source set at *size* points."""
    return Word(
        text=text,
        left=left,
        top=top,
        width=pdfmetrics.stringWidth(text, "Helvetica", size),
        height=size * TSV_BOX_HEIGHT_EM,
    )


def paragraph_page(
    line_count: int = 40,
    *,
    size: float = 11.0,
    leading_em: float = 1.25,
    left: float = 72.0,
    top: float = 60.0,
) -> dict[int, list[Word]]:
    words: list[Word] = []
    for index in range(line_count):
        y = top + index * size * leading_em
        x = left
        for token in ("Documento", "de", "teste", f"linha{index:02d}", "com", "palavras"):
            words.append(word(token, x, y, size))
            x += pdfmetrics.stringWidth(token + " ", "Helvetica", size)
    return {1: words}


def build(tmp_path: Path, pages_words, geometry=A4, name="fixed.odt") -> Path:
    target = tmp_path / name
    create_positioned_text_odf(pages_words, str(target), [geometry])
    return target


def collisions(frames, tolerance: float = OVERLAP_ALLOWANCE_CM) -> list[tuple[str, float]]:
    """Frame pairs that overlap on *both* axes.

    Both, because two frames sharing a y are side by side in different columns,
    which is ordinary. Only a pair that overlaps horizontally as well can put
    text on top of text.
    """
    found = []
    for index, upper in enumerate(frames):
        for lower in frames[index + 1 :]:
            if upper.page != lower.page:
                continue
            if upper.x >= lower.x + lower.width or lower.x >= upper.x + upper.width:
                continue
            if upper.bottom > lower.y + tolerance and lower.bottom > upper.y + tolerance:
                found.append((upper.text, round(upper.bottom - lower.y, 3)))
    return found


class TestFramesDoNotCollide:
    def test_consecutive_frames_do_not_overlap(self, tmp_path: Path):
        """The defect that prompted this file, in its own terms."""
        frames = frames_of(build(tmp_path, paragraph_page()))
        assert len(frames) == 40

        overlaps = collisions(frames)

        assert not overlaps, f"{len(overlaps)} frame(s) run into the next line: {overlaps[:5]}"

    def test_no_line_can_wrap_inside_its_frame(self, tmp_path: Path):
        """A wrapped line drops its tail onto the line below."""
        frames = frames_of(build(tmp_path, paragraph_page()))

        too_narrow = [
            frame.text
            for frame in frames
            if pdfmetrics.stringWidth(frame.text, "Helvetica", frame.font_size)
            > frame.width / CM_PER_POINT * 1.001
        ]

        assert not too_narrow, f"frames narrower than their own text: {too_narrow[:3]}"

    def test_frames_are_emitted_in_reading_order(self, tmp_path: Path):
        frames = frames_of(build(tmp_path, paragraph_page()))

        assert [frame.z for frame in frames] == sorted(frame.z for frame in frames)
        assert len({frame.z for frame in frames}) == len(frames)
        assert [frame.y for frame in frames] == sorted(frame.y for frame in frames)


class TestFrameStyle:
    @pytest.fixture
    def properties(self, tmp_path: Path) -> dict[str, str]:
        return graphic_properties(build(tmp_path, paragraph_page(2)))

    def test_wrapping_is_forbidden_by_the_style_not_only_by_the_width(self, properties):
        """No computed width can be right once the font is substituted."""
        assert properties["auto-grow-width"] == "true"
        assert properties["wrap-option"] == "no-wrap"

    def test_the_default_text_inset_is_removed(self, properties):
        """LibreOffice insets 0.25cm horizontally and 0.125cm vertically."""
        assert properties["padding"] == "0cm"
        for side in ("padding-top", "padding-bottom", "padding-left", "padding-right"):
            assert properties[side] == "0cm"

    def test_positions_are_resolved_against_the_page(self, properties):
        """Without these, svg:x/y mean whatever the default anchor frame means."""
        assert properties["horizontal-rel"] == "page"
        assert properties["vertical-rel"] == "page"
        assert properties["horizontal-pos"] == "from-left"
        assert properties["vertical-pos"] == "from-top"


class TestFontSize:
    @pytest.mark.parametrize("size", [6.0, 9.0, 11.0, 18.0, 36.0])
    def test_the_emitted_size_reproduces_the_source(self, tmp_path: Path, size: float):
        """Sized from the width, which is the quantity that breaks a layout."""
        frames = frames_of(build(tmp_path, paragraph_page(3, size=size, leading_em=2.0)))

        for frame in frames:
            assert frame.font_size == pytest.approx(size, rel=0.02)

    def test_unmeasurable_text_falls_back_to_the_height(self):
        """Helvetica metrics cannot measure CJK, and Liberation has no glyphs either."""
        line = TextLine(
            [Word(text="中文文本测试", left=72.0, top=100.0, width=60.0, height=9.3)], 100.0
        )

        _source, written = positioned_font_size(line)

        assert written == pytest.approx(9.3 / TSV_BOX_HEIGHT_EM, rel=0.05)

    def test_a_run_far_wider_than_its_text_is_clamped(self):
        """A merged or mis-detected run must not produce absurd type."""
        text = "curto"
        line = TextLine(
            [Word(text=text, left=0.0, top=0.0, width=4000.0, height=9.3)],
            0.0,
        )

        source, written = positioned_font_size(line)

        assert written <= MAX_FONT_SIZE
        assert source <= 1.31 * (9.3 / TSV_BOX_HEIGHT_EM)

    def test_sliver_artefacts_do_not_drive_the_size(self):
        """A traced rule arrives as a wide string in a 1.8pt box."""
        real = [word(token, 72.0 + index * 40, 100.0, 10.0) for index, token in enumerate("abc")]
        sliver = Word(text="x" * 60, left=72.0, top=100.0, width=1.8, height=9.3)

        clean = positioned_font_size(TextLine(list(real), 100.0))[1]
        polluted = positioned_font_size(TextLine([*real, sliver], 100.0))[1]

        assert polluted == clean


class TestVerticalPlacement:
    def test_the_baseline_lands_on_the_source_baseline(self, tmp_path: Path):
        size, top = 11.0, 100.0
        frames = frames_of(build(tmp_path, paragraph_page(1, size=size, top=top)))

        frame = frames[0]
        rendered_baseline_cm = frame.y + LIBERATION_FIRST_BASELINE_EM * size * CM_PER_POINT
        source_baseline_cm = (top + TSV_BASELINE_FRACTION * size * TSV_BOX_HEIGHT_EM) * CM_PER_POINT

        assert rendered_baseline_cm == pytest.approx(source_baseline_cm, abs=0.01)

    def test_the_font_shrinks_with_an_oversized_page(self, tmp_path: Path):
        """_page_size_cm shrinks a metre-long page, and the type must follow it.

        Otherwise full-size text sits inside frames reduced to a third, which
        wraps every line.
        """
        # 20pt source, halved: large enough that the shrunk size stays clear of
        # MIN_FONT_SIZE, where the ratio would be pinned by the clamp instead.
        page = paragraph_page(2, size=20.0, leading_em=2.0)
        a4 = frames_of(build(tmp_path, page, A4, "a4.odt"))
        # The same page in points, declared twice as tall: _page_size_cm caps
        # the long side at 29.7cm, so everything scales down by half.
        tall = (595.28, 841.89 * 2, 21.0 / 2, 29.7)
        big = frames_of(build(tmp_path, page, tall, "tall.odt"))

        assert big[0].font_size < a4[0].font_size
        assert big[0].font_size / big[0].height == pytest.approx(
            a4[0].font_size / a4[0].height, rel=0.02
        )
