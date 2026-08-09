from bigocrpdf.services.rapidocr_service.config import OcrLine, OcrWord
from bigocrpdf.services.rapidocr_service.native_text_verification import (
    NativeTextSpan,
    parse_pdftotext_bbox_layout,
    verify_ocr_lines_with_native_spans,
)


def test_parse_pdftotext_bbox_layout_scales_page_points_to_image_pixels() -> None:
    html = """\
<html xmlns="http://www.w3.org/1999/xhtml">
<body><doc><page width="200.000000" height="100.000000">
  <flow><block xMin="20" yMin="21" xMax="105" yMax="32">
    <line xMin="20" yMin="21" xMax="105" yMax="32">
      <word xMin="20" yMin="21" xMax="47" yMax="32">Hello</word>
      <word xMin="50" yMin="21" xMax="82" yMax="32">native</word>
      <word xMin="86" yMin="21" xMax="105" yMax="32">text</word>
    </line>
  </block></flow>
</page></doc></body></html>
"""

    spans = parse_pdftotext_bbox_layout(html, image_size_px=(400, 200))

    assert spans == [
        NativeTextSpan(text="Hello", bbox=[40.0, 42.0, 94.0, 64.0]),
        NativeTextSpan(text="native", bbox=[100.0, 42.0, 164.0, 64.0]),
        NativeTextSpan(text="text", bbox=[172.0, 42.0, 210.0, 64.0]),
    ]


def test_parse_pdftotext_bbox_layout_maps_source_image_rect() -> None:
    html = """\
<html xmlns="http://www.w3.org/1999/xhtml">
<body><doc><page width="200" height="100">
  <flow><block><line xMin="30" yMin="50" xMax="70" yMax="60">
    <word xMin="30" yMin="50" xMax="70" yMax="60">Inside</word>
  </line></block></flow>
</page></doc></body></html>
"""

    spans = parse_pdftotext_bbox_layout(
        html,
        image_size_px=(200, 100),
        source_rect_pts=(20.0, 10.0, 100.0, 50.0),
    )

    assert spans == [NativeTextSpan(text="Inside", bbox=[20.0, 20.0, 100.0, 40.0])]


def test_parse_pdftotext_bbox_layout_clips_to_source_image_rect() -> None:
    html = """\
<html xmlns="http://www.w3.org/1999/xhtml">
<body><doc><page width="200" height="100">
  <flow><block><line xMin="10" yMin="30" xMax="130" yMax="100">
    <word xMin="10" yMin="30" xMax="130" yMax="100">Clipped</word>
  </line></block></flow>
</page></doc></body></html>
"""

    spans = parse_pdftotext_bbox_layout(
        html,
        image_size_px=(200, 100),
        source_rect_pts=(20.0, 10.0, 100.0, 50.0),
    )

    assert spans == [NativeTextSpan(text="Clipped", bbox=[0.0, 0.0, 200.0, 100.0])]


def test_verify_ocr_lines_replaces_with_overlapping_usable_native_text() -> None:
    ocr_lines = [
        OcrLine(
            text="Hella native text",
            bbox=[40, 42, 210, 64],
            words=[OcrWord("Hella", [40, 42, 95, 64], 0.7)],
            reading_order=3,
        )
    ]
    native_spans = [NativeTextSpan(text="Hello native text", bbox=[40, 42, 210, 64])]

    verified = verify_ocr_lines_with_native_spans(ocr_lines, native_spans)

    assert verified.accepted_lines == 1
    assert verified.rejected_lines == 0
    assert verified.lines[0].text == "Hello native text"
    assert verified.lines[0].source == "pdf"
    assert verified.lines[0].reading_order == 3


def test_verify_ocr_lines_rejects_lossy_or_unrelated_native_text() -> None:
    ocr_lines = [
        OcrLine(text="Invoice number 12345", bbox=[40, 42, 210, 64], reading_order=0),
        OcrLine(text="Total amount", bbox=[40, 90, 210, 112], reading_order=1),
    ]
    native_spans = [
        NativeTextSpan(text="��� ???", bbox=[40, 42, 210, 64]),
        NativeTextSpan(text="Different footer", bbox=[40, 160, 210, 182]),
    ]

    verified = verify_ocr_lines_with_native_spans(ocr_lines, native_spans)

    assert verified.accepted_lines == 0
    assert verified.rejected_lines == 2
    assert [line.source for line in verified.lines] == ["ocr", "ocr"]
    assert [line.text for line in verified.lines] == ["Invoice number 12345", "Total amount"]
