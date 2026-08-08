from bigocrpdf.services.rapidocr_service.config import OcrDocument, OcrPage, OCRResult


def test_ocr_document_full_text_prefers_native_text():
    document = OcrDocument()
    document.append_page(
        OcrPage(
            page_index=1,
            width_px=100,
            height_px=200,
            dpi=300,
            native_text="Native text",
            text_results=[OCRResult("OCR text")],
        )
    )

    assert document.full_text() == "Native text"


def test_ocr_document_full_text_falls_back_to_ocr_results():
    document = OcrDocument()
    document.append_page(
        OcrPage(
            page_index=1,
            width_px=100,
            height_px=200,
            dpi=300,
            text_results=[OCRResult("Line 1"), OCRResult("Line 2")],
        )
    )

    assert document.full_text() == "Line 1\nLine 2"
