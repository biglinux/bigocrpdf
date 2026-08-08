from bigocrpdf.services.rapidocr_service.pdf_validation import (
    classify_text_layer,
    validate_searchable_pdf_text,
)


def test_empty_extracted_text_fails_validation():
    result = validate_searchable_pdf_text(" \n\t")
    assert result.ok is False
    assert result.extracted_chars == 0
    assert result.reason == "empty extracted text"


def test_suspicious_glyph_loss_fails_validation():
    result = validate_searchable_pdf_text("Valid document ??? □□□")
    assert result.ok is False
    assert result.reason == "suspicious glyph loss"


def test_normal_unicode_text_passes_validation():
    result = validate_searchable_pdf_text("ação João nº 中文 العربية Ελληνικά हिन्दी ภาษาไทย")
    assert result.ok is True
    assert result.suspicious_ratio == 0


def test_empty_native_text_layer_is_absent():
    result = classify_text_layer(" \n\t")
    assert result.status == "absent"
    assert result.reason == "empty"


def test_short_native_text_layer_is_rejected():
    result = classify_text_layer("abc")
    assert result.status == "rejected"
    assert result.reason == "too_few_chars"


def test_lossy_native_text_layer_is_rejected():
    result = classify_text_layer("Document ??? □□□ with glyph loss")
    assert result.status == "rejected"
    assert result.reason == "suspicious_glyph_loss"


def test_clean_native_text_layer_is_trusted():
    result = classify_text_layer("Résumé № 123. John paid $1,234.56 in São Paulo.")
    assert result.status == "trusted"
    assert result.chars > 10
