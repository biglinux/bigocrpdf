"""Tests for PDF text escaping and assembly utilities."""

from bigocrpdf.services.rapidocr_service.pdf_assembly import escape_pdf_text


class TestEscapePdfText:
    """Tests for escape_pdf_text."""

    def test_plain_ascii(self):
        assert escape_pdf_text("Hello World") == "Hello World"

    def test_escapes_backslash(self):
        assert escape_pdf_text("a\\b") == "a\\\\b"

    def test_escapes_parentheses(self):
        assert escape_pdf_text("(test)") == "\\(test\\)"

    def test_em_dash_replaced(self):
        assert escape_pdf_text("word\u2014word") == "word-word"

    def test_en_dash_replaced(self):
        assert escape_pdf_text("1\u20135") == "1-5"

    def test_smart_quotes_replaced(self):
        text = "\u201cHello\u201d \u2018world\u2019"
        assert escape_pdf_text(text) == "\"Hello\" 'world'"

    def test_ellipsis_replaced(self):
        assert escape_pdf_text("wait\u2026") == "wait..."

    def test_bullet_replaced(self):
        assert escape_pdf_text("\u2022 item") == "* item"

    def test_fi_ligature(self):
        assert escape_pdf_text("\ufb01nd") == "find"

    def test_fl_ligature(self):
        assert escape_pdf_text("\ufb02ow") == "flow"

    def test_zero_width_chars_removed(self):
        assert escape_pdf_text("a\u200bb\u200cc\ufeffd") == "abcd"

    def test_non_latin1_replaced_with_question(self):
        # Chinese character can't be encoded to latin-1
        assert escape_pdf_text("\u4e16") == "?"

    def test_combined_escaping(self):
        text = "(test\\) \u2014 \u201cquote\u201d"
        result = escape_pdf_text(text)
        assert "\\\\" in result
        assert "\\(" in result
        assert "\\)" in result
        assert "-" in result
        assert '"' in result

    def test_empty_string(self):
        assert escape_pdf_text("") == ""

    def test_only_latin1_passes(self):
        text = "\xe9\xe0\xfc"  # é à ü
        assert escape_pdf_text(text) == text
