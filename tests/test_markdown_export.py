"""Tests for Markdown export in tsv_odf_converter."""

import os
import tempfile
import threading
from unittest.mock import patch

import pytest

from bigocrpdf.utils.odf_builder import ExportCancelled
from bigocrpdf.utils.tsv_odf_converter import (
    _escape_md,
    _escape_md_cell,
    _format_table_markdown,
    _yaml_escape,
    convert_pdf_to_markdown,
    create_markdown,
)
from bigocrpdf.utils.tsv_parser import DocElement


def _mock_pdf_path(name: str) -> str:
    """Build a temp-dir path used as an opaque PDF identifier in mocked tests.

    The file is never actually read or written — ``parse_tsv_pages`` is
    mocked — but using the system temp dir instead of a hardcoded ``/tmp``
    keeps static analyzers (and Windows CI) happy.
    """
    return os.path.join(tempfile.gettempdir(), name)


class TestEscapeMd:
    def test_escapes_inline_special_chars(self):
        assert _escape_md("a*b_c") == r"a\*b\_c"
        assert _escape_md("[link]") == r"\[link\]"

    def test_passthrough_plain_text(self):
        assert _escape_md("simple words") == "simple words"

    def test_escapes_pipes_and_backticks(self):
        assert _escape_md("`code`") == r"\`code\`"
        assert _escape_md("a|b") == r"a\|b"

    def test_does_not_escape_inline_hyphens(self):
        # CPF/phone numbers should stay readable
        assert _escape_md("CPF 000.000.000-00") == "CPF 000.000.000-00"
        assert _escape_md("(00) 00000-0000") == "(00) 00000-0000"

    def test_escapes_block_markers_only_at_line_start(self):
        assert _escape_md("# heading") == r"\# heading"
        assert _escape_md("- bullet") == r"\- bullet"
        assert _escape_md("> quote") == r"\> quote"
        assert _escape_md("1. item") == r"\1. item"
        # Mid-line '#' stays untouched
        assert _escape_md("see #issue") == "see #issue"

    def test_asterisk_at_line_start_inline_escaped(self):
        # '*' is inline-escaped first; it isn't a block-level marker in the
        # line-start regex, so the inline rule is what kicks in.
        assert _escape_md("*emphasized* mid") == r"\*emphasized\* mid"


class TestEscapeMdCell:
    """Table cells need inline escapes but no line-start rules."""

    def test_escapes_pipe(self):
        assert _escape_md_cell("a|b") == r"a\|b"

    def test_escapes_emphasis_markers(self):
        assert _escape_md_cell("**bold**") == r"\*\*bold\*\*"
        assert _escape_md_cell("[link](x)") == r"\[link\](x)"

    def test_does_not_apply_line_start_rules(self):
        # A cell starting with '-' would be wrongly turned into a list item
        # outside a table, but inside the pipe-table cell it's fine.
        assert _escape_md_cell("- safe in cell") == "- safe in cell"


class TestFormatTableMarkdown:
    def test_basic_table(self):
        rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        out = _format_table_markdown(rows)
        assert out[0] == "| Name | Age |"
        assert out[1] == "|---|---|"
        assert out[2] == "| Alice | 30 |"
        assert out[3] == "| Bob | 25 |"

    def test_empty_rows_returns_empty(self):
        assert _format_table_markdown([]) == []

    def test_escapes_pipe_inside_cell(self):
        rows = [["h"], ["a|b"]]
        out = _format_table_markdown(rows)
        assert "a\\|b" in out[2]

    def test_uneven_rows_padded(self):
        rows = [["a", "b", "c"], ["x"]]
        out = _format_table_markdown(rows)
        # 3 columns expected in every row
        assert out[0].count("|") == 4
        assert out[2].count("|") == 4

    def test_escapes_markdown_specials_inside_cells(self):
        rows = [["h1"], ["**bold** and *italic*"]]
        out = _format_table_markdown(rows)
        assert r"\*\*bold\*\* and \*italic\*" in out[2]


class TestYamlEscape:
    def test_quotes_and_backslashes(self):
        assert _yaml_escape('say "hi"') == r"say \"hi\""
        assert _yaml_escape(r"a\b") == r"a\\b"

    def test_strips_control_chars_including_newlines(self):
        # Newlines and other control chars would break a single-line scalar.
        assert _yaml_escape("a\nb\tc\x00d") == "abcd"
        # Regular ASCII space stays.
        assert _yaml_escape("a b") == "a b"


class TestCreateMarkdown:
    def test_headings_emit_hash_prefixes(self):
        pages = [
            [
                DocElement("heading1", "Title"),
                DocElement("heading2", "Section"),
                DocElement("heading3", "Sub"),
            ]
        ]
        md = create_markdown(pages)
        assert "# Title" in md
        assert "## Section" in md
        assert "### Sub" in md

    def test_paragraph_text_emitted(self):
        pages = [[DocElement("paragraph", "Hello world.")]]
        md = create_markdown(pages)
        assert "Hello world." in md

    def test_kv_bolds_key(self):
        pages = [[DocElement("kv", "Author: Jane Doe")]]
        md = create_markdown(pages)
        assert "**Author:**" in md
        assert "Jane Doe" in md

    def test_table_renders_as_pipe_table(self):
        pages = [[DocElement("table", rows=[["A", "B"], ["1", "2"]])]]
        md = create_markdown(pages)
        assert "| A | B |" in md
        assert "|---|---|" in md
        assert "| 1 | 2 |" in md

    def test_pages_separated_by_thematic_break(self):
        pages = [
            [DocElement("paragraph", "page one")],
            [DocElement("paragraph", "page two")],
        ]
        md = create_markdown(pages)
        assert "page one" in md
        assert "page two" in md
        assert "\n---\n" in md

    def test_empty_pages_safe(self):
        assert create_markdown([]) == "\n"

    def test_paragraph_special_chars_escaped(self):
        pages = [[DocElement("paragraph", "use _underscores_ and *stars*")]]
        md = create_markdown(pages)
        assert r"\_underscores\_" in md
        assert r"\*stars\*" in md


class TestConvertPdfToMarkdown:
    def test_returns_empty_when_no_text(self):
        with patch(
            "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
            return_value={},
        ):
            assert convert_pdf_to_markdown("/nonexistent.pdf") == ""

    def test_front_matter_when_no_text(self):
        with patch(
            "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
            return_value={},
        ):
            out = convert_pdf_to_markdown(_mock_pdf_path("my_doc.pdf"), include_front_matter=True)
        assert out.startswith("---\n")
        assert 'title: "my_doc"' in out
        assert "pages: 0" in out

    def test_front_matter_with_content(self):
        # Mock the parser + processor so we exercise the front-matter wrapping.
        fake_words = {1: ["w1"]}
        with (
            patch(
                "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
                return_value=fake_words,
            ),
            patch(
                "bigocrpdf.utils.tsv_odf_converter.process_page",
                return_value=[DocElement("paragraph", "body text")],
            ),
        ):
            out = convert_pdf_to_markdown(_mock_pdf_path("sample.pdf"), include_front_matter=True)
        assert out.startswith("---\n")
        assert 'title: "sample"' in out
        assert "pages: 1" in out
        assert "body text" in out

    def test_cli_export_md_writes_file(self):
        # End-to-end: write Markdown to disk via the public function.
        fake_words = {1: ["w1"]}
        with (
            patch(
                "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
                return_value=fake_words,
            ),
            patch(
                "bigocrpdf.utils.tsv_odf_converter.process_page",
                return_value=[DocElement("heading1", "Hi")],
            ),
        ):
            text = convert_pdf_to_markdown(_mock_pdf_path("x.pdf"))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(path, encoding="utf-8") as f:
                assert "# Hi" in f.read()

    def test_cancel_event_set_before_call_raises(self):
        event = threading.Event()
        event.set()
        fake_words = {1: ["w1"], 2: ["w2"]}
        with patch(
            "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
            return_value=fake_words,
        ):
            with pytest.raises(ExportCancelled):
                convert_pdf_to_markdown(_mock_pdf_path("x.pdf"), cancel_event=event)

    def test_cancel_event_unset_runs_normally(self):
        event = threading.Event()  # not set
        fake_words = {1: ["w1"]}
        with (
            patch(
                "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
                return_value=fake_words,
            ),
            patch(
                "bigocrpdf.utils.tsv_odf_converter.process_page",
                return_value=[DocElement("paragraph", "ok")],
            ),
        ):
            text = convert_pdf_to_markdown(_mock_pdf_path("x.pdf"), cancel_event=event)
        assert "ok" in text

    def test_front_matter_date_is_utc_iso(self):
        # Date should always be ISO yyyy-mm-dd, never None/empty, regardless of TZ.
        import re

        with patch(
            "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
            return_value={},
        ):
            out = convert_pdf_to_markdown(_mock_pdf_path("x.pdf"), include_front_matter=True)
        m = re.search(r"^date: (\d{4}-\d{2}-\d{2})$", out, re.MULTILINE)
        assert m is not None, out


class TestUniquePath:
    def test_no_conflict_returns_input(self):
        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "doc.md")
            assert ConclusionExportMixin._unique_path(target) == target

    def test_auto_suffix_on_conflict(self):
        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "doc.md")
            open(target, "w").close()
            p1 = ConclusionExportMixin._unique_path(target)
            assert p1.endswith("doc (1).md")
            open(p1, "w").close()
            p2 = ConclusionExportMixin._unique_path(target)
            assert p2.endswith("doc (2).md")


class TestIsUserDismissed:
    def test_dismissed_message_detected(self):
        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        assert ConclusionExportMixin._is_user_dismissed(RuntimeError("Dismissed by user"))

    def test_other_errors_pass_through(self):
        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        assert not ConclusionExportMixin._is_user_dismissed(RuntimeError("Disk full"))
        assert not ConclusionExportMixin._is_user_dismissed(FileNotFoundError("nope"))
