"""Tests for Markdown export in tsv_odf_converter."""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import call, patch

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

    def test_decimal_at_line_start_not_escaped(self):
        # "1.5 million" must not be mangled to "\1.5 million"; the
        # ordered-list rule applies only when the dot is followed by
        # whitespace (or end of string).
        assert _escape_md("1.5 million users") == "1.5 million users"
        assert _escape_md("2025.06 release") == "2025.06 release"
        # End-of-string also counts as a list-marker boundary so "1." alone
        # stays escaped.
        assert _escape_md("1.") == r"\1."


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

    @pytest.mark.parametrize(
        "text",
        ["FAX NO. (614) 466-5087", "PAGE 1 OF 2", "JUN 30 SEP 22", "DIVISION: FULL PARTIAL"],
    )
    def test_form_labels_and_page_metadata_are_not_headings(self, text):
        markdown = create_markdown([[DocElement("heading2", text)]])
        assert not markdown.startswith("## ")
        assert text.replace("*", r"\*") in markdown

    def test_section_label_ending_in_colon_remains_heading(self):
        markdown = create_markdown([[DocElement("heading2", "SPECIAL INSTRUCTIONS:")]])
        assert markdown.startswith("## SPECIAL INSTRUCTIONS:")

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

    def test_kv_elements_separated_by_blank_line(self):
        # Adjacent kv elements must be separated so CommonMark renders them
        # as distinct paragraphs instead of one soft-wrapped block.
        pages = [
            [
                DocElement("kv", "Author: Jane"),
                DocElement("kv", "Date: 2025"),
            ]
        ]
        md = create_markdown(pages)
        # There must be a blank line between the two bolded keys.
        assert "**Author:** Jane\n\n**Date:** 2025" in md

    def test_paragraph_preserves_raw_lines(self):
        # When the OCR layer captured per-line breaks, the Markdown output
        # keeps them via CommonMark hard breaks (two trailing spaces).
        pages = [
            [
                DocElement(
                    "paragraph",
                    "Rua X 123 Bairro Y CEP 00000-000",
                    raw_lines=["Rua X 123", "Bairro Y", "CEP 00000-000"],
                )
            ]
        ]
        md = create_markdown(pages)
        # Hard break = two trailing spaces before the newline.
        assert "Rua X 123  \nBairro Y  \nCEP 00000-000" in md

    def test_box_drawing_diagram_is_code_not_heading(self):
        pages = [
            [
                DocElement(
                    "heading2",
                    "┌─ QOI_OP_INDEX ─┐ │ Byte[0] │ └───────────────┘",
                    raw_lines=["┌─ QOI_OP_INDEX ─┐", "│ Byte[0] │", "└───────────────┘"],
                )
            ]
        ]

        markdown = create_markdown(pages)

        assert "##" not in markdown
        assert markdown.startswith("```\n┌─ QOI_OP_INDEX ─┐")
        assert markdown.endswith("└───────────────┘\n```\n")

    def test_adjacent_diagram_fragments_share_one_code_fence(self):
        pages = [
            [
                DocElement("heading2", "┌── A ──┐", raw_lines=["┌── A ──┐", "└───────┘"]),
                DocElement("paragraph", "┌── B ──┐", raw_lines=["┌── B ──┐", "└───────┘"]),
            ]
        ]

        markdown = create_markdown(pages)

        assert markdown.count("```") == 2
        assert "└───────┘\n┌── B ──┐" in markdown


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

    def test_cancel_event_set_during_last_page_raises(self):
        event = threading.Event()
        fake_words = {1: ["w1"]}

        def process_last_page(_words, _page_num):
            event.set()
            return [DocElement("paragraph", "partial")]

        with (
            patch(
                "bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages",
                return_value=fake_words,
            ),
            patch(
                "bigocrpdf.utils.tsv_odf_converter.process_page",
                side_effect=process_last_page,
            ),
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
            assert ConclusionExportMixin._reserve_unique_path(target) == target

    def test_auto_suffix_on_conflict(self):
        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "doc.md")
            open(target, "w").close()
            p1 = ConclusionExportMixin._reserve_unique_path(target)
            assert p1.endswith("doc (1).md")
            open(p1, "w").close()
            p2 = ConclusionExportMixin._reserve_unique_path(target)
            assert p2.endswith("doc (2).md")


def test_bulk_markdown_temp_symlink_cannot_overwrite_target(tmp_path: Path) -> None:
    from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    output = tmp_path / "export.md"
    predictable_temp = tmp_path / "export.md.tmp"
    predictable_temp.symlink_to(victim)

    with patch(
        "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_markdown",
        return_value="# Exported\n",
    ):
        ConclusionExportMixin._bulk_convert_one(
            object(),
            "input.pdf",
            str(output),
            "md",
            {},
            threading.Event(),
        )

    assert output.read_text(encoding="utf-8") == "# Exported\n"
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert predictable_temp.is_symlink()


class TestIsUserDismissed:
    def test_gtk_dismissed_error_detected(self):
        import gi

        gi.require_version("Gtk", "4.0")
        from gi.repository import GLib, Gtk

        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        error = GLib.Error.new_literal(
            Gtk.DialogError.quark(),
            "Dismissed by user",
            Gtk.DialogError.DISMISSED,
        )
        assert ConclusionExportMixin._is_user_dismissed(error)

    def test_other_gtk_dialog_errors_pass_through(self):
        import gi

        gi.require_version("Gtk", "4.0")
        from gi.repository import GLib, Gtk

        from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

        error = GLib.Error.new_literal(
            Gtk.DialogError.quark(),
            "Dialog failed",
            Gtk.DialogError.FAILED,
        )
        assert not ConclusionExportMixin._is_user_dismissed(error)


def test_odf_export_options_are_forwarded_without_shared_controller_state() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

    owner = SimpleNamespace(_show_odf_file_dialog=MagicMock())
    dialog = MagicMock()

    ConclusionExportMixin._on_export_clicked(owner, True, False, "first.pdf", dialog)
    ConclusionExportMixin._on_export_clicked(owner, False, True, "second.pdf", dialog)

    assert owner._show_odf_file_dialog.call_args_list == [
        call("first.pdf", True, False),
        call("second.pdf", False, True),
    ]
    assert not hasattr(owner, "_odf_export_mode")
    assert not hasattr(owner, "_odf_open_after")


def test_single_markdown_export_uses_the_same_converter_as_bulk_export() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin

    owner = SimpleNamespace(_run_single_export=MagicMock(), _bulk_convert_one=MagicMock())
    ConclusionExportMixin._export_markdown_file(owner, "out.md", "in.pdf", True, False)

    convert = owner._run_single_export.call_args.args[-1]
    cancel_event = MagicMock()
    convert(cancel_event)
    owner._bulk_convert_one.assert_called_once_with(
        "in.pdf", "out.md", "md", {"include_front_matter": True}, cancel_event
    )


class TestSaveMdSettings:
    """MD export toggles persist through the same config flow as ODF."""

    def test_save_md_settings_writes_both_keys(self):
        from bigocrpdf.services.settings import OcrSettings

        # Minimal config double — record .set() calls and replay .get().
        store: dict = {}

        class _Cfg:
            def get(self, key, default=None):
                return store.get(key, default)

            def set(self, key, value, save_immediately=False):
                store[key] = value

            def save(self):
                pass

        s = OcrSettings.__new__(OcrSettings)
        s._config = _Cfg()  # type: ignore[attr-defined]
        s.md_include_front_matter = True
        s.md_open_after_export = True
        s._save_md_settings()

        assert store.get("md_export.include_front_matter") is True
        assert store.get("md_export.open_after_export") is True

    def test_load_md_settings_reads_both_keys(self):
        from bigocrpdf.services.settings import OcrSettings

        class _Cfg:
            def __init__(self):
                self.store = {
                    "md_export.include_front_matter": True,
                    "md_export.open_after_export": True,
                }

            def get(self, key, default=None):
                return self.store.get(key, default)

        s = OcrSettings.__new__(OcrSettings)
        s._config = _Cfg()  # type: ignore[attr-defined]
        s._load_md_settings()

        assert s.md_include_front_matter is True
        assert s.md_open_after_export is True

    def test_save_md_settings_defaults_when_unset(self):
        # Settings object freshly constructed may not have the attributes —
        # the saver must default to False instead of raising AttributeError.
        from bigocrpdf.services.settings import OcrSettings

        store: dict = {}

        class _Cfg:
            def get(self, key, default=None):
                return store.get(key, default)

            def set(self, key, value, save_immediately=False):
                store[key] = value

            def save(self):
                pass

        s = OcrSettings.__new__(OcrSettings)
        s._config = _Cfg()  # type: ignore[attr-defined]
        s._save_md_settings()  # no attributes set yet

        assert store["md_export.include_front_matter"] is False
        assert store["md_export.open_after_export"] is False
