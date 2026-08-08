"""Tests for CLI argument parsing functions."""

import pytest

from bigocrpdf.cli import _parse_page_list, _parse_ranges, build_parser


class TestParsePageList:
    """Tests for _parse_page_list."""

    def test_single_page(self):
        assert _parse_page_list("3") == [3]

    def test_range(self):
        assert _parse_page_list("1-5") == [1, 2, 3, 4, 5]

    def test_comma_separated(self):
        assert _parse_page_list("1,3,7") == [1, 3, 7]

    def test_mixed(self):
        assert _parse_page_list("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]

    def test_deduplicates(self):
        assert _parse_page_list("1-3,2-4") == [1, 2, 3, 4]

    def test_strips_whitespace(self):
        assert _parse_page_list(" 1 , 3 - 5 ") == [1, 3, 4, 5]

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _parse_page_list("0,1,-1,2")

    @pytest.mark.parametrize("value", ["0", "3-1", "0-2"])
    def test_non_positive_or_reversed_range_raises(self, value):
        with pytest.raises(ValueError, match="Invalid page specification"):
            _parse_page_list(value)

    def test_empty_parts_skipped(self):
        assert _parse_page_list(",1,,2,") == [1, 2]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid page specification"):
            _parse_page_list("abc")

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="Invalid page specification"):
            _parse_page_list("1-abc")

    def test_empty_string(self):
        assert _parse_page_list("") == []


class TestParseRanges:
    """Tests for _parse_ranges."""

    def test_single_range(self):
        assert _parse_ranges("1-5") == [(1, 5)]

    def test_multiple_ranges(self):
        assert _parse_ranges("1-5,6-10,11-15") == [(1, 5), (6, 10), (11, 15)]

    def test_single_page_becomes_range(self):
        assert _parse_ranges("3") == [(3, 3)]

    def test_mixed(self):
        assert _parse_ranges("1-5,7") == [(1, 5), (7, 7)]

    def test_empty_parts_skipped(self):
        assert _parse_ranges(",1-5,,") == [(1, 5)]

    def test_strips_whitespace(self):
        assert _parse_ranges(" 1 - 5 , 6 - 10 ") == [(1, 5), (6, 10)]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid range specification"):
            _parse_ranges("abc")

    @pytest.mark.parametrize("value", ["0", "0-2", "3-1"])
    def test_non_positive_or_reversed_range_raises(self, value):
        with pytest.raises(ValueError, match="Invalid range specification"):
            _parse_ranges(value)

    def test_empty_string(self):
        assert _parse_ranges("") == []


class TestBuildParser:
    """Tests for build_parser."""

    def test_returns_parser(self):
        p = build_parser()
        assert p is not None

    def test_version_flag(self, capsys):
        parser = build_parser()

        with pytest.raises(SystemExit) as exit_info:
            parser.parse_args(["--version"])

        assert exit_info.value.code == 0
        assert capsys.readouterr().out == "bigocrpdf-cli 3.0.0\n"

    def test_ocr_subcommand(self):
        p = build_parser()
        args = p.parse_args(["ocr", "input.pdf", "-o", "output.pdf"])
        assert args.command == "ocr"
        assert str(args.input) == "input.pdf"
        assert str(args.output) == "output.pdf"

    def test_split_by_pages(self):
        p = build_parser()
        args = p.parse_args(["split", "in.pdf", "-o", "outdir", "--pages", "5"])
        assert args.command == "split"
        assert args.pages == 5

    def test_split_by_size(self):
        p = build_parser()
        args = p.parse_args(["split", "in.pdf", "-o", "outdir", "--size", "10.5"])
        assert args.command == "split"
        assert args.size == 10.5

    def test_merge_subcommand(self):
        p = build_parser()
        args = p.parse_args(["merge", "a.pdf", "b.pdf", "-o", "merged.pdf"])
        assert args.command == "merge"
        assert len(args.inputs) == 2

    def test_compress_subcommand(self):
        p = build_parser()
        args = p.parse_args(["compress", "in.pdf", "-o", "out.pdf", "--quality", "40"])
        assert args.command == "compress"
        assert args.quality == 40

    def test_rotate_subcommand(self):
        p = build_parser()
        args = p.parse_args(["rotate", "in.pdf", "-o", "out.pdf", "--angle", "90"])
        assert args.command == "rotate"
        assert args.angle == 90

    def test_delete_subcommand(self):
        p = build_parser()
        args = p.parse_args(["delete", "in.pdf", "-o", "out.pdf", "--pages", "3,5"])
        assert args.command == "delete"
        assert args.pages == "3,5"

    def test_extract_subcommand(self):
        p = build_parser()
        args = p.parse_args(["extract", "in.pdf", "-o", "out.pdf", "--pages", "1-3"])
        assert args.command == "extract"

    def test_info_subcommand(self):
        p = build_parser()
        args = p.parse_args(["info", "input.pdf"])
        assert args.command == "info"

    def test_export_odf_subcommand(self):
        p = build_parser()
        args = p.parse_args(["export-odf", "input.pdf"])
        assert args.command == "export-odf"
        assert args.preserve_text_layout is False

        positioned_args = p.parse_args(["export-odf", "input.pdf", "--preserve-text-layout"])
        assert positioned_args.preserve_text_layout is True

    def test_export_txt_subcommand(self):
        p = build_parser()
        args = p.parse_args(["export-txt", "input.pdf"])
        assert args.command == "export-txt"

    def test_verbose_flag(self):
        p = build_parser()
        args = p.parse_args(["-v", "info", "input.pdf"])
        assert args.verbose is True

    def test_ocr_has_no_legacy_language_selection(self):
        p = build_parser()
        args = p.parse_args(["ocr", "in.pdf", "-o", "out.pdf"])
        assert not hasattr(args, "language")

        with pytest.raises(SystemExit):
            p.parse_args(["ocr", "in.pdf", "-o", "out.pdf", "--language", "arabic"])
        with pytest.raises(SystemExit):
            p.parse_args(["ocr", "in.pdf", "-o", "out.pdf", "--ocr-version", "PPOCRV5"])

    def test_ocr_pdf_mode_and_batch_options(self):
        p = build_parser()
        args = p.parse_args(
            [
                "ocr",
                "in.pdf",
                "-o",
                "out.pdf",
                "--pdf-mode",
                "auto_verified",
                "--engine-type",
                "onnxruntime",
                "--rec-batch-num",
                "8",
                "--use-textline-cls",
            ]
        )
        assert args.pdf_mode == "auto_verified"
        assert args.engine_type == "onnxruntime"
        assert args.rec_batch_num == 8
        assert args.use_textline_cls is True

    def test_ocr_no_flags(self):
        p = build_parser()
        args = p.parse_args(["ocr", "in.pdf", "-o", "out.pdf", "--no-dewarp", "--no-deskew"])
        assert args.no_dewarp is True
        assert args.no_deskew is True
