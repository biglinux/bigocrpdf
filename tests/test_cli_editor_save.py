"""Tests for the standalone editor command."""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

from bigocrpdf.cli_editor_commands import _cmd_edit


def test_edit_command_checks_runtime_before_importing_editor(tmp_path: Path) -> None:
    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"gi.repository", "bigocrpdf.ui.pdf_editor.editor_window"}:
            raise AssertionError("editor imported before runtime validation")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch("bigocrpdf._check_gtk_dependencies", return_value=False),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        result = _cmd_edit(
            argparse.Namespace(input=tmp_path / "document.pdf"),
            logging.getLogger("test"),
        )

    assert result == 1
