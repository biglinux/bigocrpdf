"""Durability regressions for small UI preference files."""

from pathlib import Path
from unittest.mock import MagicMock

from bigocrpdf.ui.pdf_editor.editor_help_controller import EditorHelpController
from bigocrpdf.ui.welcome_dialog_controller import WelcomeDialogController


def _symlinked_preference(tmp_path: Path, name: str) -> tuple[Path, Path]:
    protected = tmp_path / f"{name}.protected"
    protected.write_text("KEEP", encoding="utf-8")
    preference = tmp_path / name
    preference.symlink_to(protected)
    return preference, protected


def test_welcome_preference_replaces_symlink_without_touching_target(tmp_path: Path) -> None:
    preference, protected = _symlinked_preference(tmp_path, "welcome")
    controller = WelcomeDialogController(MagicMock(), preference)

    controller.set_show(False)

    assert protected.read_text(encoding="utf-8") == "KEEP"
    assert not preference.is_symlink()
    assert preference.read_text(encoding="utf-8") == "false"


def test_editor_help_preference_replaces_symlink_without_touching_target(tmp_path: Path) -> None:
    preference, protected = _symlinked_preference(tmp_path, "editor-help")
    controller = EditorHelpController(MagicMock(), preference)

    controller.set_show(False)

    assert protected.read_text(encoding="utf-8") == "KEEP"
    assert not preference.is_symlink()
    assert preference.read_text(encoding="utf-8") == "false"


def test_welcome_preference_does_not_read_symlink(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.write_text("false", encoding="utf-8")
    preference = tmp_path / "welcome"
    preference.symlink_to(protected)

    assert WelcomeDialogController(MagicMock(), preference).should_show() is True


def test_editor_help_preference_does_not_read_symlink(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.write_text("false", encoding="utf-8")
    preference = tmp_path / "editor-help"
    preference.symlink_to(protected)

    assert EditorHelpController(MagicMock(), preference).should_show() is True


def test_welcome_preference_only_explicit_false_disables_dialog(tmp_path: Path) -> None:
    preference = tmp_path / "welcome"
    controller = WelcomeDialogController(MagicMock(), preference)

    preference.write_text("false", encoding="utf-8")
    assert controller.should_show() is False

    preference.write_text("invalid", encoding="utf-8")
    assert controller.should_show() is True


def test_invalid_welcome_preference_encoding_defaults_to_visible(tmp_path: Path) -> None:
    preference = tmp_path / "welcome"
    preference.write_bytes(b"\xff")

    assert WelcomeDialogController(MagicMock(), preference).should_show() is True


def test_invalid_editor_help_preference_defaults_to_visible(tmp_path: Path) -> None:
    preference = tmp_path / "editor-help"
    preference.write_text("invalid", encoding="utf-8")

    assert EditorHelpController(MagicMock(), preference).should_show() is True
