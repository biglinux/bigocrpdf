"""Per-invocation command-line handling in the primary application instance."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bigocrpdf.application import BigOcrPdfApp


def _command_line(arguments: list[str], *, edit: bool) -> SimpleNamespace:
    options = SimpleNamespace(contains=lambda name: edit and name == "edit")
    return SimpleNamespace(
        get_options_dict=MagicMock(return_value=options),
        get_arguments=MagicMock(return_value=arguments),
        create_file_for_arg=MagicMock(side_effect=lambda arg: f"file:{arg}"),
    )


def _application() -> SimpleNamespace:
    return SimpleNamespace(
        _open_files=MagicMock(),
        _open_standalone_editor=MagicMock(),
        activate=MagicMock(),
    )


def test_edit_flag_reaches_an_already_running_instance() -> None:
    application = _application()
    command_line = _command_line(["bigocrpdf", "/tmp/photo.png"], edit=True)

    result = BigOcrPdfApp.on_command_line(application, application, command_line)

    assert result == 0
    application._open_files.assert_called_once_with(
        application, ["file:/tmp/photo.png"], edit_mode=True
    )


def test_edit_flag_does_not_leak_into_the_next_invocation() -> None:
    application = _application()

    BigOcrPdfApp.on_command_line(
        application, application, _command_line(["bigocrpdf", "/tmp/photo.png"], edit=True)
    )
    BigOcrPdfApp.on_command_line(
        application, application, _command_line(["bigocrpdf", "/tmp/scan.pdf"], edit=False)
    )

    assert application._open_files.call_args_list[-1].kwargs == {"edit_mode": False}


def test_edit_flag_without_files_opens_the_standalone_editor() -> None:
    application = _application()

    with patch("bigocrpdf.application.load_css"):
        BigOcrPdfApp.on_command_line(
            application, application, _command_line(["bigocrpdf"], edit=True)
        )

    application._open_standalone_editor.assert_called_once_with(application)
    application._open_files.assert_not_called()
    application.activate.assert_not_called()


def test_plain_invocation_activates_the_main_window() -> None:
    application = _application()

    BigOcrPdfApp.on_command_line(application, application, _command_line(["bigocrpdf"], edit=False))

    application.activate.assert_called_once_with()
    application._open_files.assert_not_called()


def test_open_signal_uses_normal_mode() -> None:
    application = _application()
    files = ["file:/tmp/photo.png"]

    BigOcrPdfApp.on_open(application, application, files, 1, "")

    application._open_files.assert_called_once_with(application, files, edit_mode=False)


def test_edit_mode_sends_images_to_the_editor() -> None:
    application = SimpleNamespace(
        _categorize_files=BigOcrPdfApp._categorize_files,
        _open_edit_mode=MagicMock(),
        _open_normal_mode=MagicMock(),
    )
    files = [SimpleNamespace(get_path=MagicMock(return_value="/tmp/photo.png"))]

    with patch("bigocrpdf.application.load_css"):
        BigOcrPdfApp._open_files(application, application, files, edit_mode=True)

    application._open_edit_mode.assert_called_once_with(application, [], ["/tmp/photo.png"])
    application._open_normal_mode.assert_not_called()
