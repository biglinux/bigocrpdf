"""Logging should be quiet by default and explicit in diagnostic modes."""

import argparse
import logging
from pathlib import Path

import pytest

from bigocrpdf import config
from bigocrpdf.utils.logger import setup_logger


@pytest.fixture(autouse=True)
def _restore_logging_state():
    root_level = logging.getLogger().level
    app_level = logging.getLogger(config.LOGGER_NAME).level
    configured_level = config.LOG_LEVEL
    yield
    logging.getLogger().setLevel(root_level)
    logging.getLogger(config.LOGGER_NAME).setLevel(app_level)
    config.LOG_LEVEL = configured_level


def _arguments(*, debug: bool = False, verbose: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        version=False,
        debug=debug,
        verbose=verbose,
        image_mode=False,
        edit=False,
        files=[],
    )


def test_gui_logging_is_warning_only_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "parse_command_line", lambda: _arguments())

    config.setup_environment()

    assert config.LOG_LEVEL == logging.WARNING
    assert logging.getLogger().level == logging.WARNING


def test_verbose_enables_diagnostic_logging(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "parse_command_line", lambda: _arguments(verbose=True))

    config.setup_environment()

    assert config.LOG_LEVEL == logging.DEBUG
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logger_updates_existing_root_handler(monkeypatch) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("OLD:%(message)s"))
    monkeypatch.setattr(root_logger, "handlers", [handler])

    configured = setup_logger(
        log_level=logging.INFO,
        log_format="NEW:%(message)s",
        logger_name=config.LOGGER_NAME,
    )
    record = logging.LogRecord("test", logging.INFO, "", 0, "message", (), None)

    assert root_logger.handlers == [handler]
    assert root_logger.level == logging.INFO
    assert handler.level == logging.INFO
    assert handler.format(record) == "NEW:message"
    assert configured.level == logging.INFO
