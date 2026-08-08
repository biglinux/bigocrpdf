"""Locale directory resolution, including relocatable (AppImage) layouts."""

import importlib
import struct

import pytest

from bigocrpdf.utils import i18n


def _write_catalog(root, language: str) -> None:
    """Create a minimal but valid .mo file so glob() and gettext both accept it."""
    target = root / language / "LC_MESSAGES" / f"{i18n._DOMAIN}.mo"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Empty catalog: magic, revision, 0 strings, offsets past the header.
    target.write_bytes(struct.pack("<7I", 0x950412DE, 0, 0, 28, 28, 0, 28))


def test_appdir_wins_over_the_host_locale_dir(tmp_path, monkeypatch):
    """An AppImage must use its own catalogs, never the host's."""
    appdir = tmp_path / "AppDir"
    _write_catalog(appdir / "usr/share/locale", "pt_BR")

    monkeypatch.setenv("APPDIR", str(appdir))
    monkeypatch.setenv("APPIMAGE", str(tmp_path / "x.AppImage"))
    monkeypatch.delenv("BIGOCRPDF_LOCALE_DIR", raising=False)

    assert i18n._find_locale_dir() == appdir / "usr/share/locale"


def test_explicit_override_wins_over_appdir(tmp_path, monkeypatch):
    appdir = tmp_path / "AppDir"
    _write_catalog(appdir / "usr/share/locale", "pt_BR")
    override = tmp_path / "custom"
    _write_catalog(override, "pt_BR")

    monkeypatch.setenv("APPDIR", str(appdir))
    monkeypatch.setenv("BIGOCRPDF_LOCALE_DIR", str(override))

    assert i18n._find_locale_dir() == override


def test_textdomaindir_from_apprun_is_honoured(tmp_path, monkeypatch):
    bundled = tmp_path / "AppDir/usr/share/locale"
    _write_catalog(bundled, "de")

    monkeypatch.delenv("APPDIR", raising=False)
    monkeypatch.delenv("BIGOCRPDF_LOCALE_DIR", raising=False)
    monkeypatch.setenv("TEXTDOMAINDIR", str(bundled))

    assert i18n._find_locale_dir() == bundled


def test_directory_without_catalogs_is_skipped(tmp_path, monkeypatch):
    """A pointer to an empty directory must not shadow a real catalog."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("BIGOCRPDF_LOCALE_DIR", str(empty))
    monkeypatch.delenv("APPDIR", raising=False)
    monkeypatch.delenv("TEXTDOMAINDIR", raising=False)

    assert i18n._find_locale_dir() != empty


def test_setup_i18n_is_safe_when_the_override_is_bogus(monkeypatch):
    """A bad override must fall back to the system prefix, not raise."""
    monkeypatch.setenv("BIGOCRPDF_LOCALE_DIR", "/nonexistent")
    monkeypatch.delenv("APPDIR", raising=False)
    monkeypatch.delenv("TEXTDOMAINDIR", raising=False)

    i18n.setup_i18n()
    assert isinstance(i18n._("Cancel"), str)


def test_module_imports_without_a_display():
    """The CLI imports this module before any GTK setup."""
    importlib.reload(i18n)
    assert callable(i18n._)


@pytest.mark.parametrize("variable", ["APPDIR", "TEXTDOMAINDIR", "BIGOCRPDF_LOCALE_DIR"])
def test_unset_variables_do_not_produce_candidates(variable, monkeypatch):
    monkeypatch.delenv(variable, raising=False)
    assert all(str(c) != "" for c in i18n._locale_dir_candidates())
