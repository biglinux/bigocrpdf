"""Tests for gettext locale discovery."""

from pathlib import Path

from bigocrpdf.utils import i18n


def test_find_locale_dir_supports_staged_source_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    module_path = project / "src/bigocrpdf/utils/i18n.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    locale_dir = project / "usr/share/locale/pt_BR/LC_MESSAGES"
    locale_dir.mkdir(parents=True)
    (locale_dir / "bigocrpdf.mo").touch()

    real_glob = Path.glob

    def local_glob(path: Path, pattern: str):
        if path.is_relative_to(tmp_path):
            return real_glob(path, pattern)
        return iter(())

    monkeypatch.setattr(i18n, "__file__", str(module_path))
    monkeypatch.setattr(Path, "glob", local_glob)

    assert i18n._find_locale_dir() == project / "usr/share/locale"
