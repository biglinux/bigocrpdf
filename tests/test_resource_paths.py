"""Model and font lookup, including relocatable (AppImage) layouts."""

import pytest

from bigocrpdf.services.rapidocr_service import resource_paths


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from a known state; individual tests opt back in."""
    for var in ("BIGOCRPDF_RAPIDOCR_DIR", "APPDIR"):
        monkeypatch.delenv(var, raising=False)


def _populate(root, models=("PP-OCRv6_det_small.onnx",), fonts=("latin.ttf",)):
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "fonts").mkdir(parents=True, exist_ok=True)
    for name in models:
        (root / "models" / name).write_bytes(b"")
    for name in fonts:
        (root / "fonts" / name).write_bytes(b"")
    return root


def test_appdir_wins_over_the_system_directory(tmp_path, monkeypatch):
    """An AppImage must use its own models, never the host's."""
    appdir = tmp_path / "AppDir"
    _populate(appdir / "usr/share/rapidocr")
    monkeypatch.setenv("APPDIR", str(appdir))

    assert resource_paths.find_model_dir() == appdir / "usr/share/rapidocr/models"
    assert resource_paths.find_font_dir() == appdir / "usr/share/rapidocr/fonts"


def test_explicit_override_wins_over_appdir(tmp_path, monkeypatch):
    appdir = _populate(tmp_path / "AppDir/usr/share/rapidocr")
    override = _populate(tmp_path / "custom")
    monkeypatch.setenv("APPDIR", str(tmp_path / "AppDir"))
    monkeypatch.setenv("BIGOCRPDF_RAPIDOCR_DIR", str(override))

    assert resource_paths.find_model_dir() == override / "models"
    assert appdir != override


def test_directory_without_models_is_skipped(tmp_path, monkeypatch):
    """A pointer at an empty tree must not shadow a real one."""
    empty = tmp_path / "empty"
    (empty / "models").mkdir(parents=True)
    monkeypatch.setenv("BIGOCRPDF_RAPIDOCR_DIR", str(empty))

    assert resource_paths.find_model_dir() != empty / "models"


def test_models_and_fonts_resolve_independently(tmp_path, monkeypatch):
    """Only the directory holding the right file type counts."""
    root = tmp_path / "AppDir/usr/share/rapidocr"
    (root / "models").mkdir(parents=True)
    (root / "models" / "PP-OCRv6_det_small.onnx").write_bytes(b"")
    (root / "fonts").mkdir(parents=True)  # present but empty
    monkeypatch.setenv("APPDIR", str(tmp_path / "AppDir"))

    assert resource_paths.find_model_dir() == root / "models"
    # No .ttf here, so the font lookup must keep going and not stop on it.
    assert resource_paths.find_font_dir() != root / "fonts"


def test_falls_back_to_the_system_root_when_nothing_is_found(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGOCRPDF_RAPIDOCR_DIR", str(tmp_path / "nonexistent"))

    models = resource_paths.find_model_dir()
    fonts = resource_paths.find_font_dir()

    # Either a real directory found by walking up, or the documented default.
    assert models.name == "models"
    assert fonts.name == "fonts"


def test_config_and_discovery_share_the_resolution(tmp_path, monkeypatch):
    """The two entry points must not disagree about where models live."""
    root = _populate(
        tmp_path / "AppDir/usr/share/rapidocr",
        models=("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx"),
    )
    monkeypatch.setenv("APPDIR", str(tmp_path / "AppDir"))

    from bigocrpdf.services.rapidocr_service.config import OCRConfig
    from bigocrpdf.services.rapidocr_service.discovery import ModelDiscovery

    config = OCRConfig()
    assert config.model_base_path == root / "models"
    assert config.get_det_model_path() == root / "models" / "PP-OCRv6_det_small.onnx"
    assert config.get_rec_model_path() == root / "models" / "PP-OCRv6_rec_small.onnx"

    assert ModelDiscovery().model_path == root / "models"
    assert ModelDiscovery().get_available_languages() != []


def test_a_stale_model_set_does_not_win_over_a_complete_one(tmp_path, monkeypatch):
    """Selection is by content: an older generation must not shadow the wheel's.

    A BigLinux host carries PP-OCRv5 under /usr/share/rapidocr while the
    installed wheel ships PP-OCRv6. Choosing the first directory holding any
    .onnx picks the stale set and the application reports models missing.
    """
    stale = tmp_path / "system/share/rapidocr"
    (stale / "models").mkdir(parents=True)
    (stale / "models" / "latin_PP-OCRv5_rec_mobile_infer.onnx").write_bytes(b"")

    current = tmp_path / "wheel/rapidocr/models"
    current.mkdir(parents=True)
    for name in resource_paths.DEFAULT_REQUIRED_MODELS:
        (current / name).write_bytes(b"")

    monkeypatch.setenv("BIGOCRPDF_RAPIDOCR_DIR", str(stale))
    monkeypatch.setattr(resource_paths, "rapidocr_bundled_models", lambda: current)

    assert resource_paths.find_model_dir() == current


def test_a_complete_share_tree_still_wins_over_the_wheel(tmp_path, monkeypatch):
    """When both hold the required pair, the explicit location comes first."""
    appdir = tmp_path / "AppDir"
    models = appdir / "usr/share/rapidocr/models"
    models.mkdir(parents=True)
    for name in resource_paths.DEFAULT_REQUIRED_MODELS:
        (models / name).write_bytes(b"")

    wheel = tmp_path / "wheel/rapidocr/models"
    wheel.mkdir(parents=True)
    for name in resource_paths.DEFAULT_REQUIRED_MODELS:
        (wheel / name).write_bytes(b"")

    monkeypatch.setenv("APPDIR", str(appdir))
    monkeypatch.setattr(resource_paths, "rapidocr_bundled_models", lambda: wheel)

    assert resource_paths.find_model_dir() == models


def test_bundled_lookup_does_not_import_rapidocr(monkeypatch):
    """find_spec must be enough; importing drags in the whole OCR stack."""
    import sys

    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    resource_paths.rapidocr_bundled_models()
    assert "rapidocr" not in sys.modules


def test_missing_rapidocr_is_not_fatal(monkeypatch):
    """The resolver runs before dependencies are validated."""
    monkeypatch.setattr(
        resource_paths.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ImportError("no rapidocr")),
    )
    assert resource_paths.rapidocr_bundled_models() is None
    assert resource_paths.find_model_dir().name == "models"


def test_no_hardcoded_rapidocr_paths_outside_the_resolver():
    """The absolute path may only appear in resource_paths."""
    from pathlib import Path

    import bigocrpdf

    src_root = Path(bigocrpdf.__file__).parent
    offenders = []
    for path in src_root.rglob("*.py"):
        if path.name == "resource_paths.py":
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if "/usr/share/rapidocr" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(src_root)}:{number}")

    assert not offenders, (
        "Hardcoded model paths break relocatable builds; use "
        f"resource_paths.find_model_dir()/find_font_dir(): {offenders}"
    )
