"""Where the RapidOCR models and fonts live at runtime.

The distribution package installs them under ``/usr/share/rapidocr``, but a
relocatable build carries its own copy and must never fall through to the host,
which on a non-BigLinux system has nothing there at all.  Resolution mirrors
``utils.i18n``: explicit override first, then the AppImage root, then paths
derived from this file, then the system prefix.

A candidate only counts when it actually holds the expected content, so a
directory that exists but is empty does not shadow a real one.
"""

import importlib.util
import os
import sys
from pathlib import Path

_RELATIVE = "share/rapidocr"

# The distribution default, and the last resort.
SYSTEM_ROOT = Path("/usr/share/rapidocr")

# The pair the application actually loads, matching DEFAULT_MODEL_TYPE. Used to
# pick a directory by content rather than by mere presence of some .onnx file.
DEFAULT_REQUIRED_MODELS = ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx")


def rapidocr_bundled_models() -> Path | None:
    """The models shipped inside the installed ``rapidocr`` package.

    Since 3.9.0 the PyPI wheel carries its own PP-OCRv6 models, which is how a
    pip-installed build gets them without any extra packaging step.  These are
    guaranteed to match the library that will load them, unlike a system copy
    that may lag behind.  Distribution packages usually strip them out, in
    which case this finds nothing and the system directory is used instead.

    ``find_spec`` locates the package without importing it, so this stays cheap
    and cannot fail on a broken optional dependency.
    """
    try:
        spec = importlib.util.find_spec("rapidocr")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    models = Path(spec.origin).parent / "models"
    return models if models.is_dir() else None


def _root_candidates() -> list[Path]:
    candidates: list[Path] = []

    override = os.environ.get("BIGOCRPDF_RAPIDOCR_DIR")
    if override:
        candidates.append(Path(override))

    appdir = os.environ.get("APPDIR")
    if appdir:
        candidates.append(Path(appdir) / "usr" / _RELATIVE)

    # Walking up from this module covers every layout the package can be
    # installed in -- source checkout, wheel under site-packages, AppDir --
    # without needing to know the prefix in advance.
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / _RELATIVE)
        candidates.append(parent / "usr" / _RELATIVE)

    candidates.append(Path(sys.prefix) / _RELATIVE)
    candidates.append(SYSTEM_ROOT)
    return candidates


def _first_holding(subdir: str, suffix: str) -> Path | None:
    for root in _root_candidates():
        directory = root / subdir
        try:
            if any(directory.glob(f"*{suffix}")):
                return directory
        except OSError:
            continue
    return None


def _holds_all(directory: Path, names: tuple[str, ...]) -> bool:
    try:
        return all((directory / name).is_file() for name in names)
    except OSError:
        return False


def find_model_dir(required: tuple[str, ...] = DEFAULT_REQUIRED_MODELS) -> Path:
    """Directory holding the ``.onnx`` models, or the system default.

    Selection is by content, not by mere existence.  A machine can hold an
    older generation under ``/usr/share/rapidocr`` while the installed wheel
    carries the current one, and picking the first directory with any ``.onnx``
    in it would choose the stale set and leave the application reporting that
    models are missing while usable ones sit unused.

    So a directory that holds every ``required`` file wins outright, wherever
    it is; only then does the usual precedence apply.
    """
    if required:
        for root in _root_candidates():
            if _holds_all(root / "models", required):
                return root / "models"

        bundled = rapidocr_bundled_models()
        if bundled is not None and _holds_all(bundled, required):
            return bundled

    found = _first_holding("models", ".onnx")
    if found is not None:
        return found

    bundled = rapidocr_bundled_models()
    if bundled is not None:
        return bundled

    return SYSTEM_ROOT / "models"


def find_font_dir() -> Path:
    """Directory holding the ``.ttf`` fonts, or the system default."""
    return _first_holding("fonts", ".ttf") or (SYSTEM_ROOT / "fonts")
