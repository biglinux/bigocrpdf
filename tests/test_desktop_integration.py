"""Installed desktop integration metadata."""

import ast
import tomllib
from configparser import ConfigParser
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS = ROOT / "usr/share/applications"
SERVICE_MENUS = ROOT / "usr/share/kio/servicemenus"
METAINFO = ROOT / "usr/share/metainfo/br.com.biglinux.bigocrpdf.metainfo.xml"
NEMO_ACTIONS = ROOT / "usr/share/nemo/actions"
NAUTILUS_EXTENSION = ROOT / "usr/share/nautilus-python/extensions/bigocrpdf-actions.py"

IMAGE_OCR_MIME_TYPES = {
    "image/apng",
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/x-portable-anymap",
    "image/x-portable-bitmap",
    "image/x-portable-graymap",
    "image/x-portable-pixmap",
}
EDITOR_IMAGE_MIME_TYPES = {
    "image/avif",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def _read_desktop(path: Path) -> ConfigParser:
    desktop = ConfigParser(interpolation=None)
    cast(Any, desktop).optionxform = str
    desktop.read(path, encoding="utf-8")
    return desktop


def _list(value: str) -> set[str]:
    return {item for item in value.split(";") if item}


def _frozenset_assignment(name: str) -> set[str]:
    tree = ast.parse(NAUTILUS_EXTENSION.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if isinstance(node.value, ast.Call) and node.value.args:
            return set(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"Assignment not found: {name}")


def test_desktop_mime_types_match_supported_workflows() -> None:
    image = _read_desktop(APPLICATIONS / "br.com.biglinux.bigocrimage.desktop")
    editor = _read_desktop(APPLICATIONS / "br.com.biglinux.bigocrpdf-editor.desktop")
    image_service = _read_desktop(SERVICE_MENUS / "ocrimage.desktop")
    create_service = _read_desktop(SERVICE_MENUS / "createpdf.desktop")

    assert _list(image["Desktop Entry"]["MimeType"]) == IMAGE_OCR_MIME_TYPES
    assert _list(image_service["Desktop Entry"]["MimeType"]) == IMAGE_OCR_MIME_TYPES
    assert _list(editor["Desktop Entry"]["MimeType"]) == EDITOR_IMAGE_MIME_TYPES | {
        "application/pdf"
    }
    assert _list(create_service["Desktop Entry"]["MimeType"]) == EDITOR_IMAGE_MIME_TYPES


def test_file_manager_mime_types_match_supported_workflows() -> None:
    nemo_ocr = _read_desktop(NEMO_ACTIONS / "bigocrpdf-ocrimage.nemo_action")
    nemo_create = _read_desktop(NEMO_ACTIONS / "bigocrpdf-createpdf.nemo_action")

    assert _list(nemo_ocr["Nemo Action"]["Mimetypes"]) == IMAGE_OCR_MIME_TYPES
    assert _list(nemo_create["Nemo Action"]["Mimetypes"]) == EDITOR_IMAGE_MIME_TYPES
    assert _frozenset_assignment("_OCR_IMAGE_MIMES") == IMAGE_OCR_MIME_TYPES
    assert _frozenset_assignment("_PDF_IMAGE_MIMES") == EDITOR_IMAGE_MIME_TYPES


def test_service_menus_are_final_local_file_entries() -> None:
    assert not list(SERVICE_MENUS.glob("*.desktop.in"))

    for path in SERVICE_MENUS.glob("*.desktop"):
        desktop = _read_desktop(path)
        entry = desktop["Desktop Entry"]
        assert entry["Type"] == "Service"
        assert entry["X-KDE-Protocols"] == "file"
        assert "ServiceTypes" not in entry
        for action in _list(entry["Actions"]):
            assert f"Desktop Action {action}" in desktop

    image = _read_desktop(SERVICE_MENUS / "ocrimage.desktop")
    assert image["Desktop Entry"]["X-KDE-RequiredNumberOfUrls"] == "1"
    assert image["Desktop Action OCR"]["Exec"] == "bigocrimage %f"


def test_editor_name_covers_project_locales() -> None:
    editor = _read_desktop(APPLICATIONS / "br.com.biglinux.bigocrpdf-editor.desktop")
    supported = {path.stem for path in (ROOT / "locale").glob("*.po")} - {"en"}
    translated = {
        key.removeprefix("Name[").removesuffix("]")
        for key in editor["Desktop Entry"]
        if key.startswith("Name[")
    }
    assert translated == supported


def test_appstream_release_matches_project_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]

    component = ElementTree.parse(METAINFO).getroot()
    assert component.findtext("launchable") == "br.com.biglinux.bigocrpdf.desktop"
    release = component.find("./releases/release")
    assert release is not None
    assert release.attrib == {"version": version, "date": "2026-08-02"}
