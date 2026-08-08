"""Nautilus context-menu integration."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

EXTENSION = (
    Path(__file__).resolve().parents[1]
    / "usr/share/nautilus-python/extensions/bigocrpdf-actions.py"
)


class _GObjectBase:
    pass


class _MenuProvider:
    pass


class _MenuItem:
    def __init__(self, **properties):
        self.properties = properties
        self.callback = None
        self.callback_args = ()

    def connect(self, _signal, callback, *args):
        self.callback = callback
        self.callback_args = args

    def activate(self):
        assert self.callback is not None
        self.callback(self, *self.callback_args)


class _Location:
    def __init__(self, path):
        self.path = path

    def get_path(self):
        return self.path


class _File:
    def __init__(self, mime_type, path):
        self.mime_type = mime_type
        self.location = _Location(path)

    def get_mime_type(self):
        return self.mime_type

    def get_location(self):
        return self.location


def _load_extension(monkeypatch):
    gi = ModuleType("gi")
    repository = ModuleType("gi.repository")
    repository.GObject = SimpleNamespace(GObject=_GObjectBase)
    repository.Nautilus = SimpleNamespace(MenuProvider=_MenuProvider, MenuItem=_MenuItem)
    gi.repository = repository
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    spec = importlib.util.spec_from_file_location("bigocrpdf_nautilus_test", EXTENSION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ = lambda text: text
    return module


def _labels(items):
    return [item.properties["label"] for item in items]


def test_nautilus_menu_matches_selected_workflow(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    extension = module.BigOcrPdfExtension()

    pdf = [_File("application/pdf", "/tmp/document.pdf")]
    assert _labels(extension.get_file_items(pdf)) == ["Text recognition (OCR)", "Edit pages"]

    png = [_File("image/png", "/tmp/page.png")]
    assert _labels(extension.get_file_items(png)) == ["Text recognition (OCR)", "Create PDF"]

    gifs = [_File("image/gif", "/tmp/one.gif"), _File("image/gif", "/tmp/two.gif")]
    assert extension.get_file_items(gifs) == []

    unsupported = [_File("image/heif", "/tmp/page.heif")]
    assert extension.get_file_items(unsupported) == []

    remote = [_File("image/png", None)]
    assert extension.get_file_items(remote) == []


def test_nautilus_action_launches_selected_local_files(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    commands = []
    monkeypatch.setattr(module.subprocess, "Popen", commands.append)

    files = [_File("image/png", "/tmp/one.png"), _File("image/png", "/tmp/two.png")]
    items = module.BigOcrPdfExtension().get_file_items(files)
    assert _labels(items) == ["Create PDF"]

    items[0].activate()
    assert commands == [["bigocrpdf", "--edit", "/tmp/one.png", "/tmp/two.png"]]
