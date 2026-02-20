"""Tests for config_manager module."""

import json
import os
import tempfile

from bigocrpdf.utils.config_manager import ConfigManager


class TestConfigManager:
    def _make_manager(self, tmp_dir, initial=None):
        path = os.path.join(tmp_dir, "config.json")
        if initial:
            with open(path, "w") as f:
                json.dump(initial, f)
        return ConfigManager(config_path=path)

    def test_get_default_value(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            assert cm.get("nonexistent.key", "fallback") == "fallback"

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.set("ocr.language", "deu", save_immediately=False)
            assert cm.get("ocr.language") == "deu"

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cm = ConfigManager(config_path=path)
            cm.set("test.key", "value123")
            # Reload from disk
            cm2 = ConfigManager(config_path=path)
            assert cm2.get("test.key") == "value123"

    def test_nested_key_path(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.set("a.b.c", 42, save_immediately=False)
            assert cm.get("a.b.c") == 42

    def test_default_config_has_ocr_language(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            lang = cm.get("ocr.language")
            assert isinstance(lang, str)

    def test_load_existing_config(self):
        with tempfile.TemporaryDirectory() as d:
            initial = {"ocr": {"language": "jpn"}}
            cm = self._make_manager(d, initial=initial)
            assert cm.get("ocr.language") == "jpn"

    def test_save_returns_true(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            assert cm.save() is True

    def test_set_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.set("ocr.language", "fra", save_immediately=False)
            cm.set("ocr.language", "deu", save_immediately=False)
            assert cm.get("ocr.language") == "deu"
