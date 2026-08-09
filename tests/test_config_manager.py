"""Tests for config_manager module."""

import builtins
import json
import os
import tempfile
from pathlib import Path

import pytest

from bigocrpdf.utils import config_manager as config_manager_module
from bigocrpdf.utils.config_manager import DEFAULT_CONFIG, ConfigManager


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

    def test_relative_filename_uses_current_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)

        manager = ConfigManager("settings.json")
        manager.set("ocr.language", "latin")

        assert ConfigManager("settings.json").get("ocr.language") == "latin"

    def test_stale_instances_merge_independent_updates(self, tmp_path: Path):
        config_path = tmp_path / "settings.json"
        pdf_app = ConfigManager(str(config_path))
        image_app = ConfigManager(str(config_path))

        pdf_app.set("rapidocr.language", "english")
        image_app.set("image_window.width", 1024)

        reloaded = ConfigManager(str(config_path))
        assert reloaded.get("rapidocr.language") == "english"
        assert reloaded.get("image_window.width") == 1024

    def test_reload_observes_another_process_update(self, tmp_path: Path):
        config_path = tmp_path / "settings.json"
        long_lived = ConfigManager(str(config_path))
        external = ConfigManager(str(config_path))
        external.set("rapidocr.language", "english")

        assert long_lived.get("rapidocr.language") is None
        assert long_lived.reload() is True
        assert long_lived.get("rapidocr.language") == "english"

    def test_reload_read_failure_preserves_memory_and_reports_failure(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        config_path = tmp_path / "settings.json"
        manager = ConfigManager(str(config_path))
        manager.set("rapidocr.language", "english")
        real_open = builtins.open

        def fail_config_read(path, *args, **kwargs):
            if Path(path) == config_path and kwargs.get("encoding") == "utf-8":
                raise OSError("temporary read failure")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fail_config_read)

        assert manager.reload() is False
        assert manager.get("rapidocr.language") == "english"
        assert (
            json.loads(config_path.read_text(encoding="utf-8"))["rapidocr"]["language"] == "english"
        )

    def test_nested_key_path(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.set("a.b.c", 42, save_immediately=False)
            assert cm.get("a.b.c") == 42

    def test_default_config_ocr_language_absent(self):
        # Language is intentionally omitted from defaults so Settings can
        # auto-detect from system locale on first run; see config_manager
        # DEFAULT_CONFIG and services.settings._load_language_settings.
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            assert cm.get("ocr.language") is None

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

    def test_save_replaces_symlink_without_overwriting_its_target(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        original = '{"version": 1, "owner": "victim"}\n'
        victim.write_text(original, encoding="utf-8")
        config_path = tmp_path / "settings.json"
        config_path.symlink_to(victim)

        manager = ConfigManager(config_path=str(config_path))
        manager.set("ocr.language", "latin")

        assert victim.read_text(encoding="utf-8") == original
        assert not config_path.is_symlink()
        assert ConfigManager(config_path=str(config_path)).get("ocr.language") == "latin"

    def test_load_does_not_import_configuration_through_symlink(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        victim.write_text(
            json.dumps({"version": 1, "ocr": {"quality": "attacker"}}),
            encoding="utf-8",
        )
        config_path = tmp_path / "settings.json"
        config_path.symlink_to(victim)

        manager = ConfigManager(config_path=str(config_path))

        assert manager.get("ocr.quality") == DEFAULT_CONFIG["ocr"]["quality"]
        assert json.loads(victim.read_text(encoding="utf-8"))["ocr"]["quality"] == "attacker"
        assert not config_path.is_symlink()

    @pytest.mark.parametrize(
        "raw_config",
        [
            "[]",
            "null",
            "42",
            '{"version": "1"}',
            '{"version": 1, "output": []}',
            '{"version": 1, "rapidocr": "fast"}',
            '{"version": 1, "window": {"width": "wide"}}',
        ],
    )
    def test_semantically_invalid_config_is_backed_up_and_replaced(
        self, tmp_path: Path, raw_config: str
    ):
        config_path = tmp_path / "settings.json"
        config_path.write_text(raw_config, encoding="utf-8")

        manager = ConfigManager(config_path=str(config_path))

        assert manager.get("version") == DEFAULT_CONFIG["version"]
        assert manager.get("window.width") == DEFAULT_CONFIG["window"]["width"]
        backups = list(tmp_path.glob("settings.json.corrupt*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == raw_config
        assert json.loads(config_path.read_text(encoding="utf-8"))["version"] == 1

    def test_malformed_config_backup_never_overwrites_existing_backup(self, tmp_path: Path):
        config_path = tmp_path / "settings.json"
        config_path.write_text('{"version":', encoding="utf-8")
        first_backup = tmp_path / "settings.json.corrupt"
        first_backup.write_text("older corrupt config", encoding="utf-8")

        ConfigManager(config_path=str(config_path))

        assert first_backup.read_text(encoding="utf-8") == "older corrupt config"
        assert (tmp_path / "settings.json.corrupt.1").read_text(encoding="utf-8") == '{"version":'
        assert json.loads(config_path.read_text(encoding="utf-8")) == DEFAULT_CONFIG

    def test_future_config_version_is_preserved_before_defaults_are_published(self, tmp_path: Path):
        config_path = tmp_path / "settings.json"
        future_config = {"version": DEFAULT_CONFIG["version"] + 1, "future": {"mode": True}}
        raw_config = json.dumps(future_config)
        config_path.write_text(raw_config, encoding="utf-8")

        manager = ConfigManager(config_path=str(config_path))

        assert manager.get("future.mode") is None
        assert json.loads(config_path.read_text(encoding="utf-8")) == DEFAULT_CONFIG
        assert (tmp_path / "settings.json.corrupt").read_text(encoding="utf-8") == raw_config

    def test_partial_current_config_is_deep_merged_and_persisted(self, tmp_path: Path):
        config_path = tmp_path / "settings.json"
        config_path.write_text(
            json.dumps({"version": 1, "ocr": {"quality": "custom"}, "date": {}}),
            encoding="utf-8",
        )

        manager = ConfigManager(config_path=str(config_path))

        assert manager.get("ocr.quality") == "custom"
        assert manager.get("ocr.alignment") == DEFAULT_CONFIG["ocr"]["alignment"]
        assert manager.get("date.format_order.year") == 1
        persisted = json.loads(config_path.read_text(encoding="utf-8"))
        assert persisted["window"] == DEFAULT_CONFIG["window"]

        manager.get("date.format_order")["year"] = 99
        other = ConfigManager(config_path=str(tmp_path / "other.json"))
        assert other.get("date.format_order.year") == 1

    def test_backup_failure_preserves_corrupt_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config_path = tmp_path / "settings.json"
        raw_config = '{"version":'
        config_path.write_text(raw_config, encoding="utf-8")

        def fail_backup(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated backup failure")

        monkeypatch.setattr(config_manager_module, "write_text_file_atomically", fail_backup)

        manager = ConfigManager(config_path=str(config_path))

        assert manager.get("window.width") == DEFAULT_CONFIG["window"]["width"]
        assert config_path.read_text(encoding="utf-8") == raw_config
        assert list(tmp_path.glob("settings.json.corrupt*")) == []
