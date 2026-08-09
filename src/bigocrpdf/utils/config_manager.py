"""
BigOcrPdf - Configuration Manager

This module provides centralized JSON-based configuration management.
It handles loading, saving, and migrating settings from legacy file format.
"""

import copy
import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bigocrpdf.utils.durable_writes import write_text_file_atomically
from bigocrpdf.utils.logger import logger

# Configuration directory - defined locally to avoid circular imports
CONFIG_DIR: Final[str] = os.path.expanduser("~/.config/bigocrpdf")

# Configuration file path
CONFIG_FILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "settings.json")

# Legacy file paths (for migration)
LEGACY_PATHS: Final[dict[str, str]] = {
    "lang": os.path.join(CONFIG_DIR, "lang"),
    "quality": os.path.join(CONFIG_DIR, "quality"),
    "align": os.path.join(CONFIG_DIR, "align"),
    "same_folder": os.path.join(CONFIG_DIR, "same-folder"),
    "savefile": os.path.join(CONFIG_DIR, "savefile"),
    "selected_file": os.path.join(CONFIG_DIR, "selected-file"),
}

# Default configuration values
DEFAULT_CONFIG: Final[dict[str, Any]] = {
    "version": 1,
    "window": {
        "width": 820,
        "height": 600,
    },
    "ocr": {
        # "language" intentionally omitted to allow automatic detection
        "quality": "normal",
        "alignment": "alignrotate",
    },
    "output": {
        "suffix": "ocr",
        "overwrite_existing": False,
        "save_in_same_folder": True,
        "destination_folder": "",
    },
    "date": {
        "include_date": False,
        "include_year": False,
        "include_month": False,
        "include_day": False,
        "include_time": False,
        "format_order": {
            "year": 1,
            "month": 2,
            "day": 3,
        },
    },
    "text_extraction": {
        "save_txt": False,
        "separate_folder": False,
        "txt_folder": "",
    },
}

# Every application-owned top-level key is a section addressed through dotted paths.
CONFIG_OBJECT_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        *(DEFAULT_CONFIG.keys() - {"version"}),
        "editor",
        "editor_window",
        "image_export",
        "image_window",
        "md_export",
        "multi_pdf_dialog",
        "odf_export",
        "rapidocr",
        "ui",
    }
)
_MISSING: Final[object] = object()


@dataclass(frozen=True)
class _MigrationRule:
    """Configuration for migrating a single legacy setting.

    Attributes:
        legacy_key: Key in LEGACY_PATHS dict identifying the legacy file.
        section: Target section in the config (e.g., "ocr", "output").
        target_key: Target key within the section.
        transformer: Function to transform raw string to proper type.
    """

    legacy_key: str
    section: str
    target_key: str
    transformer: Callable[[str], Any]


def _transform_string(value: str) -> str | None:
    """Transform string value, returning None if empty."""
    return value if value else None


def _transform_boolean(value: str) -> bool:
    """Transform string to boolean (case-insensitive 'true' check)."""
    return value.lower() == "true"


def _transform_directory(value: str) -> str | None:
    """Transform directory path, validating it exists."""
    return value if value and os.path.isdir(value) else None


class ConfigManager:
    """Manages application configuration in JSON format.

    This class provides a centralized way to load, save, and access
    configuration settings. It supports automatic migration from
    legacy individual text files to the new JSON format.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_path: Optional path to the configuration file.
                        Defaults to CONFIG_FILE_PATH.
        """
        self.config_path = config_path or CONFIG_FILE_PATH
        self._config: dict[str, Any] = {}
        self._base_config: dict[str, Any] = {}
        self._force_replace_digest: str | None = None

        # Ensure config directory exists
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)

        # Load or create configuration
        self._load_config()

    def _load_config(self) -> bool:
        """Load configuration, reporting an unreadable existing file."""
        try:
            raw_config = self._read_raw_config()
        except OSError as e:
            logger.error(f"Error reading config: {e}")
            if not self._config:
                defaults = self._get_default_config()
                self._config = defaults
                self._base_config = copy.deepcopy(defaults)
            if os.path.islink(self.config_path):
                return self.reset_to_defaults()
            return False

        if raw_config is not None:
            try:
                loaded_config = json.loads(raw_config)
            except json.JSONDecodeError as e:
                self._recover_invalid_config(raw_config, f"invalid JSON: {e}")
                return True

            validation_error = self._config_validation_error(loaded_config)
            if validation_error is not None:
                self._recover_invalid_config(raw_config, validation_error)
                return True

            self._base_config = copy.deepcopy(loaded_config)
            self._config = loaded_config
            logger.info("Configuration loaded from JSON")
            if self._upgrade_config():
                return self.save()
            return True
        # Check for legacy files and migrate
        self._base_config = {}
        self._config = self._get_default_config()
        self._migrate_from_legacy()
        return self.save()

    def _config_validation_error(self, config: object) -> str | None:
        """Return a reason when a loaded configuration cannot be safely migrated."""
        if not isinstance(config, dict):
            return "configuration root must be an object"

        version = config.get("version", 0)
        if type(version) is not int:
            return "configuration version must be an integer"
        if version > DEFAULT_CONFIG["version"]:
            return f"unsupported future configuration version: {version}"

        for key in CONFIG_OBJECT_SECTIONS & config.keys():
            if not isinstance(config[key], dict):
                return f"configuration section {key!r} must be an object"

        return self._default_shape_error(config, DEFAULT_CONFIG)

    def _default_shape_error(
        self,
        config: dict[str, Any],
        defaults: dict[str, Any],
        prefix: str = "",
    ) -> str | None:
        """Validate types for fields whose schema is defined by DEFAULT_CONFIG."""
        for key, default_value in defaults.items():
            if key not in config:
                continue
            value = config[key]
            key_path = f"{prefix}.{key}" if prefix else key
            if isinstance(default_value, dict):
                if not isinstance(value, dict):
                    return f"configuration field {key_path!r} must be an object"
                error = self._default_shape_error(value, default_value, key_path)
                if error is not None:
                    return error
            elif type(value) is not type(default_value):
                return f"configuration field {key_path!r} has an invalid type"
        return None

    def _recover_invalid_config(self, raw_config: str, reason: str) -> None:
        """Preserve an invalid document, then atomically publish fresh defaults."""
        logger.error(f"Invalid configuration: {reason}")
        self._base_config = {}
        self._config = self._get_default_config()
        backup_path = self._next_corrupt_backup_path()
        try:
            published_backup = write_text_file_atomically(
                backup_path,
                lambda backup_file: backup_file.write(raw_config),
                overwrite=False,
            )
        except (OSError, ValueError) as e:
            logger.error(f"Could not preserve invalid configuration: {e}")
            return

        logger.warning(f"Invalid configuration preserved at: {published_backup}")
        self._force_replace_digest = hashlib.sha256(raw_config.encode("utf-8")).hexdigest()
        self.save()

    def _next_corrupt_backup_path(self) -> Path:
        """Choose a deterministic backup name without overwriting prior recovery data."""
        base_path = Path(f"{self.config_path}.corrupt")
        candidate = base_path
        counter = 1
        while os.path.lexists(candidate):
            candidate = Path(f"{base_path}.{counter}")
            counter += 1
        return candidate

    def _get_default_config(self) -> dict[str, Any]:
        """Get a copy of the default configuration.

        Returns:
            Deep copy of default configuration dictionary.
        """
        return copy.deepcopy(DEFAULT_CONFIG)

    def _upgrade_config(self) -> bool:
        """Merge missing defaults and return whether the document changed."""
        current_version = self._config.get("version", 0)
        changed = self._merge_defaults(self._config, DEFAULT_CONFIG)

        if current_version != DEFAULT_CONFIG["version"]:
            self._config["version"] = DEFAULT_CONFIG["version"]
            changed = True
            logger.info(f"Configuration upgraded to version {DEFAULT_CONFIG['version']}")
        return changed

    def _merge_defaults(self, config: dict[str, Any], defaults: dict[str, Any]) -> bool:
        """Merge default values into config for missing keys.

        Args:
            config: Current configuration dictionary.
            defaults: Default configuration dictionary.
        """
        changed = False
        for key, value in defaults.items():
            if key not in config:
                config[key] = copy.deepcopy(value)
                changed = True
            elif isinstance(value, dict) and isinstance(config.get(key), dict):
                changed = self._merge_defaults(config[key], value) or changed
        return changed

    def _migrate_from_legacy(self) -> None:
        """Migrate settings from legacy individual files to JSON format.

        Uses a data-driven approach with migration rules to reduce complexity.
        Each rule specifies the legacy key, target config path, and value transformer.
        """
        migration_rules: list[_MigrationRule] = [
            _MigrationRule("lang", "ocr", "language", _transform_string),
            _MigrationRule("quality", "ocr", "quality", _transform_string),
            _MigrationRule("align", "ocr", "alignment", _transform_string),
            _MigrationRule("same_folder", "output", "save_in_same_folder", _transform_boolean),
            _MigrationRule("savefile", "output", "destination_folder", _transform_directory),
        ]

        migrated_count = sum(self._apply_migration_rule(rule) for rule in migration_rules)

        if migrated_count > 0:
            logger.info(f"Migrated {migrated_count} settings from legacy files to JSON")

    def _apply_migration_rule(self, rule: "_MigrationRule") -> bool:
        """Apply a single migration rule.

        Args:
            rule: The migration rule to apply.

        Returns:
            True if the migration was successful, False otherwise.
        """
        legacy_path = LEGACY_PATHS.get(rule.legacy_key)
        if not legacy_path or not os.path.exists(legacy_path):
            return False

        raw_value = self._read_legacy_file(legacy_path)
        if raw_value is None:
            return False

        transformed = rule.transformer(raw_value)
        if transformed is None:
            return False

        self._config[rule.section][rule.target_key] = transformed
        return True

    def _read_legacy_file(self, path: str) -> str | None:
        """Read and return stripped content from a legacy file.

        Args:
            path: Path to the legacy file.

        Returns:
            Stripped file content, or None if read fails.
        """
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if save was successful, False otherwise.
        """
        changes = self._config_changes(self._base_config, self._config)
        try:
            with self._exclusive_config_lock():
                if self._force_replace_digest is not None:
                    current_raw = self._read_raw_config()
                    current_digest = (
                        hashlib.sha256(current_raw.encode("utf-8")).hexdigest()
                        if current_raw is not None
                        else None
                    )
                    if current_digest != self._force_replace_digest:
                        raise ValueError(
                            "configuration changed while invalid data was being recovered"
                        )
                    published_config = copy.deepcopy(self._config)
                else:
                    published_config = self._load_latest_for_merge()
                    self._apply_config_changes(published_config, changes)
                    self._merge_defaults(published_config, DEFAULT_CONFIG)
                    published_config["version"] = DEFAULT_CONFIG["version"]

                write_text_file_atomically(
                    self.config_path,
                    lambda config_file: json.dump(
                        published_config,
                        config_file,
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Error saving config: {e}")
            return False

        self._config = published_config
        self._base_config = copy.deepcopy(published_config)
        self._force_replace_digest = None
        logger.debug("Configuration saved to JSON")
        return True

    def reload(self) -> bool:
        """Merge pending local edits, then refresh this instance from disk."""
        if self._config_changes(self._base_config, self._config) and not self.save():
            return False
        return self._load_config()

    def reset_to_defaults(self) -> bool:
        """Replace the complete settings document after an explicit user reset."""
        defaults = self._get_default_config()
        try:
            with self._exclusive_config_lock():
                write_text_file_atomically(
                    self.config_path,
                    lambda config_file: json.dump(
                        defaults,
                        config_file,
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
        except (OSError, TypeError, ValueError) as error:
            logger.error(f"Error resetting config: {error}")
            return False
        self._config = defaults
        self._base_config = copy.deepcopy(defaults)
        self._force_replace_digest = None
        return True

    @contextmanager
    def _exclusive_config_lock(self) -> Iterator[None]:
        lock_path = f"{self.config_path}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _read_raw_config(self) -> str | None:
        try:
            with open(
                self.config_path,
                encoding="utf-8",
                opener=lambda path, flags: os.open(
                    path,
                    flags
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                ),
            ) as config_file:
                if not stat.S_ISREG(os.fstat(config_file.fileno()).st_mode):
                    raise OSError("configuration path is not a regular file")
                return config_file.read()
        except FileNotFoundError:
            return None

    def _load_latest_for_merge(self) -> dict[str, Any]:
        raw_config = self._read_raw_config()
        if raw_config is None:
            return {}
        try:
            loaded_config = json.loads(raw_config)
        except json.JSONDecodeError as error:
            raise ValueError(f"cannot merge invalid configuration JSON: {error}") from error
        validation_error = self._config_validation_error(loaded_config)
        if validation_error is not None:
            raise ValueError(f"cannot merge invalid configuration: {validation_error}")
        return loaded_config

    @classmethod
    def _config_changes(
        cls,
        base: dict[str, Any],
        current: dict[str, Any],
        prefix: tuple[str, ...] = (),
    ) -> dict[tuple[str, ...], Any]:
        changes: dict[tuple[str, ...], Any] = {}
        for key in base.keys() | current.keys():
            path = (*prefix, key)
            if key not in current:
                changes[path] = _MISSING
                continue
            if key not in base:
                changes[path] = copy.deepcopy(current[key])
                continue
            base_value = base[key]
            current_value = current[key]
            if isinstance(base_value, dict) and isinstance(current_value, dict):
                changes.update(cls._config_changes(base_value, current_value, path))
            elif base_value != current_value:
                changes[path] = copy.deepcopy(current_value)
        return changes

    @staticmethod
    def _apply_config_changes(
        config: dict[str, Any],
        changes: dict[tuple[str, ...], Any],
    ) -> None:
        for path, value in sorted(changes.items(), key=lambda item: len(item[0])):
            parent: dict[str, Any] = config
            for key in path[:-1]:
                child = parent.get(key)
                if not isinstance(child, dict):
                    child = {}
                    parent[key] = child
                parent = child
            if value is _MISSING:
                parent.pop(path[-1], None)
            else:
                parent[path[-1]] = copy.deepcopy(value)

    def get(self, key_path: str, default: Any = None, expected_type: type | None = None) -> Any:
        """Get a configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path to the config value (e.g., "ocr.language")
            default: Default value if key not found
            expected_type: If provided, return default when value is not this type

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        if expected_type is not None and not isinstance(value, expected_type):
            return default

        return value

    def set(self, key_path: str, value: Any, save_immediately: bool = True) -> None:
        """Set a configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
            save_immediately: Whether to save to file immediately
        """
        keys = key_path.split(".")
        config = self._config

        # Navigate to parent key
        for key in keys[:-1]:
            if not isinstance(config.get(key), dict):
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

        if save_immediately:
            self.save()


# Singleton instance for global access
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance.

    Returns:
        The singleton ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
