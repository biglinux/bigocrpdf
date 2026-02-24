"""
BigOcrPdf - Configuration Manager

This module provides centralized JSON-based configuration management.
It handles loading, saving, and migrating settings from legacy file format.
"""

import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

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

        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Load or create configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info("Configuration loaded from JSON")

                # Upgrade config if needed
                self._upgrade_config()

            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Error loading config: {e}")
                self._config = self._get_default_config()
                self._migrate_from_legacy()
        else:
            # Check for legacy files and migrate
            self._config = self._get_default_config()
            self._migrate_from_legacy()
            self.save()

    def _get_default_config(self) -> dict[str, Any]:
        """Get a copy of the default configuration.

        Returns:
            Deep copy of default configuration dictionary.
        """
        return copy.deepcopy(DEFAULT_CONFIG)

    def _upgrade_config(self) -> None:
        """Upgrade configuration to latest version if needed."""
        current_version = self._config.get("version", 0)

        if current_version < DEFAULT_CONFIG["version"]:
            # Add any missing keys from default config
            self._merge_defaults(self._config, DEFAULT_CONFIG)
            self._config["version"] = DEFAULT_CONFIG["version"]
            logger.info(f"Configuration upgraded to version {DEFAULT_CONFIG['version']}")

    def _merge_defaults(self, config: dict, defaults: dict) -> None:
        """Merge default values into config for missing keys.

        Args:
            config: Current configuration dictionary.
            defaults: Default configuration dictionary.
        """
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config.get(key), dict):
                self._merge_defaults(config[key], value)

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
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.debug("Configuration saved to JSON")
            return True
        except OSError as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path to the config value (e.g., "ocr.language")
            default: Default value if key not found

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
            if key not in config:
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
