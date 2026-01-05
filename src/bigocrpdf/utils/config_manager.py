"""
BigOcrPdf - Configuration Manager

This module provides centralized JSON-based configuration management.
It handles loading, saving, and migrating settings from legacy file format.
"""

import copy
import json
import os
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
        "language": "eng",
        "quality": "normal",
        "alignment": "alignrotate",
    },
    "output": {
        "suffix": "ocr",
        "overwrite_existing": False,
        "save_in_same_folder": False,
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
        self._dirty = False

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
            self._dirty = True
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
        """Migrate settings from legacy individual files."""
        migrated = False

        # Migrate language setting
        if os.path.exists(LEGACY_PATHS["lang"]):
            try:
                with open(LEGACY_PATHS["lang"], encoding="utf-8") as f:
                    lang = f.read().strip()
                    if lang:
                        self._config["ocr"]["language"] = lang
                        migrated = True
            except OSError:
                pass

        # Migrate quality setting
        if os.path.exists(LEGACY_PATHS["quality"]):
            try:
                with open(LEGACY_PATHS["quality"], encoding="utf-8") as f:
                    quality = f.read().strip()
                    if quality:
                        self._config["ocr"]["quality"] = quality
                        migrated = True
            except OSError:
                pass

        # Migrate alignment setting
        if os.path.exists(LEGACY_PATHS["align"]):
            try:
                with open(LEGACY_PATHS["align"], encoding="utf-8") as f:
                    align = f.read().strip()
                    if align:
                        self._config["ocr"]["alignment"] = align
                        migrated = True
            except OSError:
                pass

        # Migrate same folder setting
        if os.path.exists(LEGACY_PATHS["same_folder"]):
            try:
                with open(LEGACY_PATHS["same_folder"], encoding="utf-8") as f:
                    value = f.read().strip().lower()
                    self._config["output"]["save_in_same_folder"] = value == "true"
                    migrated = True
            except OSError:
                pass

        # Migrate destination folder
        if os.path.exists(LEGACY_PATHS["savefile"]):
            try:
                with open(LEGACY_PATHS["savefile"], encoding="utf-8") as f:
                    folder = f.read().strip()
                    if folder and os.path.isdir(folder):
                        self._config["output"]["destination_folder"] = folder
                        migrated = True
            except OSError:
                pass

        if migrated:
            logger.info("Migrated settings from legacy files to JSON")
            self._dirty = True

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if save was successful, False otherwise.
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            self._dirty = False
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
        self._dirty = True

        if save_immediately:
            self.save()

    def get_section(self, section: str) -> dict[str, Any]:
        """Get an entire configuration section.

        Args:
            section: Section name (e.g., "ocr", "output")

        Returns:
            Dictionary of section settings
        """
        return self._config.get(section, {}).copy()

    def set_section(
        self, section: str, values: dict[str, Any], save_immediately: bool = True
    ) -> None:
        """Set an entire configuration section.

        Args:
            section: Section name
            values: Dictionary of settings to set
            save_immediately: Whether to save to file immediately
        """
        self._config[section] = values
        self._dirty = True

        if save_immediately:
            self.save()

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty

    def reset_to_defaults(self, save_immediately: bool = True) -> None:
        """Reset all settings to default values.

        Args:
            save_immediately: Whether to save to file immediately
        """
        self._config = self._get_default_config()
        self._dirty = True

        if save_immediately:
            self.save()

        logger.info("Configuration reset to defaults")

    def export_to_dict(self) -> dict[str, Any]:
        """Export configuration as a dictionary.

        Returns:
            Copy of the current configuration
        """
        return copy.deepcopy(self._config)

    def import_from_dict(self, data: dict[str, Any], save_immediately: bool = True) -> None:
        """Import configuration from a dictionary.

        Args:
            data: Configuration dictionary to import
            save_immediately: Whether to save to file immediately
        """
        self._config = data
        self._dirty = True

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


def get_config(key_path: str, default: Any = None) -> Any:
    """Convenience function to get a configuration value.

    Args:
        key_path: Dot-separated path to the config value
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return get_config_manager().get(key_path, default)


def set_config(key_path: str, value: Any, save_immediately: bool = True) -> None:
    """Convenience function to set a configuration value.

    Args:
        key_path: Dot-separated path to the config value
        value: Value to set
        save_immediately: Whether to save to file immediately
    """
    get_config_manager().set(key_path, value, save_immediately)
