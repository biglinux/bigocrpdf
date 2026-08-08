"""Confirmed settings reset while preserving the active file queue."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.services.settings import OcrSettings


class SettingsResetController:
    """Own the reset confirmation and queue-preserving transaction."""

    def __init__(
        self,
        parent: Adw.ApplicationWindow,
        settings: "OcrSettings",
        sync_ui_to_settings: Callable[[], None],
        show_toast: Callable[[str], None],
    ) -> None:
        self._parent = parent
        self._settings = settings
        self._sync_ui_to_settings = sync_ui_to_settings
        self._show_toast = show_toast

    def confirm(self) -> None:
        """Ask before restoring all settings to defaults."""
        dialog = Adw.AlertDialog(
            heading=_("Reset All Settings?"),
            body=_(
                "This will restore all options to their default values. "
                "Your file queue will not be affected."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("reset", _("Reset"))
        dialog.set_response_appearance("reset", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_response)
        dialog.present(self._parent)

    def _on_response(self, _dialog: Adw.AlertDialog, response: str) -> None:
        """Restore defaults while retaining the current queue state."""
        if response != "reset":
            return

        selected_files = list(self._settings.selected_files)
        original_paths = dict(self._settings.original_file_paths)
        page_ranges = dict(self._settings.page_ranges)
        file_modifications = dict(self._settings.file_modifications)

        try:
            self._settings.reset_to_defaults()
        except OSError as error:
            logger.error("Failed to reset settings: %s", error)
            self._show_toast(_("Error saving settings: {0}").format(error))
            return
        finally:
            self._settings.selected_files = selected_files
            self._settings.original_file_paths = original_paths
            self._settings.page_ranges = page_ranges
            self._settings.file_modifications = file_modifications
            self._sync_ui_to_settings()

        logger.info("Settings reset to defaults via menu")
        self._show_toast(_("Settings restored to defaults"))
