"""Interrupted OCR session discovery and UI recovery."""

from typing import TYPE_CHECKING

import gi

gi.require_version("Adw", "1")
from gi.repository import Adw

from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.processing_controller import ProcessingController
    from bigocrpdf.services.settings import OcrSettings
    from bigocrpdf.ui.window_ui import BigOcrPdfUI


class SessionRecoveryController:
    """Own discovery, resumption, and dismissal of interrupted sessions."""

    def __init__(
        self,
        parent: Adw.ApplicationWindow,
        settings: "OcrSettings",
        ui: "BigOcrPdfUI",
        processing: "ProcessingController",
    ) -> None:
        self._parent = parent
        self._settings = settings
        self._ui = ui
        self._processing = processing

    def check(self) -> None:
        """Offer recovery when an incomplete processing session exists."""
        processor = self._processing.ocr_processor
        if not processor.has_resumable_session():
            return

        session_info = processor.get_resumable_session_info()
        if not session_info:
            return

        logger.info(
            f"Found incomplete session with {session_info.get('pending_files', 0)} pending files"
        )
        pending = session_info.get("pending_files", 0)
        completed = session_info.get("completed_files", 0)
        total = session_info.get("total_files", 0)
        dialog = Adw.AlertDialog(
            heading=_("Resume Previous Session?"),
            body=_(
                "An incomplete processing session was found.\n\n"
                "Completed: {completed}/{total}\n"
                "Remaining: {pending}\n\n"
                "Would you like to resume processing?"
            ).format(completed=completed, total=total, pending=pending),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("discard", _("Discard"))
        dialog.add_response("resume", _("Resume"))
        dialog.set_response_appearance("discard", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("resume", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("resume")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_response)
        dialog.present(self._parent)

    def _on_response(self, _dialog: Adw.AlertDialog, response: str) -> None:
        """Resume or discard only when the matching action is explicit."""
        if response == "resume":
            self._resume()
        elif response == "discard":
            self._discard()

    def _resume(self) -> None:
        """Restore pending files and update the queue UI."""
        if self._processing.ocr_processor.resume_previous_session():
            self._ui.settings_page_manager._populate_file_list()
            file_count = len(self._settings.selected_files)
            self._ui.custom_header_bar.update_queue_size(file_count)
            self._ui.show_toast(
                ngettext(
                    "Session resumed with {count} file",
                    "Session resumed with {count} files",
                    file_count,
                ).format(count=file_count)
            )
            logger.info("User chose to resume previous session")
            return

        logger.warning("Failed to resume session")
        self._ui.show_toast(_("Could not resume session"))

    def _discard(self) -> None:
        """Discard the interrupted session checkpoint."""
        if not self._processing.ocr_processor.discard_previous_session():
            logger.warning("Failed to discard previous incomplete session")
            return
        logger.info("User discarded previous incomplete session")
        self._ui.show_toast(_("Previous session discarded"))
