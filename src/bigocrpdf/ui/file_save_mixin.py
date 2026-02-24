"""File Save and Export Dialog Mixin for DialogsManager."""

import os
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class FileSaveDialogMixin:
    """Mixin providing file save/export dialog management."""

    def _save_text_to_file(self, text: str) -> None:
        """Save text to file with dialog

        Args:
            text: Text to save
        """
        # Create file save dialog
        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Save Extracted Text"))
        save_dialog.set_modal(True)
        save_dialog.set_initial_name("extracted_text.txt")

        # Show save dialog
        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_save_dialog_response(d, r, text),
        )

    def _on_save_dialog_response(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, text: str
    ) -> None:
        """Handle save dialog response

        Args:
            dialog: File dialog
            result: Async result
            text: Text to save
        """
        try:
            file = dialog.save_finish(result)
            file_path = file.get_path()

            # Check if file exists
            if os.path.exists(file_path):
                self._show_file_exists_dialog(file_path, text)
                return

            # Save file
            self._write_text_to_file(file_path, text)

        except Exception as e:
            if "Dismissed" not in str(e):
                logger.error(f"Error saving text to file: {e}")
                self._show_error_dialog(_("Save Failed"), str(e))

    def _show_file_exists_dialog(self, file_path: str, text: str) -> None:
        """Show dialog for handling existing files

        Args:
            file_path: Path to existing file
            text: Text to save
        """
        dialog = Adw.AlertDialog(
            heading=_("File Already Exists"),
            body=_("The file '{0}' already exists. What would you like to do?").format(
                os.path.basename(file_path)
            ),
        )

        dialog.add_response("overwrite", _("Overwrite"))
        dialog.add_response("rename", _("Auto-Rename"))
        dialog.add_response("cancel", _("Cancel"))

        dialog.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect("response", self._on_file_exists_response, file_path, text)
        dialog.present(self.window)

    def _on_file_exists_response(
        self, dialog: Adw.AlertDialog, response: str, file_path: str, text: str
    ) -> None:
        """Handle file exists dialog response

        Args:
            dialog: Message dialog
            response: Response ID
            file_path: File path
            text: Text to save
        """
        if response == "overwrite":
            self._write_text_to_file(file_path, text)
        elif response == "rename":
            new_path = self._generate_unique_filename(file_path)
            self._write_text_to_file(new_path, text)

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate unique filename by appending number

        Args:
            file_path: Original file path

        Returns:
            Unique file path
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{name}-{counter}{ext}")
            counter += 1

        return new_path

    def _write_text_to_file(self, file_path: str, text: str) -> None:
        """Write text to file

        Args:
            file_path: Path to save file
            text: Text content
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"Text saved to {file_path}")
            self._show_success_toast(
                _("Text saved to {filename}").format(filename=os.path.basename(file_path))
            )

        except Exception as e:
            logger.error(f"Error writing text to file: {e}")
            self._show_error_dialog(_("Save Failed"), str(e))

    def _show_success_toast(self, message: str) -> None:
        """Show success toast notification

        Args:
            message: Message to display
        """
        if hasattr(self.window, "toast_overlay") and self.window.toast_overlay:
            toast = Adw.Toast.new(message)
            toast.set_timeout(3)
            self.window.toast_overlay.add_toast(toast)

    def _show_error_dialog(self, title: str, message: str) -> None:
        """Show error dialog

        Args:
            title: Dialog title
            message: Error message
        """
        error_dialog = Adw.AlertDialog(heading=title, body=message)
        error_dialog.add_response("ok", _("OK"))
        error_dialog.present(self.window)

    def show_resume_session_dialog(
        self, session_info: dict, on_resume: Callable[[], None], on_discard: Callable[[], None]
    ) -> None:
        """Show dialog asking user whether to resume an incomplete session.

        Args:
            session_info: Dictionary with session information
            on_resume: Callback when user chooses to resume
            on_discard: Callback when user chooses to discard
        """
        pending = session_info.get("pending_files", 0)
        completed = session_info.get("completed_files", 0)
        total = session_info.get("total_files", 0)

        dialog = Adw.AlertDialog(
            heading=_("Resume Previous Session?"),
            body=_(
                "An incomplete processing session was found.\n\n"
                "Progress: {completed} of {total} files completed\n"
                "Remaining: {pending} files\n\n"
                "Would you like to resume processing the remaining files?"
            ).format(completed=completed, total=total, pending=pending),
        )
        dialog.add_response("discard", _("Discard"))
        dialog.add_response("resume", _("Resume"))
        dialog.set_response_appearance("discard", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("resume", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("resume")

        def on_response(d: Adw.AlertDialog, response: str) -> None:
            if response == "resume":
                on_resume()
            else:
                on_discard()

        dialog.connect("response", on_response)
        dialog.present(self.window)
