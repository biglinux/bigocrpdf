"""Actions, drag-and-drop, preferences, and session UI for the main window."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from bigocrpdf.ui.session_recovery_controller import SessionRecoveryController
from bigocrpdf.ui.settings_reset_controller import SettingsResetController
from bigocrpdf.ui.welcome_dialog_controller import WelcomeDialogController
from bigocrpdf.ui.widgets import get_default_clipboard, parse_clipboard_file_paths
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file
from bigocrpdf.utils.temp_manager import mkstemp, remove_file

if TYPE_CHECKING:
    from bigocrpdf import OcrDependencyState
    from bigocrpdf.processing_controller import ProcessingController
    from bigocrpdf.services.settings import OcrSettings
    from bigocrpdf.ui.file_selection_manager import FileSelectionManager
    from bigocrpdf.ui.window_ui import BigOcrPdfUI

_QUEUE_ACTION_NAMES = (
    "add-files",
    "start-processing",
    "remove-all-files",
    "paste-clipboard",
)


class WindowController:
    """Own main-window actions and transient interaction workflows."""

    def __init__(
        self,
        *,
        parent: Adw.ApplicationWindow,
        settings: OcrSettings,
        ui: BigOcrPdfUI,
        file_manager: FileSelectionManager,
        processing: ProcessingController,
        ocr_dependency: OcrDependencyState,
        show_ocr_unavailable: Callable[[], bool],
        welcome_config_path: str,
    ) -> None:
        self.parent = parent
        self.settings = settings
        self.ui = ui
        self.file_manager = file_manager
        self.processing = processing
        self.ocr_dependency = ocr_dependency
        self._show_ocr_unavailable = show_ocr_unavailable
        self.welcome = WelcomeDialogController(parent, welcome_config_path)
        self.sessions = SessionRecoveryController(parent, settings, ui, processing)
        self.reset = SettingsResetController(
            parent,
            settings,
            ui.settings_page_manager.sync_ui_to_settings,
            ui.show_toast,
        )

    def setup_window_actions(self) -> None:
        """Set up window-level actions for keyboard shortcuts."""
        # Add files action (Ctrl+O)
        add_files_action = Gio.SimpleAction.new("add-files", None)
        add_files_action.connect("activate", self._on_add_files_action)
        self.parent.add_action(add_files_action)

        # Start processing action (Ctrl+Enter)
        start_action = Gio.SimpleAction.new("start-processing", None)
        start_action.connect("activate", self._on_start_processing_action)
        self.parent.add_action(start_action)

        # Cancel processing action (Escape)
        cancel_action = Gio.SimpleAction.new("cancel-processing", None)
        cancel_action.connect("activate", self._on_cancel_processing_action)
        self.parent.add_action(cancel_action)

        # Remove all files action (Ctrl+R)
        remove_all_action = Gio.SimpleAction.new("remove-all-files", None)
        remove_all_action.connect("activate", self._on_remove_all_files_action)
        self.parent.add_action(remove_all_action)

        # Paste from clipboard action (Ctrl+V)
        paste_action = Gio.SimpleAction.new("paste-clipboard", None)
        paste_action.connect("activate", self._on_paste_clipboard_action)
        self.parent.add_action(paste_action)

        self.sync_for_page()
        logger.info("Window actions set up for keyboard shortcuts")

    def _is_queue_page(self) -> bool:
        """Return whether the queue is the actual visible main page."""
        return self.ui.main_stack.get_visible_child_name() == "main_view"

    def sync_for_page(self, page_name: str | None = None) -> None:
        """Enable only the window actions valid for the visible main page."""
        current_page = page_name or self.ui.main_stack.get_visible_child_name()
        is_queue_page = current_page == "main_view"

        for action_name in _QUEUE_ACTION_NAMES:
            action = self.parent.lookup_action(action_name)
            if action is not None:
                simple_action = cast(Gio.SimpleAction, action)
                is_enabled = is_queue_page
                if action_name == "start-processing":
                    is_enabled = is_enabled and self.ocr_dependency.is_available
                simple_action.set_enabled(is_enabled)

        cancel_action = self.parent.lookup_action("cancel-processing")
        if cancel_action is not None:
            cast(Gio.SimpleAction, cancel_action).set_enabled(current_page == "terminal")

    def _on_add_files_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle add files shortcut (Ctrl+O)."""
        if self._is_queue_page():
            self.file_manager.show_open_files_dialog()

    def _on_start_processing_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle start processing shortcut (Ctrl+Enter)."""
        if self._is_queue_page() and self.settings.selected_files:
            if not self.ocr_dependency.is_available:
                self._show_ocr_unavailable()
                return
            self.processing.start()

    def _on_cancel_processing_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle cancel processing shortcut (Escape)."""
        current_page = self.ui.main_stack.get_visible_child_name()
        if current_page == "terminal":
            self.processing.cancel()

    def _on_remove_all_files_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle remove all files shortcut (Ctrl+R)."""
        if self._is_queue_page():
            self.ui.settings_page_manager._remove_all_files()

    # ------------------------------------------------------------------
    # Clipboard paste (Ctrl+V)
    # ------------------------------------------------------------------

    def _on_paste_clipboard_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle paste from clipboard (Ctrl+V).

        Reads the system clipboard and adds images or file URIs to the
        processing queue. Supports:
        - Image data (screenshots / copied images)
        - File URIs (files copied in a file manager)
        """
        if not self._is_queue_page():
            return

        clipboard = get_default_clipboard()
        if clipboard is None:
            logger.warning("Clipboard is unavailable because no display is active")
            return
        formats = clipboard.get_formats()

        # Prefer file URIs first (copied files from file manager)
        # File managers use x-special/gnome-copied-files or text/uri-list
        uri_mime_types = ["x-special/gnome-copied-files", "text/uri-list"]
        has_uris = any(formats.contain_mime_type(m) for m in uri_mime_types)

        if has_uris:
            clipboard.read_async(
                uri_mime_types,
                GLib.PRIORITY_DEFAULT,
                None,
                self._on_clipboard_uris_ready,
            )
        # Then try image texture
        elif formats.contain_gtype(Gdk.Texture):
            clipboard.read_texture_async(None, self._on_clipboard_texture_ready)
        else:
            logger.debug("Clipboard has no image or file URI content")

    def _on_clipboard_uris_ready(self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult) -> None:
        """Handle clipboard data containing file URIs."""
        raw = self._read_clipboard_uri_text(clipboard, result)
        if raw is None or not self._is_queue_page():
            return

        file_paths = parse_clipboard_file_paths(raw)
        if not file_paths:
            logger.debug("Clipboard URIs contain no valid file paths")
            return

        supported = self._supported_clipboard_files(file_paths)
        if not supported:
            self.ui.show_toast(_("No supported files in clipboard"))
            return

        added = self._add_clipboard_supported_files(supported)
        if added:
            self.ui.update_file_info()
            self.ui.show_toast(
                ngettext(
                    "{count} file added from clipboard",
                    "{count} files added from clipboard",
                    added,
                ).format(count=added)
            )

    def _read_clipboard_uri_text(
        self,
        clipboard: Gdk.Clipboard,
        result: Gio.AsyncResult,
    ) -> str | None:
        try:
            stream, _mime = clipboard.read_finish(result)
        except GLib.Error as error:
            logger.error("Failed to read clipboard URIs: %s", error)
            return None

        if stream is None:
            return None

        try:
            data = stream.read_bytes(1024 * 1024, None).get_data()
            if data is None:
                return None
            raw = data.decode("utf-8", errors="replace")
            return raw
        except GLib.Error as error:
            logger.error("Failed to read clipboard stream: %s", error)
            return None
        finally:
            try:
                stream.close(None)
            except GLib.Error as error:
                logger.error("Failed to close clipboard stream: %s", error)

    def _supported_clipboard_files(self, file_paths: list[str]) -> list[str]:
        return [p for p in file_paths if p.lower().endswith(".pdf") or is_image_file(p)]

    def _add_clipboard_supported_files(self, supported: list[str]) -> int:
        image_files = [path for path in supported if is_image_file(path)]
        pdf_files = [path for path in supported if not is_image_file(path)]
        added = self.settings.add_files(pdf_files) if pdf_files else 0

        if len(image_files) > 1:
            self._show_clipboard_image_merge_dialog(image_files)
        elif image_files and self._add_clipboard_image_as_pdf(image_files[0]):
            added += 1

        return added

    def _show_clipboard_image_merge_dialog(self, image_files: list[str]) -> None:
        self.ui.dialogs_manager.show_image_merge_dialog(
            image_files,
            heading=_("Multiple Images Pasted"),
            body=ngettext(
                "You pasted {count} image. How would you like to add it?",
                "You pasted {count} images. How would you like to add them?",
                len(image_files),
            ).format(count=len(image_files)),
            on_complete=self.ui.update_file_info,
        )

    def _add_clipboard_image_as_pdf(self, image_file: str) -> bool:
        try:
            pdf_path = images_to_pdf([image_file])
            return self.settings._add_generated_file(pdf_path, image_file)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Failed to convert pasted image to PDF: %s", error)
            return False

    def _on_clipboard_texture_ready(
        self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult
    ) -> None:
        """Handle clipboard image texture (e.g. screenshot)."""
        try:
            texture = clipboard.read_texture_finish(result)
        except GLib.Error as error:
            logger.error("Failed to read clipboard image: %s", error)
            return

        if texture is None:
            return
        if not self._is_queue_page():
            return

        # Save texture as temporary PNG
        tmp_path: str | None = None
        try:
            png_bytes = texture.save_to_png_bytes()
            png_data = png_bytes.get_data()
            if png_data is None:
                return
            fd, tmp_path = mkstemp(suffix=".png", prefix="bigocrpdf_paste_")
            with os.fdopen(fd, "wb") as image_file:
                image_file.write(png_data)
        except (GLib.Error, OSError) as error:
            logger.error("Failed to save clipboard image: %s", error)
            if tmp_path is not None:
                remove_file(tmp_path)
            return

        # Convert to PDF and add to queue
        try:
            pdf_path = images_to_pdf([tmp_path])
            if self.settings._add_generated_file(pdf_path, tmp_path):
                self.ui.update_file_info()
                self.ui.show_toast(_("Image from clipboard added"))
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Failed to convert clipboard image to PDF: %s", error)
        finally:
            remove_file(tmp_path)

    def setup_global_drag_drop(self) -> None:
        """Set up global drag and drop for the entire window."""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.connect("drop", self._on_global_drop)
        drop_target.connect("enter", self._on_global_drop_enter)
        self.parent.add_controller(drop_target)
        logger.info("Global drag & drop enabled")

    def _on_global_drop(self, _drop_target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Handle global file drop on the window.

        Args:
            _drop_target: The drop target controller
            value: The dropped file or files
            _x: X coordinate
            _y: Y coordinate

        Returns:
            True if drop was handled
        """
        if not self._is_queue_page():
            return False

        return self.ui.settings_page_manager._on_drop(_drop_target, value, _x, _y)

    def _on_global_drop_enter(
        self, _drop_target: Gtk.DropTarget, _x: float, _y: float
    ) -> Gdk.DragAction:
        """Handle drag enter on the window."""
        if self._is_queue_page():
            return Gdk.DragAction.COPY
        return Gdk.DragAction(0)

    def window_buttons_on_left(self) -> bool:
        """Detect if window buttons (close/min/max) are on the left side."""
        settings = Gtk.Settings.get_default()
        if settings is None:
            return False

        layout = settings.get_property("gtk-decoration-layout") or ""
        left, separator, _right = layout.partition(":")
        return bool(separator and "close" in left.split(","))

    def clear_file_queue(self) -> None:
        """Clear all files from the queue and repaint the queue from the model.

        The repaint is unconditional because an empty model does not imply an
        empty screen: every file is removed as it finishes processing, while
        the terminal page is showing and nothing repaints. Returning early on
        "the model did not change" therefore left a finished batch still listed
        and the header button stuck on the label it took when processing began.

        It repaints from the model rather than asserting zero, because
        ``_clear_files`` also reports False when saving failed and the queue was
        rolled back -- and in that case the files are still there.
        """
        self.settings._clear_files()
        self.ui.update_file_info()
