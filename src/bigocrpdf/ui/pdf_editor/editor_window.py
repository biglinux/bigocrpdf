"""
BigOcrPdf - PDF Editor Window

Main editor window for PDF page manipulation before OCR processing.
Redesigned with a visible action bar for discoverability and accessibility.
"""

import os
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from bigocrpdf.ui.pdf_editor.editor_help_controller import EditorHelpController
from bigocrpdf.ui.pdf_editor.editor_page_actions_mixin import EditorPageActionsMixin
from bigocrpdf.ui.pdf_editor.editor_tools_controller import EditorToolsController
from bigocrpdf.ui.pdf_editor.editor_window_layout import EditorWindowLayoutMixin
from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger


def _canonical_source_path(path: str) -> str:
    """Return a stable local identity for materialization comparisons."""
    return os.path.realpath(os.path.abspath(path))


def requires_materialization(
    document: PDFDocument,
    original_path: str,
    original_page_count: int,
) -> bool:
    """Return whether metadata alone cannot represent the edited document."""
    canonical_original = _canonical_source_path(original_path)
    ordered_pages = sorted(document.pages, key=lambda page: page.position)
    actual_structure = [
        (
            page.position,
            _canonical_source_path(page.source_file or original_path),
            page.page_number,
        )
        for page in ordered_pages
    ]
    expected_structure = [
        (position, canonical_original, position + 1)
        for position in range(max(0, original_page_count))
    ]
    if actual_structure != expected_structure:
        return True
    return any(
        not page.deleted and page.included_for_ocr and (page.flip_horizontal or page.flip_vertical)
        for page in ordered_pages
    )


class PDFEditorWindow(EditorWindowLayoutMixin, EditorPageActionsMixin, Adw.Window):
    """Main PDF editor window.

    Provides a PDFArranger-style interface for:
    - Viewing page thumbnails in a grid
    - Selecting pages for OCR
    - Rotating pages
    - Deleting pages
    - Zooming thumbnails

    UI Layout:
    - Header bar: Back + Title + Apply button
    - Action bar: Add Files | Rotate L/R | Undo | Include/Exclude All | Zoom | Overflow menu
    - Content: Page grid with thumbnails
    - Status bar: Page/included counts
    - Notification banner: Revealer-based inline feedback

    Attributes:
        document: The PDFDocument being edited
        on_save_callback: Callback when saving changes
    """

    def __init__(
        self,
        application: Gtk.Application,
        pdf_path: str | None = None,
        on_save_callback: Callable[[PDFDocument], bool | None] | None = None,
        on_close_callback: Callable[[], None] | None = None,
        initial_state: dict | None = None,
        standalone: bool = False,
    ) -> None:
        """Initialize the PDF editor window.

        Args:
            application: The Gtk Application instance
            pdf_path: Path to the PDF file to edit (None for empty editor)
            on_save_callback: Callback when user saves changes
            on_close_callback: Callback when window is closed
            initial_state: Optional dictionary to restore page states
            standalone: If True, show Save As button instead of Apply
        """
        super().__init__(application=application)

        self._help = EditorHelpController(self)
        self._tools = EditorToolsController(self)
        self._pdf_path = pdf_path
        self._on_save_callback = on_save_callback
        self._on_close_callback = on_close_callback
        self._initial_state = initial_state
        self._standalone = standalone
        self._document: PDFDocument | None = None
        self._original_page_count: int | None = None
        self._undo_stack: list[list[dict]] = []
        self._notification_timer_id: int | None = None
        self._close_prepared = False

        # Window configuration
        if pdf_path:
            self.set_title(_("PDF Editor - {}").format(os.path.basename(pdf_path)))
        else:
            self.set_title(_("PDF Editor"))
        w, h = self._load_editor_window_size()
        self.set_default_size(w, h)
        self.set_modal(False)

        self._setup_actions()
        self._setup_ui()
        self._setup_keyboard_shortcuts()
        self._setup_drag_drop()
        self._load_document()

        # Connect close request handler
        self.connect("close-request", self._on_close_request)

        # Show help on first use
        if self._help.should_show():
            GLib.idle_add(self._help.show)

    def _setup_actions(self) -> None:
        """Set up window actions for the overflow menu."""
        action_group = Gio.SimpleActionGroup()

        actions = {
            "compress": self._tools.compress,
            "split-pages": self._tools.split_pages,
            "split-size": self._tools.split_size,
            "reverse": self._tools.reverse,
            "rotate-left": self._on_rotate_left,
            "rotate-right": self._on_rotate_right,
            "flip-horizontal": self._on_flip_horizontal,
            "flip-vertical": self._on_flip_vertical,
            "help": self._help.show,
        }

        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            action_group.add_action(action)

        self.insert_action_group("editor", action_group)

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        key_controller = Gtk.EventControllerKey()
        key_controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _setup_drag_drop(self) -> None:
        """Set up drag and drop for external files (PDFs and images)."""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList])
        drop_target.connect("drop", self._on_external_file_drop)
        self._split_view.add_controller(drop_target)

    # -- Undo stack ---------------------------------------------------------

    _MAX_UNDO = 50

    def _push_undo(self) -> None:
        """Snapshot current page state before a mutating operation."""
        if not self._document:
            return
        snapshot = [p.to_dict() for p in self._document.pages]
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)

    def _undo(self) -> None:
        """Restore the most recent page state snapshot."""
        if not self._undo_stack or not self._document:
            return
        snapshot = self._undo_stack.pop()
        self._document.pages = [PageState.from_dict(d) for d in snapshot]
        self._document.total_pages = len(self._document.pages)
        self._document.update_positions()
        self._page_grid.refresh()

    def _load_document(self) -> None:
        """Load the PDF document."""
        if not self._pdf_path:
            # Standalone mode without a file — start with empty document
            self._document = None
            self._update_status_bar()
            return

        try:
            renderer = get_thumbnail_renderer()
            page_count = renderer.get_page_count(self._pdf_path)

            if page_count == 0:
                self._show_error(_("Could not open PDF file or file has no pages."))
                return
            self._original_page_count = page_count

            # Load document if not already loaded
            if not self._document:
                if self._initial_state:
                    try:
                        logger.info("Restoring editor state from saved configuration")
                        self._document = PDFDocument.from_dict(self._initial_state)
                        # Ensure path matches current file
                        self._document.path = self._pdf_path
                    except Exception as e:
                        logger.error(f"Failed to restore state: {e}")
                        self._document = PDFDocument(
                            path=self._pdf_path,
                            total_pages=page_count,
                        )
                else:
                    self._document = PDFDocument(
                        path=self._pdf_path,
                        total_pages=page_count,
                    )
                self._page_grid.load_document(self._document)
            else:
                self._page_grid.load_document(self._document)
            self._update_status_bar()

            logger.info(f"Loaded PDF with {page_count} pages: {self._pdf_path}")

        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            self._show_error(_("Failed to load PDF: {}").format(str(e)))

    def _update_status_bar(self) -> None:
        """Update the status bar labels."""
        total = self._page_grid.get_total_pages()
        included = self._page_grid.get_ocr_count()
        selected_count = len(self._page_grid.selected_indices)

        self._status_label.set_text(
            ngettext(
                "{total} page · {included} included",
                "{total} pages · {included} included",
                total,
            ).format(total=total, included=included)
        )

        if selected_count > 0:
            self._selection_label.set_text(_("{count} selected").format(count=selected_count))
            self._selection_label.set_visible(True)
        else:
            self._selection_label.set_visible(False)

    def _on_selection_changed(self, grid: PageGrid) -> None:
        """Handle selection changes in the grid.

        Args:
            grid: The page grid
        """
        self._update_status_bar()
        count = len(grid._selected_indices)
        if count > 0:
            self.announce(
                ngettext(
                    "{count} page selected",
                    "{count} pages selected",
                    count,
                ).format(count=count),
                Gtk.AccessibleAnnouncementPriority.MEDIUM,
            )

    def _on_page_ocr_toggled(self, grid: PageGrid, page_num: int, active: bool) -> None:
        """Handle OCR toggle for a page.

        Args:
            grid: The page grid
            page_num: Page number
            active: New OCR state
        """
        logger.debug(f"Page {page_num} OCR toggled to {active}")
        self._update_status_bar()

    def _show_notification(self, message: str, icon_name: str, timeout: int = 3) -> None:
        """Show or update the inline notification banner.

        Args:
            message: Message text
            icon_name: Icon name for the notification
            timeout: Seconds before auto-hide (0 = persistent)
        """
        # Cancel previous timer
        if self._notification_timer_id is not None:
            GLib.source_remove(self._notification_timer_id)
            self._notification_timer_id = None

        self._notification_icon.set_from_icon_name(icon_name)
        self._notification_label.set_text(message)
        set_a11y_label(self._notification_box, message)
        self._notification_revealer.set_reveal_child(True)

        if timeout > 0:
            self._notification_timer_id = GLib.timeout_add_seconds(timeout, self._hide_notification)

    def _hide_notification(self) -> bool:
        """Hide the notification banner.

        Returns:
            False to stop the timer.
        """
        self._notification_revealer.set_reveal_child(False)
        self._notification_timer_id = None
        return False

    def _show_info(self, message: str, timeout: int = 3) -> None:
        """Show a success/info banner (e.g. save completed)."""
        self._show_notification(message, "emblem-ok-symbolic", timeout)

    def _show_saving(self) -> None:
        """Show a persistent 'Saving…' banner until the next notification."""
        self._show_notification(_("Saving…"), "document-save-symbolic", timeout=0)

    def _save_with_feedback(self, save_fn, success_message: str) -> None:
        """Run a blocking save while showing saving/saved feedback.

        Shows a persistent 'Saving…' banner, then defers the blocking save to
        the next idle cycle so the banner is painted before the UI freezes, and
        finally reports success or failure.

        Args:
            save_fn: Callable returning True on success.
            success_message: Banner text shown when save_fn returns True.
        """
        self._show_saving()

        def _run() -> bool:
            try:
                ok = save_fn()
            except Exception as e:
                logger.error("Save failed: %s", e)
                self._show_error(_("Failed to save PDF."))
                return False
            if ok:
                self._show_info(success_message)
            else:
                self._show_error(_("Failed to save PDF."))
            return False

        # Small delay so the "Saving…" banner is painted before the blocking
        # save freezes the main loop.
        GLib.timeout_add(50, _run)

    def _on_back_clicked(self, _button: Gtk.Button) -> None:
        """Handle back button click.

        Args:
            _button: The button widget
        """
        # User requested discard/cancel, so close without saving
        if self._document:
            self._document.clear_modifications()
        self._close_window()

    def _on_zoom_dropdown_changed(self, dropdown: Gtk.DropDown, _param) -> None:
        """Handle zoom dropdown selection change."""
        zoom_levels = [50, 75, 100, 150, 200, 300, 400]
        selected = dropdown.get_selected()
        if 0 <= selected < len(zoom_levels):
            self._page_grid.set_zoom_level(zoom_levels[selected])

    def _zoom_step(self, direction: int) -> None:
        """Step zoom in (+1) or out (-1) via keyboard."""
        current = self._zoom_dropdown.get_selected()
        model = self._zoom_dropdown.get_model()
        if model is None:
            return
        n_items = model.get_n_items()
        new_idx = max(0, min(n_items - 1, current + direction))
        if new_idx != current:
            self._zoom_dropdown.set_selected(new_idx)

    def _on_ok_clicked(self, _button: Gtk.Button) -> None:
        """Handle OK button click - apply changes and close.

        Args:
            _button: The button widget
        """
        if self._save_and_callback():
            self._close_window()

    def _on_page_layout_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Persist the viewer page-layout (/PageLayout) selection."""
        idx = combo.get_selected()
        if 0 <= idx < len(self._page_layout_values):
            get_config_manager().set(
                "output.page_layout", self._page_layout_values[idx], save_immediately=True
            )
            logger.info("Editor page layout changed to: %s", self._page_layout_values[idx])

    def _default_save_dir(self) -> str:
        """Pick the folder to open the Save dialog in.

        Prefers the directory of the originally opened source files (where the
        images/PDFs live), so a document assembled into a temporary file does
        not default the dialog to the temp dir. Falls back to the user's home
        when no source folder is writable; never returns the temp directory.
        """
        import tempfile

        candidates: list[str] = []
        if self._document:
            for page in self._document.get_active_pages():
                src = getattr(page, "source_file", "") or ""
                if src:
                    candidates.append(os.path.dirname(os.path.abspath(src)))
        if self._pdf_path:
            candidates.append(os.path.dirname(os.path.abspath(self._pdf_path)))

        tmp_dir = os.path.realpath(tempfile.gettempdir())
        for folder in candidates:
            if not folder or not os.path.isdir(folder) or not os.access(folder, os.W_OK):
                continue
            real = os.path.realpath(folder)
            if real == tmp_dir or real.startswith(tmp_dir + os.sep):
                continue
            return folder
        return os.path.expanduser("~")

    def _on_save_as_clicked(self, _button: Gtk.Button) -> None:
        """Handle Save As button click — show file dialog and save PDF."""
        if not self._document:
            return

        dialog = Gtk.FileDialog()
        dialog.set_title(_("Save PDF As"))

        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_mime_type("application/pdf")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(pdf_filter)
        dialog.set_filters(filters)

        if self._pdf_path:
            name = os.path.splitext(os.path.basename(self._pdf_path))[0]
            dialog.set_initial_name(f"{name}-edited.pdf")
        else:
            dialog.set_initial_name(_("document.pdf"))
        dialog.set_initial_folder(Gio.File.new_for_path(self._default_save_dir()))

        dialog.save(self, None, self._on_save_as_response)

    def _on_save_as_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle Save As file dialog response."""
        try:
            gfile = dialog.save_finish(result)
            if not gfile:
                return
            dest_path = gfile.get_path()
            if not dest_path:
                return
            document = self._document
            if document is None:
                return

            from bigocrpdf.ui.pdf_editor.page_operations import (
                apply_changes_to_pdf_atomically,
            )

            def _save() -> bool:
                if not apply_changes_to_pdf_atomically(document, dest_path):
                    return False
                document.clear_modifications()
                logger.info("Saved PDF via Save As: %s", dest_path)
                return True

            self._save_with_feedback(_save, _("Saved: {}").format(os.path.basename(dest_path)))
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error(f"Save As error: {e}")

    def _save_and_callback(self) -> bool:
        """Save editor changes and trigger callback.

        If only rotations/deletions on the original file, saves state
        metadata (no intermediate file). If pages from other files were
        added, creates a merged PDF.

        Returns:
            True when the changes were applied successfully.
        """
        if not self._document or not self._on_save_callback:
            return False

        try:
            original_path = self._document.path

            original_page_count = self._original_page_count
            if original_page_count is None:
                original_page_count = self._document.total_pages
            needs_merge = requires_materialization(
                self._document,
                original_path,
                original_page_count,
            )

            if needs_merge:
                return self._save_merged_pdf(original_path)

            # No merge needed — just pass modifications as state
            if self._on_save_callback(self._document) is False:
                self._show_error(_("Error saving changes."))
                return False
            self._document.clear_modifications()
            logger.info("Editor changes saved as metadata (no intermediate file)")
            return True

        except Exception as e:
            logger.error(f"Error saving editor changes: {e}")
            self._show_error(_("Error saving changes."))
            return False

    def _save_merged_pdf(self, original_path: str) -> bool:
        """Create a merged PDF when pages from multiple sources are present.

        Returns:
            True when the merged document was created and delivered.
        """
        from bigocrpdf.utils.temp_manager import mkstemp as _mkstemp
        from bigocrpdf.utils.temp_manager import remove_file as _rmfile

        # Tracked temp file — cleaned up after OCR or on exit
        fd, temp_path = _mkstemp(suffix=".pdf", prefix="bigocr_merge_")
        os.close(fd)

        logger.info("Merging pages from multiple sources into new PDF...")
        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        delivered = False
        if self._document is not None and apply_changes_to_pdf(self._document, temp_path):
            active_count = sum(page.included_for_ocr for page in self._document.get_active_pages())
            clean_doc = PDFDocument(path=temp_path, total_pages=active_count)
            for i in range(active_count):
                clean_doc.pages[i].source_file = temp_path
                clean_doc.pages[i].page_number = i + 1

            try:
                if self._on_save_callback and self._on_save_callback(clean_doc) is False:
                    self._show_error(_("Error saving changes."))
                    return False
                delivered = True
            finally:
                if not delivered:
                    _rmfile(temp_path)
            self._document.clear_modifications()
            logger.info(f"Merged PDF saved to {temp_path}")
            return True

        _rmfile(temp_path)
        self._show_error(_("Failed to merge PDF pages."))
        return False

    def _maybe_save_and_close(self) -> None:
        """Check for unsaved changes and close."""
        if self._document and self._document.modified:
            dialog = Adw.AlertDialog()
            dialog.set_heading(_("Unsaved Changes"))
            dialog.set_body(_("What would you like to do with your changes?"))

            dialog.add_response("discard", _("Discard"))
            dialog.add_response("save", _("Apply"))
            dialog.set_response_appearance("discard", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
            dialog.set_default_response("save")

            dialog.connect("response", self._on_save_dialog_response)
            dialog.present(self)
        else:
            self._close_window()

    def _on_save_dialog_response(self, dialog: Adw.AlertDialog, response: str) -> None:
        """Handle save dialog response.

        Args:
            dialog: The dialog
            response: Response ID
        """
        if response == "save":
            if self._save_and_callback():
                self._close_window()
        elif response == "discard":
            if self._document:
                self._document.clear_modifications()
            self._close_window()
        # "cancel" does nothing

    def _close_window(self) -> None:
        """Close the window."""
        self._prepare_close()
        if self._on_close_callback:
            self._on_close_callback()
        self.close()

    def _prepare_close(self) -> None:
        """Idempotently cancel window-owned timers and thumbnail requests."""
        if getattr(self, "_close_prepared", False):
            return
        self._close_prepared = True
        timer_id = getattr(self, "_notification_timer_id", None)
        self._notification_timer_id = None
        if timer_id is not None:
            GLib.source_remove(timer_id)
        page_grid = getattr(self, "_page_grid", None)
        if page_grid is not None:
            page_grid.cancel_thumbnail_requests()

    @staticmethod
    def _window_buttons_on_left() -> bool:
        """Detect if window buttons (close/min/max) are on the left side."""
        try:
            settings = Gio.Settings.new("org.gnome.desktop.wm.preferences")
            layout = settings.get_string("button-layout")
            if layout and ":" in layout:
                left, _right = layout.split(":", 1)
                if "close" in left:
                    return True
        except Exception:
            pass
        return False

    def _on_close_request(self, window: Adw.Window) -> bool:
        """Handle window close request."""
        self._save_editor_window_size()
        if self._document and self._document.modified:
            self._maybe_save_and_close()
            return True
        self._prepare_close()
        return False

    @staticmethod
    def _load_editor_window_size() -> tuple[int, int]:
        """Load editor window size from configuration."""
        config = get_config_manager()
        width = config.get("editor_window.width", 900)
        height = config.get("editor_window.height", 700)
        return max(width, 400), max(height, 300)

    def _save_editor_window_size(self) -> None:
        """Save current editor window size to configuration."""
        config = get_config_manager()
        width, height = self.get_width(), self.get_height()
        if width > 0 and height > 0:
            config.set("editor_window.width", width, save_immediately=False)
            config.set("editor_window.height", height, save_immediately=True)

    def _show_error(self, message: str) -> None:
        """Show an error dialog.

        Args:
            message: Error message
        """
        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Error"))
        dialog.set_body(message)
        dialog.add_response("ok", _("OK"))
        dialog.present(self)

    @property
    def document(self) -> PDFDocument | None:
        """Get the current document.

        Returns:
            The PDFDocument being edited
        """
        return self._document
