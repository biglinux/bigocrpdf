"""PDF editor window: page manipulation before OCR, with a visible action bar."""

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
        save_in_place: bool = False,
    ) -> None:
        """Initialize the PDF editor window.

        Args:
            application: The Gtk Application instance
            pdf_path: Path to the PDF file to edit (None for empty editor)
            on_save_callback: Callback when user saves changes
            on_close_callback: Callback when window is closed
            initial_state: Optional dictionary to restore page states
            standalone: If True, show Save As button instead of Apply
            save_in_place: If True, Apply rewrites ``pdf_path`` instead of
                handing the changes to ``on_save_callback``
        """
        super().__init__(application=application)
        self.add_css_class("bigocrpdf-editor")

        self._help = EditorHelpController(self)
        self._tools = EditorToolsController(self)
        self._pdf_path = pdf_path
        self._on_save_callback = on_save_callback
        self._on_close_callback = on_close_callback
        self._initial_state = initial_state
        self._standalone = standalone
        self._save_in_place = save_in_place
        self._document: PDFDocument | None = None
        self._original_page_count: int | None = None
        self._undo_stack: list[list[dict]] = []
        self._notification_timer_id: int | None = None
        # The banner of work running in the background, shown again whenever
        # another message is dismissed before that work ends.
        self._busy_message: str | None = None
        self._close_prepared = False
        # A save runs in a worker; edits wait for it, and a close request
        # made meanwhile is carried out once it succeeds.
        self._saving = False
        self._close_after_save = False

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

        self.connect("close-request", self._on_close_request)

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

        self._actions: dict[str, Gio.SimpleAction] = {}
        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            action_group.add_action(action)
            self._actions[name] = action

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
            # Standalone mode without a file: start with an empty document
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
            self._notification_timer_id = GLib.timeout_add_seconds(
                timeout, self._on_notification_timeout
            )

    def _on_notification_timeout(self) -> bool:
        self._notification_timer_id = None
        self._hide_notification()
        return False

    def _hide_notification(self) -> None:
        """Hide the banner, or return to the busy banner while that work runs."""
        if self._notification_timer_id is not None:
            GLib.source_remove(self._notification_timer_id)
            self._notification_timer_id = None
        if self._busy_message is not None:
            self._show_notification(self._busy_message, "content-loading-symbolic", timeout=0)
            return
        self._notification_revealer.set_reveal_child(False)

    def _show_busy(self, message: str) -> None:
        """Show a banner that stays until ``_end_busy``."""
        self._busy_message = message
        self._show_notification(message, "content-loading-symbolic", timeout=0)
        self.announce(message, Gtk.AccessibleAnnouncementPriority.MEDIUM)

    def _end_busy(self) -> None:
        """Drop the busy banner, leaving any other message to its own timer."""
        message, self._busy_message = self._busy_message, None
        if self._notification_label.get_text() == message:
            self._hide_notification()

    def _show_info(self, message: str, timeout: int = 3) -> None:
        """Show a success/info banner (e.g. save completed)."""
        self._show_notification(message, "emblem-ok-symbolic", timeout)

    def _run_save(
        self, name: str, write: Callable[[], None], on_written: Callable[[], None]
    ) -> None:
        """Run ``write`` in a worker with editing locked, then ``on_written`` on the main loop.

        On failure the edits stay in the editor and the error names the cause.
        """
        from bigocrpdf.ui.pdf_editor.page_thumbnail_export import run_in_worker

        def work() -> Exception | None:
            try:
                write()
            except Exception as error:
                logger.exception("Could not save %s", name)
                return error
            return None

        def finish(error: Exception | None) -> None:
            self._set_saving(False)
            if error is None:
                on_written()
                return
            self._close_after_save = False
            self._show_error(
                _("Could not save {name}: {error}").format(name=name, error=error)
                + "\n\n"
                + _(
                    "Your changes are still in the editor. Check that the source files "
                    "still exist and that the disk has free space, then save again."
                )
            )

        self._set_saving(True)
        run_in_worker(work, finish)

    def _set_saving(self, saving: bool) -> None:
        """Lock the controls that edit the document while a save writes it."""
        self._saving = saving
        for widget in (
            self._sidebar_box,
            self._page_grid,
            self._add_button_header,
            self._apply_button,
        ):
            widget.set_sensitive(not saving)
        if saving:
            self._show_busy(_("Saving…"))
        else:
            self._end_busy()

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

    def _on_ok_clicked(self, _button: Gtk.Button | None) -> None:
        """Apply the changes and go back."""
        self._apply_and_close()

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

    def _on_save_as_clicked(self, _button: Gtk.Button | None) -> None:
        """Ask where to save the PDF, then save it."""
        self._save_as(close_after=False)

    def _save_as(self, close_after: bool) -> None:
        """Show the Save dialog and save the PDF, closing the editor afterwards if asked."""
        if not self._document or self._saving:
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

        dialog.save(self, None, lambda d, r: self._on_save_as_response(d, r, close_after))

    def _on_save_as_response(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, close_after: bool
    ) -> None:
        """Save the PDF to the chosen file."""
        try:
            gfile = dialog.save_finish(result)
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error(f"Save As error: {e}")
            return
        dest_path = gfile.get_path() if gfile else None
        if dest_path:
            self._publish(dest_path, close_after)

    def _publish(self, dest_path: str, close_after: bool) -> None:
        """Write the PDF to ``dest_path`` in a worker, closing the editor afterwards if asked."""
        document = self._document
        if document is None or self._saving:
            return

        from bigocrpdf.ui.pdf_editor.page_operations import publish_changes_to_pdf

        name = os.path.basename(dest_path)
        snapshot = document.snapshot()

        def saved() -> None:
            document.clear_modifications()
            logger.info("Saved PDF: %s", dest_path)
            self._show_info(_("Saved: {}").format(name))
            if close_after or self._close_after_save:
                self._close_window()

        self._run_save(name, lambda: publish_changes_to_pdf(snapshot, dest_path), saved)

    def _apply_and_close(self) -> None:
        """Hand the changes to the caller, or rewrite the opened file, then close.

        Rotations, deletions and OCR selection on the original file travel as
        metadata. Added, reordered or flipped pages need a merged PDF, written
        in a worker first.
        """
        document = self._document
        if self._save_in_place and self._pdf_path:
            self._publish(self._pdf_path, close_after=True)
            return
        if document is None or self._on_save_callback is None or self._saving:
            return
        original_page_count = self._original_page_count
        if original_page_count is None:
            original_page_count = document.total_pages
        if requires_materialization(document, document.path, original_page_count):
            self._apply_merged(document)
            return
        if self._deliver(document):
            document.clear_modifications()
            logger.info("Editor changes saved as metadata (no intermediate file)")
            self._close_window()

    def _apply_merged(self, document: PDFDocument) -> None:
        """Write the merged PDF in a worker, then hand it to the caller and close."""
        from bigocrpdf.ui.pdf_editor.page_operations import write_changes_to_pdf
        from bigocrpdf.utils.temp_manager import mkstemp, remove_file

        # Tracked temp file: the caller owns it once delivered, else it goes now.
        fd, temp_path = mkstemp(suffix=".pdf", prefix="bigocr_merge_")
        os.close(fd)
        snapshot = document.snapshot()
        page_count = sum(page.included_for_ocr for page in snapshot.pages)

        def write() -> None:
            try:
                write_changes_to_pdf(snapshot, temp_path)
            except BaseException:
                remove_file(temp_path)
                raise

        def written() -> None:
            merged = PDFDocument(path=temp_path, total_pages=page_count)
            if not self._deliver(merged):
                remove_file(temp_path)
                return
            document.clear_modifications()
            logger.info(f"Merged PDF saved to {temp_path}")
            self._close_window()

        self._run_save(os.path.basename(document.path), write, written)

    def _deliver(self, document: PDFDocument) -> bool:
        """Pass the edited document to the caller; report a refusal."""
        try:
            accepted = self._on_save_callback is not None and (
                self._on_save_callback(document) is not False
            )
        except Exception:
            logger.exception("Error saving editor changes")
            accepted = False
        if not accepted:
            self._show_error(_("Error saving changes."))
        return accepted

    def _maybe_save_and_close(self) -> None:
        """Check for unsaved changes and close."""
        if self._document and self._document.modified:
            dialog = Adw.AlertDialog()
            dialog.set_heading(_("Unsaved Changes"))
            dialog.set_body(_("What would you like to do with your changes?"))

            dialog.add_response("discard", _("Discard"))
            dialog.add_response("save", _("Save As") if self._standalone else _("Apply"))
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
            if self._standalone:
                self._save_as(close_after=True)
            else:
                self._apply_and_close()
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
        if self._saving:
            self._close_after_save = True
            return True
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
