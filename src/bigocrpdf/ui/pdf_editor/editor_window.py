"""
BigOcrPdf - PDF Editor Window

Main editor window for PDF page manipulation before OCR processing.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.tooltip_helper import get_tooltip_helper

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow


class PDFEditorWindow(Adw.Window):
    """Main PDF editor window.

    Provides a PDFArranger-style interface for:
    - Viewing page thumbnails in a grid
    - Selecting pages for OCR
    - Rotating pages
    - Deleting pages
    - Zooming thumbnails

    Attributes:
        document: The PDFDocument being edited
        on_save_callback: Callback when saving changes
    """

    def __init__(
        self,
        application: Gtk.Application,
        pdf_path: str,
        on_save_callback: Callable[[PDFDocument], None] | None = None,
        on_close_callback: Callable[[], None] | None = None,
        parent_window: "BigOcrPdfWindow | None" = None,
        initial_state: dict | None = None,
    ) -> None:
        """Initialize the PDF editor window.

        Args:
            application: The Gtk Application instance
            pdf_path: Path to the PDF file to edit
            on_save_callback: Callback when user saves changes
            on_close_callback: Callback when window is closed
            parent_window: Optional parent window reference
            initial_state: Optional dictionary to restore page states
        """
        super().__init__(application=application)

        self._pdf_path = pdf_path
        self._on_save_callback = on_save_callback
        self._on_close_callback = on_close_callback
        self._parent_window = parent_window
        self._initial_state = initial_state
        self._document: PDFDocument | None = None

        # Window configuration
        self.set_title(_("PDF Editor - {}").format(os.path.basename(pdf_path)))
        self.set_default_size(900, 700)
        self.set_modal(False)

        if parent_window:
            self.set_transient_for(parent_window)

        self._setup_actions()
        self._setup_ui()
        self._setup_keyboard_shortcuts()
        self._setup_drag_drop()
        self._load_document()

        # Connect close request handler
        self.connect("close-request", self._on_close_request)

    def _setup_ui(self) -> None:
        """Set up the window UI."""
        # Main layout with toolbar view
        self._toolbar_view = Adw.ToolbarView()
        self.set_content(self._toolbar_view)

        # Header bar
        self._header_bar = Adw.HeaderBar()
        self._header_bar.set_show_end_title_buttons(True)
        self._header_bar.set_show_start_title_buttons(True)

        # Title Widget
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label=os.path.basename(self._pdf_path))
        title_label.add_css_class("title")
        title_box.append(title_label)

        self._header_bar.set_title_widget(title_box)

        # --- Header Bar Controls (Only Back and Main Actions) ---

        # Back Button
        self._back_btn = Gtk.Button()
        self._back_btn.set_icon_name("go-previous-symbolic")
        get_tooltip_helper().add_tooltip(self._back_btn, _("Go back to the file list"))
        self._back_btn.connect("clicked", self._on_back_clicked)
        self._header_bar.pack_start(self._back_btn)

        # OK Button (Primary)
        self._ok_button = Gtk.Button(label=_("OK"))
        self._ok_button.add_css_class("suggested-action")
        get_tooltip_helper().add_tooltip(self._ok_button, _("Save changes and go back"))
        self._ok_button.connect("clicked", self._on_ok_clicked)
        self._header_bar.pack_end(self._ok_button)

        # Add Files button (simple button — no folder option)
        self._add_button = Gtk.Button(label=_("Add Files"))
        self._add_button.add_css_class("suggested-action")
        get_tooltip_helper().add_tooltip(self._add_button, _("Add more files to this document"))
        self._add_button.connect("clicked", self._on_add_files_clicked)

        self._header_bar.pack_end(self._add_button)

        self._toolbar_view.add_top_bar(self._header_bar)

        # Main content: Page grid
        self._page_grid = PageGrid()
        self._page_grid.connect("selection-changed", self._on_selection_changed)
        self._page_grid.connect("page-ocr-toggled", self._on_page_ocr_toggled)
        self._toolbar_view.set_content(self._page_grid)

        # Status bar / Footer (Contains all other controls now)
        self._status_bar = self._create_status_bar()
        self._toolbar_view.add_bottom_bar(self._status_bar)

    def _create_status_bar(self) -> Gtk.Box:
        """Create the status bar with tools.

        Returns:
            Status bar widget
        """
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_box.add_css_class("toolbar")
        status_box.add_css_class("editor-status-bar")
        status_box.set_margin_start(12)
        status_box.set_margin_end(12)
        status_box.set_margin_top(6)
        status_box.set_margin_bottom(6)

        # --- Left Side: Stats ---

        # Page counts
        self._pages_label = Gtk.Label()
        self._pages_label.add_css_class("dim-label")
        status_box.append(self._pages_label)

        # Separator
        sep1 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        status_box.append(sep1)

        # Included count
        self._selected_label = Gtk.Label()
        self._selected_label.add_css_class("dim-label")
        status_box.append(self._selected_label)

        sep2 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        status_box.append(sep2)

        # --- Center Left: Selection Tools ---

        self._select_all_btn = Gtk.Button(label=_("Include All"))
        self._select_all_btn.add_css_class("flat")
        get_tooltip_helper().add_tooltip(self._select_all_btn, _("Select all pages for processing"))
        self._select_all_btn.connect("clicked", self._on_select_all)
        status_box.append(self._select_all_btn)

        self._deselect_all_btn = Gtk.Button(label=_("Exclude All"))
        self._deselect_all_btn.add_css_class("flat")
        get_tooltip_helper().add_tooltip(self._deselect_all_btn, _("Deselect all pages"))
        self._deselect_all_btn.connect("clicked", self._on_deselect_all)
        status_box.append(self._deselect_all_btn)

        # Separator
        status_box.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # --- Center Right: Rotation Tools ---

        self._rotate_left_btn = Gtk.Button()
        self._rotate_left_btn.set_icon_name("object-rotate-left-symbolic")
        self._rotate_left_btn.add_css_class("flat")
        get_tooltip_helper().add_tooltip(
            self._rotate_left_btn, _("Rotate selected pages to the left")
        )
        self._rotate_left_btn.connect("clicked", self._on_rotate_left)
        status_box.append(self._rotate_left_btn)

        self._rotate_right_btn = Gtk.Button()
        self._rotate_right_btn.set_icon_name("object-rotate-right-symbolic")
        self._rotate_right_btn.add_css_class("flat")
        get_tooltip_helper().add_tooltip(
            self._rotate_right_btn, _("Rotate selected pages to the right")
        )
        self._rotate_right_btn.connect("clicked", self._on_rotate_right)
        status_box.append(self._rotate_right_btn)

        # Expanding spacer to push Zoom to right
        filler = Gtk.Box()
        filler.set_hexpand(True)
        status_box.append(filler)

        # --- Right Side: Zoom ---
        zoom_levels = Gtk.StringList.new(["50%", "75%", "100%", "150%", "200%"])
        self._zoom_dropdown = Gtk.DropDown(model=zoom_levels)
        self._zoom_dropdown.set_selected(2)  # Default to 100%
        get_tooltip_helper().add_tooltip(self._zoom_dropdown, _("Change the size of page previews"))
        self._zoom_dropdown.connect("notify::selected", self._on_zoom_dropdown_changed)
        status_box.append(self._zoom_dropdown)

        return status_box

    def _setup_actions(self) -> None:
        """Set up window actions."""
        # Create action group and insert it
        action_group = Gio.SimpleActionGroup()
        self.insert_action_group("win", action_group)

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Create event controller for key presses
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _setup_drag_drop(self) -> None:
        """Set up drag and drop for external files (PDFs and images)."""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList])
        drop_target.connect("drop", self._on_external_file_drop)
        self._toolbar_view.add_controller(drop_target)

    def _on_external_file_drop(
        self, _target: Gtk.DropTarget, value: Gdk.FileList, _x: float, _y: float
    ) -> bool:
        """Handle external file drop onto the editor.

        Accepts PDFs and image files, adding their pages to the document.

        Args:
            _target: The drop target
            value: The dropped file list
            _x: X coordinate
            _y: Y coordinate

        Returns:
            True if the drop was handled
        """
        supported_extensions = (
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".webp",
            ".avif",
        )

        try:
            # Extract file paths from drop
            file_paths: list[str] = []
            if isinstance(value, Gio.File):
                path = value.get_path()
                if path:
                    file_paths.append(path)
            elif hasattr(value, "get_files"):
                for f in value.get_files():
                    path = f.get_path()
                    if path:
                        file_paths.append(path)
            elif hasattr(value, "__iter__"):
                for f in value:
                    if isinstance(f, Gio.File):
                        path = f.get_path()
                        if path:
                            file_paths.append(path)

            # Filter for supported files
            valid_paths = [
                p
                for p in file_paths
                if os.path.exists(p) and p.lower().endswith(supported_extensions)
            ]

            if not valid_paths:
                return False

            self._add_files_to_document(valid_paths)
            return True

        except Exception as e:
            logger.error(f"Error handling dropped files: {e}")
            return False

    def _add_files_to_document(self, file_paths: list[str]) -> None:
        """Add external files (PDFs or images) to the current document.

        Args:
            file_paths: List of file paths to add
        """
        if not self._document:
            return

        added_count = 0
        current_total = self._document.total_pages
        renderer = get_thumbnail_renderer()

        for path in file_paths:
            try:
                page_count = renderer.get_page_count(path)
                if page_count > 0:
                    for i in range(page_count):
                        new_page = PageState(
                            page_number=i + 1,
                            position=current_total + added_count + i,
                            source_file=path,
                        )
                        self._document.pages.append(new_page)
                    added_count += page_count
                    logger.info(f"Added {page_count} pages from: {path}")
            except Exception as e:
                logger.error(f"Failed to add file {path}: {e}")
                self._show_error(
                    _("Failed to add file {}: {}").format(os.path.basename(path), str(e))
                )

        if added_count > 0:
            self._document.total_pages += added_count
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            logger.info(
                f"Added {added_count} pages via drag-and-drop. Total: {self._document.total_pages}"
            )

    def _on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Handle keyboard shortcuts.

        Args:
            controller: The event controller
            keyval: The key value
            keycode: The key code
            state: Modifier state

        Returns:
            True if the event was handled
        """
        ctrl = state & Gdk.ModifierType.CONTROL_MASK

        if keyval == Gdk.KEY_l and ctrl:
            # Ctrl+L: Rotate left
            self._on_rotate_left(None)
            return True
        elif keyval == Gdk.KEY_r and ctrl:
            # Ctrl+R: Rotate right
            self._on_rotate_right(None)
            return True
        elif keyval == Gdk.KEY_Escape:
            # Escape: Close window
            self.close()
            return True

        return False

    def _load_document(self) -> None:
        """Load the PDF document."""
        try:
            renderer = get_thumbnail_renderer()
            page_count = renderer.get_page_count(self._pdf_path)

            if page_count == 0:
                self._show_error(_("Could not open PDF file or file has no pages."))
                return

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
        ocr_count = self._page_grid.get_ocr_count()

        self._pages_label.set_text(_("Pages: {}").format(total))
        self._selected_label.set_text(_("Included: {}").format(ocr_count))

    def _on_selection_changed(self, grid: PageGrid) -> None:
        """Handle selection changes in the grid.

        Args:
            grid: The page grid
        """
        self._update_status_bar()

    def _on_page_ocr_toggled(self, grid: PageGrid, page_num: int, active: bool) -> None:
        """Handle OCR toggle for a page.

        Args:
            grid: The page grid
            page_num: Page number
            active: New OCR state
        """
        logger.debug(f"Page {page_num} OCR toggled to {active}")

    def _on_back_clicked(self, _button: Gtk.Button) -> None:
        """Handle back button click.

        Args:
            _button: The button widget
        """
        # User requested discard/cancel, so close without saving
        if self._document:
            self._document.clear_modifications()
        self._close_window()

    def _on_add_files_clicked(self, _button: Gtk.Button) -> None:
        """Handle Add Files button click.

        Args:
            _button: The button widget
        """
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Add Files"))

        # Set up file filter for PDFs and Images
        filter_any = Gtk.FileFilter()
        filter_any.set_name(_("PDFs and Images"))
        filter_any.add_mime_type("application/pdf")
        filter_any.add_pattern("*.pdf")

        # Add image patterns
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff", "*.bmp"]:
            filter_any.add_pattern(pattern)
            filter_any.add_pattern(pattern.upper())

        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(filter_any)
        dialog.set_filters(store)

        dialog.open_multiple(self, None, self._on_pdfs_selected)

    def _on_pdfs_selected(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle PDF file selection result.

        Args:
            dialog: The file dialog
            result: The async result
        """
        try:
            files = dialog.open_multiple_finish(result)
            if files:
                file_paths = [f.get_path() for f in files if f.get_path()]
                if file_paths:
                    self._add_files_to_document(file_paths)
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error(f"Error selecting files: {e}")

    def _on_zoom_combo_changed(self, combo: Gtk.ComboBoxText) -> None:
        """Handle zoom combo change (legacy — kept for compatibility).

        Args:
            combo: The combo box
        """
        text = combo.get_active_text()
        if not text:
            return

        try:
            zoom_level = int(text.rstrip("%"))
            self._page_grid.set_zoom_level(zoom_level)
        except ValueError:
            logger.error(f"Invalid zoom level text: {text}")

    def _on_zoom_dropdown_changed(self, dropdown: Gtk.DropDown, _param) -> None:
        """Handle zoom dropdown selection change.

        Args:
            dropdown: The dropdown widget
            _param: GObject param spec (unused)
        """
        zoom_levels = [50, 75, 100, 150, 200]
        selected = dropdown.get_selected()
        if 0 <= selected < len(zoom_levels):
            self._page_grid.set_zoom_level(zoom_levels[selected])

    def _on_ok_clicked(self, _button: Gtk.Button) -> None:
        """Handle OK button click - apply changes and close.

        Args:
            _button: The button widget
        """
        self._save_and_callback()
        self._close_window()

    def _save_and_callback(self) -> None:
        """Save editor changes and trigger callback.

        If only rotations/deletions on the original file, saves state
        metadata (no intermediate file). If pages from other files were
        added, creates a merged PDF.
        """
        if not self._document or not self._on_save_callback:
            return

        try:
            original_path = self._document.path

            # Check if merge is needed (pages from multiple source files)
            source_files = {p.source_file for p in self._document.pages if not p.deleted}
            needs_merge = len(source_files) > 1 or (
                source_files and original_path not in source_files
            )

            if needs_merge:
                self._save_merged_pdf(original_path)
            else:
                # No merge needed — just pass modifications as state
                self._on_save_callback(self._document)
                self._document.clear_modifications()
                logger.info("Editor changes saved as metadata (no intermediate file)")

        except Exception as e:
            logger.error(f"Error saving editor changes: {e}")
            self._show_error(_("Error saving changes."))

    def _save_merged_pdf(self, original_path: str) -> None:
        """Create a merged PDF when pages from multiple sources are present."""
        import tempfile

        # Use /tmp for merge temp files — cleaned up after OCR, and
        # automatically removed on reboot if cleanup fails
        fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_merge_")
        os.close(fd)

        logger.info("Merging pages from multiple sources into new PDF...")
        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        if apply_changes_to_pdf(self._document, temp_path):
            active_count = len(self._document.get_active_pages())
            clean_doc = PDFDocument(path=temp_path, total_pages=active_count)
            for i in range(active_count):
                clean_doc.pages[i].source_file = temp_path
                clean_doc.pages[i].page_number = i + 1

            self._on_save_callback(clean_doc)
            self._document.clear_modifications()
            logger.info(f"Merged PDF saved to {temp_path}")
        else:
            # Clean up failed temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self._show_error(_("Failed to merge PDF pages."))

    def _on_select_all(self, _button: Gtk.Button) -> None:
        """Handle Select All button click - selects all pages for OCR.

        Args:
            _button: The button widget
        """
        self._page_grid.select_all_for_ocr()

    def _on_deselect_all(self, _button: Gtk.Button) -> None:
        """Handle Deselect All button click - deselects all pages from OCR.

        Args:
            _button: The button widget
        """
        self._page_grid.deselect_all_for_ocr()

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
            self._save_and_callback()
            self._close_window()
        elif response == "discard":
            if self._document:
                self._document.clear_modifications()
            self._close_window()
        # "cancel" does nothing

    def _close_window(self) -> None:
        """Close the window."""
        if self._on_close_callback:
            self._on_close_callback()
        self.close()

    def _on_close_request(self, window: Adw.Window) -> bool:
        """Handle window close request.

        Args:
            window: The window

        Returns:
            True to prevent default close, False to allow
        """
        if self._document and self._document.modified:
            self._maybe_save_and_close()
            return True  # Prevent default close
        return False  # Allow close

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

    def _on_rotate_left(self, _source: GObject.Object | None) -> None:
        """Handle rotate left action.

        Rotates selected pages. If no pages are selected, rotates
        all included (non-excluded) pages.

        Args:
            _source: The source widget (unused)
        """
        if not self._document:
            return

        self._rotate_selected_pages(-90)

    def _on_rotate_right(self, _source: GObject.Object | None) -> None:
        """Handle rotate right action.

        Rotates selected pages. If no pages are selected, rotates
        all included (non-excluded) pages.

        Args:
            _source: The source widget (unused)
        """
        if not self._document:
            return

        self._rotate_selected_pages(90)

    def _rotate_selected_pages(self, degrees: int) -> None:
        """Rotate all included (non-excluded) pages by degrees.

        Args:
            degrees: Rotation angle (90 or -90)
        """
        thumbnails = self._page_grid._thumbnails
        rotated = 0
        for thumb in thumbnails:
            if not thumb.page_state.deleted:
                thumb.page_state.rotate(degrees)
                rotated += 1

        if rotated > 0:
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            logger.info(f"Rotated {rotated} included page(s) by {degrees}°")
