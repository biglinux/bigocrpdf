"""
BigOcrPdf - PDF Editor Window

Main editor window for PDF page manipulation before OCR processing.
Redesigned with a visible action bar for discoverability and accessibility.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.a11y import set_a11y_label
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
        self._undo_stack: list[list[dict]] = []
        self._notification_timer_id: int | None = None

        # Window configuration
        self.set_title(_("PDF Editor - {}").format(os.path.basename(pdf_path)))
        self.set_default_size(900, 700)
        self.set_modal(False)

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

        # --- Header Bar (all tools integrated per GTK4/GNOME HIG) ---
        self._header_bar = Adw.HeaderBar()
        self._header_bar.set_show_end_title_buttons(True)
        self._header_bar.set_show_start_title_buttons(True)

        # Detect window button side (same logic as main window)
        buttons_left = self._window_buttons_on_left()
        if buttons_left:
            self._header_bar.set_decoration_layout("close,maximize,minimize:")
        else:
            self._header_bar.set_decoration_layout("menu:minimize,maximize,close")

        # Add Files (start)
        self._add_button = Gtk.Button()
        add_content = Adw.ButtonContent(icon_name="list-add-symbolic", label=_("Add"))
        self._add_button.set_child(add_content)
        get_tooltip_helper().add_tooltip(
            self._add_button, _("Add PDF or image files to this document")
        )
        set_a11y_label(self._add_button, _("Add PDF or image files"))
        self._add_button.connect("clicked", self._on_add_files_clicked)
        self._header_bar.pack_start(self._add_button)

        # Rotate Left (start)
        self._rotate_left_btn = Gtk.Button()
        self._rotate_left_btn.set_icon_name("object-rotate-left-symbolic")
        get_tooltip_helper().add_tooltip(
            self._rotate_left_btn, _("Rotate selected pages left (Ctrl+L)")
        )
        set_a11y_label(self._rotate_left_btn, _("Rotate selected pages left"))
        self._rotate_left_btn.connect("clicked", self._on_rotate_left)
        self._header_bar.pack_start(self._rotate_left_btn)

        # Rotate Right (start)
        self._rotate_right_btn = Gtk.Button()
        self._rotate_right_btn.set_icon_name("object-rotate-right-symbolic")
        get_tooltip_helper().add_tooltip(
            self._rotate_right_btn, _("Rotate selected pages right (Ctrl+R)")
        )
        set_a11y_label(self._rotate_right_btn, _("Rotate selected pages right"))
        self._rotate_right_btn.connect("clicked", self._on_rotate_right)
        self._header_bar.pack_start(self._rotate_right_btn)

        # Undo (start)
        self._undo_btn = Gtk.Button()
        self._undo_btn.set_icon_name("edit-undo-symbolic")
        self._undo_btn.set_sensitive(False)
        get_tooltip_helper().add_tooltip(self._undo_btn, _("Undo last change (Ctrl+Z)"))
        set_a11y_label(self._undo_btn, _("Undo last change"))
        self._undo_btn.connect("clicked", lambda _b: self._undo())
        self._header_bar.pack_start(self._undo_btn)

        # Include All (start)
        self._select_all_btn = Gtk.Button()
        self._select_all_btn.set_icon_name("edit-select-all-symbolic")
        get_tooltip_helper().add_tooltip(
            self._select_all_btn, _("Include all pages in the final document")
        )
        set_a11y_label(self._select_all_btn, _("Include all pages"))
        self._select_all_btn.connect("clicked", self._on_select_all)
        self._header_bar.pack_start(self._select_all_btn)

        # Exclude All (start)
        self._deselect_all_btn = Gtk.Button()
        self._deselect_all_btn.set_icon_name("edit-clear-all-symbolic")
        get_tooltip_helper().add_tooltip(
            self._deselect_all_btn, _("Exclude all pages from the final document")
        )
        set_a11y_label(self._deselect_all_btn, _("Exclude all pages"))
        self._deselect_all_btn.connect("clicked", self._on_deselect_all)
        self._header_bar.pack_start(self._deselect_all_btn)

        # Cancel + Apply centered in the title area
        title_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        title_actions.set_halign(Gtk.Align.CENTER)

        self._cancel_button = Gtk.Button(label=_("Cancel"))
        get_tooltip_helper().add_tooltip(self._cancel_button, _("Discard changes and close"))
        set_a11y_label(self._cancel_button, _("Cancel and close"))
        self._cancel_button.connect("clicked", self._on_back_clicked)
        title_actions.append(self._cancel_button)

        self._apply_button = Gtk.Button(label=_("Apply"))
        self._apply_button.add_css_class("suggested-action")
        get_tooltip_helper().add_tooltip(self._apply_button, _("Save changes and go back (Ctrl+S)"))
        set_a11y_label(self._apply_button, _("Save changes and go back"))
        self._apply_button.connect("clicked", self._on_ok_clicked)
        title_actions.append(self._apply_button)

        self._header_bar.set_title_widget(title_actions)

        # Tools packed from end (first = rightmost after window controls)
        self._reverse_btn = Gtk.Button()
        self._reverse_btn.set_icon_name("view-sort-descending-symbolic")
        get_tooltip_helper().add_tooltip(self._reverse_btn, _("Reverse the page order"))
        set_a11y_label(self._reverse_btn, _("Reverse page order"))
        self._reverse_btn.set_action_name("editor.reverse")
        self._header_bar.pack_end(self._reverse_btn)

        split_menu = Gio.Menu()
        split_menu.append(_("Split by page count…"), "editor.split-pages")
        split_menu.append(_("Split by file size…"), "editor.split-size")
        self._split_btn = Gtk.MenuButton()
        self._split_btn.set_icon_name("edit-cut-symbolic")
        self._split_btn.set_menu_model(split_menu)
        get_tooltip_helper().add_tooltip(self._split_btn, _("Split the document into parts"))
        set_a11y_label(self._split_btn, _("Split document"))
        self._header_bar.pack_end(self._split_btn)

        self._compress_btn = Gtk.Button()
        self._compress_btn.set_icon_name("image-resize-symbolic")
        get_tooltip_helper().add_tooltip(
            self._compress_btn, _("Compress the PDF to reduce file size")
        )
        set_a11y_label(self._compress_btn, _("Compress PDF"))
        self._compress_btn.set_action_name("editor.compress")
        self._header_bar.pack_end(self._compress_btn)

        self._save_copy_btn = Gtk.Button()
        self._save_copy_btn.set_icon_name("document-save-as-symbolic")
        get_tooltip_helper().add_tooltip(self._save_copy_btn, _("Save a copy of the document"))
        set_a11y_label(self._save_copy_btn, _("Save a copy"))
        self._save_copy_btn.set_action_name("editor.save-copy")
        self._header_bar.pack_end(self._save_copy_btn)

        self._toolbar_view.add_top_bar(self._header_bar)

        # --- Content area: Overlay with PageGrid + Notification Banner ---
        content_overlay = Gtk.Overlay()

        self._page_grid = PageGrid()
        self._page_grid.on_before_mutate = self._push_undo
        self._page_grid.connect("selection-changed", self._on_selection_changed)
        self._page_grid.connect("page-ocr-toggled", self._on_page_ocr_toggled)
        content_overlay.set_child(self._page_grid)

        # Notification banner
        self._notification_revealer = Gtk.Revealer()
        self._notification_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._notification_revealer.set_transition_duration(200)
        self._notification_revealer.set_reveal_child(False)
        self._notification_revealer.set_valign(Gtk.Align.START)
        self._notification_revealer.set_halign(Gtk.Align.CENTER)
        self._notification_revealer.set_can_target(False)

        self._notification_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._notification_box.add_css_class("editor-notification")
        self._notification_box.set_margin_top(8)

        self._notification_icon = Gtk.Image()
        self._notification_icon.set_icon_size(Gtk.IconSize.NORMAL)
        self._notification_box.append(self._notification_icon)

        self._notification_label = Gtk.Label()
        self._notification_label.set_wrap(True)
        self._notification_label.set_max_width_chars(60)
        self._notification_box.append(self._notification_label)

        self._notification_revealer.set_child(self._notification_box)
        content_overlay.add_overlay(self._notification_revealer)

        self._toolbar_view.set_content(content_overlay)

        # --- Status Bar (bottom) ---
        self._status_bar = self._create_status_bar()
        self._toolbar_view.add_bottom_bar(self._status_bar)

    def _create_status_bar(self) -> Gtk.Box:
        """Create the status bar with filename, page counts, and zoom.

        Returns:
            Status bar widget.
        """
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_box.add_css_class("editor-status-bar")
        status_box.set_margin_start(12)
        status_box.set_margin_end(12)
        status_box.set_margin_top(4)
        status_box.set_margin_bottom(4)

        # Filename label (start)
        self._filename_label = Gtk.Label(label=os.path.basename(self._pdf_path))
        self._filename_label.add_css_class("dim-label")
        self._filename_label.set_halign(Gtk.Align.START)
        self._filename_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        set_a11y_label(self._filename_label, _("Current file"))
        status_box.append(self._filename_label)

        # Spacer
        filler = Gtk.Box()
        filler.set_hexpand(True)
        status_box.append(filler)

        # Page/included counts
        self._status_label = Gtk.Label()
        self._status_label.add_css_class("dim-label")
        set_a11y_label(self._status_label, _("Document status"))
        status_box.append(self._status_label)

        # Selection info
        self._selection_label = Gtk.Label()
        self._selection_label.add_css_class("dim-label")
        self._selection_label.set_halign(Gtk.Align.END)
        status_box.append(self._selection_label)

        # Zoom dropdown (end)
        zoom_levels = Gtk.StringList.new(["50%", "75%", "100%", "150%", "200%", "300%", "400%"])
        self._zoom_dropdown = Gtk.DropDown(model=zoom_levels)
        self._zoom_dropdown.set_selected(2)  # Default 100%
        get_tooltip_helper().add_tooltip(self._zoom_dropdown, _("Change the size of page previews"))
        set_a11y_label(self._zoom_dropdown, _("Zoom level"))
        self._zoom_dropdown.connect("notify::selected", self._on_zoom_dropdown_changed)
        status_box.append(self._zoom_dropdown)

        return status_box

    def _setup_actions(self) -> None:
        """Set up window actions for the overflow menu."""
        action_group = Gio.SimpleActionGroup()

        actions = {
            "save-copy": self._on_save_copy,
            "compress": self._on_tools_compress,
            "split-pages": self._on_tools_split_pages,
            "split-size": self._on_tools_split_size,
            "reverse": self._on_tools_reverse,
        }

        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            action_group.add_action(action)

        self.insert_action_group("editor", action_group)

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

        self._push_undo()
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

    # -- Undo stack ---------------------------------------------------------

    _MAX_UNDO = 50

    def _push_undo(self) -> None:
        """Snapshot current page state before a mutating operation."""
        if not self._document:
            return
        snapshot = [p.to_dict() for p in self._document.pages]
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

    # -- Keyboard handling -------------------------------------------------

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
        elif keyval == Gdk.KEY_z and ctrl:
            # Ctrl+Z: Undo
            self._undo()
            return True
        elif keyval == Gdk.KEY_a and ctrl:
            # Ctrl+A: Select all
            self._page_grid.select_all()
            return True
        elif keyval == Gdk.KEY_Up and ctrl:
            # Ctrl+Up: Move selected pages up (accessible reorder)
            self._move_selected_pages(-1)
            return True
        elif keyval == Gdk.KEY_Down and ctrl:
            # Ctrl+Down: Move selected pages down (accessible reorder)
            self._move_selected_pages(1)
            return True
        elif keyval == Gdk.KEY_s and ctrl:
            # Ctrl+S: Save (OK)
            self._on_ok_clicked(None)
            return True
        elif keyval == Gdk.KEY_Delete:
            # Delete: Toggle exclude on selected pages
            if self._page_grid._selected_indices:
                self._push_undo()
                self._page_grid.toggle_ocr_for_selected()
            return True
        elif keyval == Gdk.KEY_Escape:
            # Escape: Close window
            self.close()
            return True
        elif keyval in (Gdk.KEY_Page_Up, Gdk.KEY_Page_Down):
            # Page Up/Down: Scroll the page grid
            vadj = self._page_grid.get_vadjustment()
            step = vadj.get_page_size() * 0.8
            if keyval == Gdk.KEY_Page_Up:
                vadj.set_value(max(vadj.get_lower(), vadj.get_value() - step))
            else:
                vadj.set_value(
                    min(vadj.get_upper() - vadj.get_page_size(), vadj.get_value() + step)
                )
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

            # Kick off fast batch thumbnail preload (pdftoppm) in background
            renderer.batch_preload(self._pdf_path, page_count)

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
        selected_count = len(self._page_grid._selected_indices)

        self._status_label.set_text(
            _("{total} pages · {included} included").format(total=total, included=included)
        )

        if selected_count > 0:
            self._selection_label.set_text(_("{count} selected").format(count=selected_count))
            self._selection_label.set_visible(True)
        else:
            self._selection_label.set_visible(False)

        # Update undo button sensitivity
        self._undo_btn.set_sensitive(len(self._undo_stack) > 0)

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

    # ------------------------------------------------------------------
    # Overflow menu (advanced features)
    # ------------------------------------------------------------------

    def _on_save_copy(self, _action, _param) -> None:
        """Save included pages as a new PDF file.

        Unifies the old Save As and Extract Selected Pages into one action:
        saves only the included (non-excluded) pages to a chosen location.
        """
        if not self._document:
            return

        dialog = Gtk.FileDialog()
        dialog.set_title(_("Save a copy"))
        dialog.set_initial_name(os.path.basename(self._pdf_path))

        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_pattern("*.pdf")
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(pdf_filter)
        dialog.set_filters(store)

        dialog.save(self, None, self._on_save_copy_response)

    def _on_save_copy_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle Save Copy dialog response."""
        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                if path:
                    from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

                    if apply_changes_to_pdf(self._document, path):
                        logger.info("Saved PDF copy to %s", path)
                        self._show_info(_("Saved to {}").format(os.path.basename(path)))
                    else:
                        self._show_error(_("Failed to save PDF."))
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error("Save copy error: %s", e)

    def _on_tools_compress(self, _action, _param) -> None:
        """Show compress dialog and compress the document."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Compress PDF"))
        dialog.set_body(
            _(
                "Reduce the file size by compressing the images inside the PDF. "
                "Lower values produce smaller files but with less image detail."
            )
        )

        # Quality spin button with description
        quality_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        quality_box.set_halign(Gtk.Align.CENTER)

        quality_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        quality_row.set_halign(Gtk.Align.CENTER)
        quality_label = Gtk.Label(label=_("Image Quality:"))
        quality_spin = Gtk.SpinButton.new_with_range(10, 95, 5)
        quality_spin.set_value(60)
        quality_row.append(quality_label)
        quality_row.append(quality_spin)
        quality_box.append(quality_row)

        quality_hint = Gtk.Label(
            label=_("10 = smallest file, 95 = best quality. 60 is a good default.")
        )
        quality_hint.add_css_class("dim-label")
        quality_hint.add_css_class("caption")
        quality_box.append(quality_hint)

        # DPI spin button with description
        dpi_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        dpi_box.set_halign(Gtk.Align.CENTER)

        dpi_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dpi_row.set_halign(Gtk.Align.CENTER)
        dpi_label = Gtk.Label(label=_("Image Resolution (DPI):"))
        dpi_spin = Gtk.SpinButton.new_with_range(72, 600, 10)
        dpi_spin.set_value(150)
        dpi_row.append(dpi_label)
        dpi_row.append(dpi_spin)
        dpi_box.append(dpi_row)

        dpi_hint = Gtk.Label(
            label=_("72 = screen only, 150 = digital reading, 300 = print quality.")
        )
        dpi_hint.add_css_class("dim-label")
        dpi_hint.add_css_class("caption")
        dpi_box.append(dpi_hint)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.append(quality_box)
        content_box.append(dpi_box)
        dialog.set_extra_child(content_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("compress", _("Compress"))
        dialog.set_response_appearance("compress", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: (
                self._do_compress(int(quality_spin.get_value()), int(dpi_spin.get_value()))
                if r == "compress"
                else None
            ),
        )
        dialog.present(self)

    def _do_compress(self, quality: int, dpi: int) -> None:
        """Execute PDF compression."""
        if not self._document:
            return

        import tempfile

        from bigocrpdf.services.pdf_operations import compress_pdf

        # First, apply current edits into a temp file
        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        fd, tmp_edited = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_edit_")
        os.close(fd)

        if not apply_changes_to_pdf(self._document, tmp_edited):
            self._show_error(_("Failed to prepare document for compression."))
            os.unlink(tmp_edited)
            return

        # Now compress
        fd2, tmp_compressed = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_cmp_")
        os.close(fd2)

        result = compress_pdf(tmp_edited, tmp_compressed, image_quality=quality, image_dpi=dpi)
        os.unlink(tmp_edited)

        if result.success:
            # Prompt save location
            dialog = Gtk.FileDialog()
            dialog.set_title(_("Save Compressed PDF"))
            dialog.set_initial_name("compressed_" + os.path.basename(self._pdf_path))

            pdf_filter = Gtk.FileFilter()
            pdf_filter.set_name(_("PDF Files"))
            pdf_filter.add_pattern("*.pdf")
            store = Gio.ListStore.new(Gtk.FileFilter)
            store.append(pdf_filter)
            dialog.set_filters(store)

            dialog.save(
                self,
                None,
                lambda d, r: self._finish_compress_save(d, r, tmp_compressed, result.message),
            )
        else:
            os.unlink(tmp_compressed)
            self._show_error(_("Compression failed: {}").format(result.message))

    def _finish_compress_save(self, dialog, result, tmp_path, message) -> None:
        """Finish saving the compressed file."""
        import shutil

        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                if path:
                    shutil.move(tmp_path, path)
                    self._show_info(message)
                    return
        except GLib.Error:
            pass

        # Cleanup on failure/cancel
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def _on_tools_split_pages(self, _action, _param) -> None:
        """Show split-by-pages dialog."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Split by Page Count"))
        dialog.set_body(_("Split the document into parts with a fixed number of pages each."))

        spin_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        spin_box.set_halign(Gtk.Align.CENTER)
        spin_label = Gtk.Label(label=_("Pages per file:"))
        spin = Gtk.SpinButton.new_with_range(1, 9999, 1)
        spin.set_value(5)
        spin_box.append(spin_label)
        spin_box.append(spin)
        dialog.set_extra_child(spin_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("split", _("Split"))
        dialog.set_response_appearance("split", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: self._do_split_by_pages(int(spin.get_value())) if r == "split" else None,
        )
        dialog.present(self)

    def _do_split_by_pages(self, pages_per_file: int) -> None:
        """Execute split by page count."""
        self._pick_output_dir_and_split("pages", pages_per_file)

    def _on_tools_split_size(self, _action, _param) -> None:
        """Show split-by-size dialog."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Split by File Size"))
        dialog.set_body(_("Split the document so each part is at most the specified size."))

        spin_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        spin_box.set_halign(Gtk.Align.CENTER)
        spin_label = Gtk.Label(label=_("Max size (MB):"))
        spin = Gtk.SpinButton.new_with_range(1, 500, 1)
        spin.set_value(10)
        spin_box.append(spin_label)
        spin_box.append(spin)
        dialog.set_extra_child(spin_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("split", _("Split"))
        dialog.set_response_appearance("split", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: self._do_split_by_size(float(spin.get_value())) if r == "split" else None,
        )
        dialog.present(self)

    def _do_split_by_size(self, max_mb: float) -> None:
        """Execute split by file size."""
        self._pick_output_dir_and_split("size", max_mb)

    def _pick_output_dir_and_split(self, mode: str, value: float) -> None:
        """Let user pick an output directory, then run the split."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Select Output Directory"))

        dialog.select_folder(
            self,
            None,
            lambda d, r: self._finish_split(d, r, mode, value),
        )

    def _finish_split(self, dialog, result, mode: str, value: float) -> None:
        """Finish the split operation after directory selection."""
        import tempfile

        try:
            folder = dialog.select_folder_finish(result)
            if not folder:
                return
            output_dir = folder.get_path()
            if not output_dir:
                return
        except GLib.Error:
            return

        # Apply current edits to a temp file first
        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_split_")
        os.close(fd)

        if not apply_changes_to_pdf(self._document, tmp_path):
            self._show_error(_("Failed to prepare document for splitting."))
            os.unlink(tmp_path)
            return

        from bigocrpdf.services.pdf_operations import split_by_pages, split_by_size

        original_stem = Path(self._document.path).stem
        try:
            if mode == "pages":
                result_split = split_by_pages(
                    tmp_path, output_dir, int(value), prefix=original_stem
                )
            else:
                result_split = split_by_size(tmp_path, output_dir, value, prefix=original_stem)

            self._show_info(
                _("Split into {} parts ({} pages)").format(
                    result_split.parts, result_split.total_pages
                )
            )
        except Exception as e:
            self._show_error(_("Split failed: {}").format(str(e)))
        finally:
            os.unlink(tmp_path)

    def _on_tools_reverse(self, _action, _param) -> None:
        """Reverse the page order."""
        if not self._document:
            return

        self._push_undo()
        active = self._document.get_active_pages()
        total = len(active)
        for i, page in enumerate(active):
            page.position = total - 1 - i

        self._document.mark_modified()
        self._page_grid.refresh()
        self._update_status_bar()
        logger.info("Reversed page order")

    def _show_info(self, message: str) -> None:
        """Show a non-blocking inline notification.

        Args:
            message: Info message
        """
        self._show_notification(message, "emblem-ok-symbolic")

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

    def _on_zoom_dropdown_changed(self, dropdown: Gtk.DropDown, _param) -> None:
        """Handle zoom dropdown selection change.

        Args:
            dropdown: The dropdown widget
            _param: GObject param spec (unused)
        """
        zoom_levels = [50, 75, 100, 150, 200, 300, 400]
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

        if self._document is not None and apply_changes_to_pdf(self._document, temp_path):
            active_count = len(self._document.get_active_pages())
            clean_doc = PDFDocument(path=temp_path, total_pages=active_count)
            for i in range(active_count):
                clean_doc.pages[i].source_file = temp_path
                clean_doc.pages[i].page_number = i + 1

            if self._on_save_callback:
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
        """Rotate selected pages by degrees. If none selected, rotate all included.

        Args:
            degrees: Rotation angle (90 or -90)
        """
        self._push_undo()
        thumbnails = self._page_grid._thumbnails
        selected = self._page_grid._selected_indices
        rotated = 0

        if selected:
            for idx in selected:
                if idx < len(thumbnails) and not thumbnails[idx].page_state.deleted:
                    thumbnails[idx].page_state.rotate(degrees)
                    rotated += 1
        else:
            for thumb in thumbnails:
                if not thumb.page_state.deleted:
                    thumb.page_state.rotate(degrees)
                    rotated += 1

        if rotated > 0 and self._document is not None:
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            target = "selected" if selected else "included"
            logger.info(f"Rotated {rotated} {target} page(s) by {degrees}°")

    def _move_selected_pages(self, direction: int) -> None:
        """Move selected pages up or down by one position.

        Pushes undo before the move.

        Provides keyboard-accessible alternative to drag-and-drop reordering.

        Args:
            direction: -1 to move up, +1 to move down
        """
        if not self._document:
            return

        selected = sorted(self._page_grid._selected_indices)
        if not selected:
            return

        self._push_undo()
        pages = self._document.pages
        total = len(pages)

        # Moving up: process from top; moving down: process from bottom
        if direction == 1:
            selected = list(reversed(selected))

        swaps: list[tuple[int, int]] = []
        for idx in selected:
            new_idx = idx + direction
            if new_idx < 0 or new_idx >= total:
                return  # Cannot move beyond bounds
            pages[idx], pages[new_idx] = pages[new_idx], pages[idx]
            swaps.append((idx, new_idx))

        # Update positions
        for i, page in enumerate(pages):
            page.position = i

        # Update selection indices
        self._page_grid._selected_indices = {
            idx + direction for idx in self._page_grid._selected_indices
        }

        self._document.mark_modified()
        # Swap thumbnails in FlowBox without remove/insert (preserves scroll)
        self._page_grid.swap_pages_in_grid(swaps)
        self._update_status_bar()
        logger.info(f"Moved {len(selected)} page(s) {'up' if direction < 0 else 'down'}")
