"""Page manipulation actions for PDFEditorWindow: rotate, flip, move, drop, keyboard."""

from __future__ import annotations

import os
import tempfile
from urllib.parse import unquote, urlparse

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gdk, Gio, GLib, Gtk

from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class EditorPageActionsMixin:
    """Mixin providing page-level actions for the PDF editor."""

    def _on_external_file_drop(
        self, _target: Gtk.DropTarget, value: Gdk.FileList, _x: float, _y: float
    ) -> bool:
        """Handle external file drop onto the editor."""
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
        """Add external files (PDFs or images) to the current document."""
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

    def _on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Handle keyboard shortcuts."""
        ctrl = state & Gdk.ModifierType.CONTROL_MASK

        if ctrl:
            return self._handle_ctrl_shortcut(keyval)

        if keyval == Gdk.KEY_Delete:
            if self._page_grid._selected_indices:
                self._push_undo()
                self._page_grid.toggle_ocr_for_selected()
            return True
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        if keyval in (Gdk.KEY_Page_Up, Gdk.KEY_Page_Down):
            self._scroll_page(keyval == Gdk.KEY_Page_Up)
            return True
        if keyval in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
            self._zoom_step(1)
            return True
        if keyval in (Gdk.KEY_minus, Gdk.KEY_underscore, Gdk.KEY_KP_Subtract):
            self._zoom_step(-1)
            return True

        return False

    def _handle_ctrl_shortcut(self, keyval: int) -> bool:
        """Dispatch Ctrl+key shortcuts. Returns True if handled."""
        dispatch = {
            Gdk.KEY_l: lambda: self._rotate_selected_pages(-90),
            Gdk.KEY_r: lambda: self._rotate_selected_pages(90),
            Gdk.KEY_z: self._undo,
            Gdk.KEY_a: self._page_grid.select_all,
            Gdk.KEY_Up: lambda: self._move_selected_pages(-1),
            Gdk.KEY_Down: lambda: self._move_selected_pages(1),
            Gdk.KEY_v: self._paste_from_clipboard,
        }
        handler = dispatch.get(keyval)
        if handler:
            handler()
            return True
        if keyval == Gdk.KEY_s:
            (self._on_save_as_clicked if self._standalone else self._on_ok_clicked)(None)
            return True
        return False

    def _scroll_page(self, up: bool) -> None:
        """Scroll the page grid by ~80% of visible area."""
        vadj = self._page_grid.get_vadjustment()
        step = vadj.get_page_size() * 0.8
        if up:
            vadj.set_value(max(vadj.get_lower(), vadj.get_value() - step))
        else:
            vadj.set_value(min(vadj.get_upper() - vadj.get_page_size(), vadj.get_value() + step))

    # ------------------------------------------------------------------
    # Clipboard paste (Ctrl+V)
    # ------------------------------------------------------------------

    _SUPPORTED_EXTENSIONS = (
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

    def _paste_from_clipboard(self) -> None:
        """Read clipboard and add images or PDFs as new pages."""
        clipboard = Gdk.Display.get_default().get_clipboard()
        formats = clipboard.get_formats()

        uri_mime_types = ["x-special/gnome-copied-files", "text/uri-list"]
        has_uris = any(formats.contain_mime_type(m) for m in uri_mime_types)

        if has_uris:
            clipboard.read_async(
                uri_mime_types,
                GLib.PRIORITY_DEFAULT,
                None,
                self._on_editor_clipboard_uris_ready,
            )
        elif formats.contain_gtype(Gdk.Texture):
            clipboard.read_texture_async(None, self._on_editor_clipboard_texture_ready)

    def _on_editor_clipboard_uris_ready(
        self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult
    ) -> None:
        """Handle clipboard file URIs in the editor."""
        try:
            stream, _mime = clipboard.read_finish(result)
        except Exception as e:
            logger.error(f"Editor: failed to read clipboard URIs: {e}")
            return

        if stream is None:
            return

        try:
            raw = stream.read_bytes(1024 * 1024, None).get_data().decode("utf-8", errors="replace")
            stream.close(None)
        except Exception as e:
            logger.error(f"Editor: failed to decode clipboard stream: {e}")
            return

        file_paths: list[str] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line in ("copy", "cut"):
                continue
            parsed = urlparse(line)
            if parsed.scheme == "file":
                path = unquote(parsed.path)
                if os.path.isfile(path):
                    file_paths.append(path)
            elif os.path.isfile(line):
                file_paths.append(line)

        valid = [p for p in file_paths if p.lower().endswith(self._SUPPORTED_EXTENSIONS)]
        if valid:
            self._add_files_to_document(valid)

    def _on_editor_clipboard_texture_ready(
        self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult
    ) -> None:
        """Handle clipboard image texture in the editor."""
        try:
            texture = clipboard.read_texture_finish(result)
        except Exception as e:
            logger.error(f"Editor: failed to read clipboard image: {e}")
            return

        if texture is None:
            return

        try:
            png_bytes = texture.save_to_png_bytes()
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="bigocrpdf_paste_")
            os.write(fd, png_bytes.get_data())
            os.close(fd)
        except Exception as e:
            logger.error(f"Editor: failed to save clipboard image: {e}")
            return

        self._add_files_to_document([tmp_path])

    def _on_add_files_clicked(self, _button: Gtk.Button) -> None:
        """Handle Add Files button click."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Add Files"))

        filter_any = Gtk.FileFilter()
        filter_any.set_name(_("PDFs and Images"))
        filter_any.add_mime_type("application/pdf")
        for mime in ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"]:
            filter_any.add_mime_type(mime)

        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(filter_any)
        dialog.set_filters(store)

        dialog.open_multiple(self, None, self._on_pdfs_selected)

    def _on_pdfs_selected(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle PDF file selection result."""
        try:
            files = dialog.open_multiple_finish(result)
            if files:
                file_paths = [f.get_path() for f in files if f.get_path()]
                if file_paths:
                    self._add_files_to_document(file_paths)
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error(f"Error selecting files: {e}")

    def _on_select_all(self, _button: Gtk.Button) -> None:
        """Handle Select All button click."""
        self._page_grid.select_all_for_ocr()

    def _on_deselect_all(self, _button: Gtk.Button) -> None:
        """Handle Deselect All button click."""
        self._page_grid.deselect_all_for_ocr()

    def _on_rotate_left(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        """Handle rotate left action."""
        if not self._document:
            return
        self._rotate_selected_pages(-90)

    def _on_rotate_right(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        """Handle rotate right action."""
        if not self._document:
            return
        self._rotate_selected_pages(90)

    def _on_flip_horizontal(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        """Handle flip horizontal action."""
        if not self._document:
            return
        self._flip_selected_pages(horizontal=True)

    def _on_flip_vertical(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        """Handle flip vertical action."""
        if not self._document:
            return
        self._flip_selected_pages(horizontal=False)

    def _rotate_selected_pages(self, degrees: int) -> None:
        """Rotate selected pages by degrees. If none selected, rotate all included."""
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

    def _flip_selected_pages(self, horizontal: bool = True) -> None:
        """Flip selected pages. If none selected, flip all included."""
        self._push_undo()
        thumbnails = self._page_grid._thumbnails
        selected = self._page_grid._selected_indices
        flipped = 0

        if selected:
            for idx in selected:
                if idx < len(thumbnails) and not thumbnails[idx].page_state.deleted:
                    if horizontal:
                        thumbnails[idx].page_state.toggle_flip_horizontal()
                    else:
                        thumbnails[idx].page_state.toggle_flip_vertical()
                    flipped += 1
        else:
            for thumb in thumbnails:
                if not thumb.page_state.deleted:
                    if horizontal:
                        thumb.page_state.toggle_flip_horizontal()
                    else:
                        thumb.page_state.toggle_flip_vertical()
                    flipped += 1

        if flipped > 0 and self._document is not None:
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            target = "selected" if selected else "included"
            direction = "horizontally" if horizontal else "vertically"
            logger.info(f"Flipped {flipped} {target} page(s) {direction}")

    def _move_selected_pages(self, direction: int) -> None:
        """Move selected pages up or down by one position."""
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

        for i, page in enumerate(pages):
            page.position = i

        self._page_grid._selected_indices = {
            idx + direction for idx in self._page_grid._selected_indices
        }

        self._document.mark_modified()
        # Swap thumbnails in FlowBox without remove/insert (preserves scroll)
        self._page_grid.swap_pages_in_grid(swaps)
        self._update_status_bar()
        logger.info(f"Moved {len(selected)} page(s) {'up' if direction < 0 else 'down'}")
