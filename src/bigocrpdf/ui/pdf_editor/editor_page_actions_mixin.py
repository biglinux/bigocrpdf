"""Page manipulation actions for PDFEditorWindow: rotate, flip, move, drop, keyboard."""
# Host attributes are supplied by PDFEditorWindow's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import os
from typing import cast

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk

from bigocrpdf.ui.pdf_editor.page_model import PageState
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.ui.widgets import get_default_clipboard, parse_clipboard_file_paths
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.temp_manager import mkstemp, remove_file


def _valid_external_drop_paths(value, supported_extensions: tuple[str, ...]) -> list[str]:
    return [
        path
        for path in _external_drop_paths(value)
        if os.path.exists(path) and path.lower().endswith(supported_extensions)
    ]


def _external_drop_paths(value) -> list[str]:
    if isinstance(value, Gio.File):
        path = value.get_path()
        return [path] if path else []

    if hasattr(value, "get_files"):
        return _gio_file_paths(value.get_files())

    if hasattr(value, "__iter__"):
        return _gio_file_paths(
            [file_value for file_value in value if isinstance(file_value, Gio.File)]
        )

    return []


def _gio_file_paths(files) -> list[str]:
    paths = []
    for file_value in files:
        path = file_value.get_path()
        if path:
            paths.append(path)
    return paths


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
            valid_paths = _valid_external_drop_paths(value, supported_extensions)
            if not valid_paths:
                return False

            return self._add_files_to_document(valid_paths) > 0

        except Exception as e:
            logger.error(f"Error handling dropped files: {e}")
            return False

    def _add_files_to_document(self, file_paths: list[str]) -> int:
        """Add external files (PDFs or images) to the current document."""
        if not self._document:
            return 0

        added_pages: list[PageState] = []
        current_total = self._document.total_pages
        renderer = get_thumbnail_renderer()

        for path in file_paths:
            try:
                page_count = renderer.get_page_count(path)
                if page_count > 0:
                    for i in range(page_count):
                        added_pages.append(
                            PageState(
                                page_number=i + 1,
                                position=current_total + len(added_pages),
                                source_file=path,
                            )
                        )
                    logger.info(f"Added {page_count} pages from: {path}")
            except Exception as e:
                logger.error(f"Failed to add file {path}: {e}")
                self._show_error(
                    _("Failed to add file {}: {}").format(os.path.basename(path), str(e))
                )

        added_count = len(added_pages)
        if added_count > 0:
            self._push_undo()
            self._document.pages.extend(added_pages)
            self._document.total_pages += added_count
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            logger.info(
                f"Added {added_count} pages via drag-and-drop. Total: {self._document.total_pages}"
            )
        return added_count

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
            if self._page_grid.selected_indices:
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
        clipboard = get_default_clipboard()
        if clipboard is None:
            logger.warning("Editor clipboard is unavailable because no display is active")
            return
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
            data = stream.read_bytes(1024 * 1024, None).get_data()
            if data is None:
                return
            raw = data.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Editor: failed to decode clipboard stream: {e}")
            return
        finally:
            try:
                stream.close(None)
            except Exception as e:
                logger.error(f"Editor: failed to close clipboard stream: {e}")

        file_paths = parse_clipboard_file_paths(raw)

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
            png_data = png_bytes.get_data()
            if png_data is None:
                return
            fd, tmp_path = mkstemp(suffix=".png", prefix="bigocrpdf_paste_")
            try:
                os.write(fd, png_data)
            finally:
                os.close(fd)
        except Exception as e:
            logger.error(f"Editor: failed to save clipboard image: {e}")
            return

        if self._add_files_to_document([tmp_path]) == 0:
            remove_file(tmp_path)

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

        dialog.open_multiple(cast(Gtk.Window, self), None, self._on_pdfs_selected)

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
        thumbnails = self._page_grid.thumbnails
        selected = self._page_grid.selected_indices
        if selected:
            targets = [
                thumbnails[idx]
                for idx in selected
                if 0 <= idx < len(thumbnails) and not thumbnails[idx].page_state.deleted
            ]
        else:
            targets = [thumb for thumb in thumbnails if not thumb.page_state.deleted]

        if not targets or degrees % 360 == 0 or self._document is None:
            return

        self._push_undo()
        for thumb in targets:
            thumb.page_state.rotate(degrees)

        rotated = len(targets)
        if rotated > 0:
            self._document.mark_modified()
            self._page_grid.refresh()
            self._update_status_bar()
            target = "selected" if selected else "included"
            logger.info(f"Rotated {rotated} {target} page(s) by {degrees}°")

    def _flip_selected_pages(self, horizontal: bool = True) -> None:
        """Flip selected pages. If none selected, flip all included."""
        thumbnails = self._page_grid.thumbnails
        selected = self._page_grid.selected_indices
        if selected:
            targets = [
                thumbnails[idx]
                for idx in selected
                if 0 <= idx < len(thumbnails) and not thumbnails[idx].page_state.deleted
            ]
        else:
            targets = [thumb for thumb in thumbnails if not thumb.page_state.deleted]

        if not targets or self._document is None:
            return

        self._push_undo()
        for thumb in targets:
            if horizontal:
                thumb.page_state.toggle_flip_horizontal()
            else:
                thumb.page_state.toggle_flip_vertical()

        flipped = len(targets)
        if flipped > 0:
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

        pages = self._document.pages
        total = len(pages)
        selected = sorted(idx for idx in self._page_grid.selected_indices if 0 <= idx < total)
        if not selected:
            return
        if any(not 0 <= idx + direction < total for idx in selected):
            return

        self._push_undo()

        # Moving up: process from top; moving down: process from bottom
        if direction == 1:
            selected = list(reversed(selected))

        swaps: list[tuple[int, int]] = []
        for idx in selected:
            new_idx = idx + direction
            pages[idx], pages[new_idx] = pages[new_idx], pages[idx]
            swaps.append((idx, new_idx))

        for i, page in enumerate(pages):
            page.position = i

        self._page_grid.selected_indices = {
            idx + direction for idx in self._page_grid.selected_indices
        }

        self._document.mark_modified()
        # Swap thumbnails in FlowBox without remove/insert (preserves scroll)
        self._page_grid.swap_pages_in_grid(swaps)
        self._update_status_bar()
        logger.info(f"Moved {len(selected)} page(s) {'up' if direction < 0 else 'down'}")
