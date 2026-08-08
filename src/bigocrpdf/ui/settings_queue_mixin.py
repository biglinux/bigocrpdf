"""File queue panel, drag-and-drop, and file management for SettingsPageManager."""
# Host attributes are supplied by SettingsPageManager's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import cast

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk, Pango

from bigocrpdf.ui.widgets import get_default_clipboard
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger


def _run_pdf_info_command(command: list[str], timeout: int) -> str:
    try:
        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("PDF metadata command failed: %s", error)
        return ""

    if result.returncode != 0:
        logger.debug(
            "PDF metadata command returned %s: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return ""
    return result.stdout


def _pdfinfo_fields(file_path: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in _run_pdf_info_command(["pdfinfo", file_path], 10).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if value:
                info[key.strip()] = value
    return info


def _pdf_image_metadata(file_path: str) -> tuple[int, Counter[str], list[dict]]:
    image_count = 0
    image_formats: Counter[str] = Counter()
    image_details: list[dict] = []
    page_img_counts: dict[int, int] = {}

    for line in _run_pdf_info_command(["pdfimages", "-list", file_path], 15).splitlines()[2:]:
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            page_num = int(parts[0])
        except ValueError:
            continue
        image_count += 1
        enc = parts[8]
        image_formats[enc] += 1
        in_page_idx = page_img_counts.get(page_num, 0)
        page_img_counts[page_num] = in_page_idx + 1
        image_details.append(
            {
                "page": page_num,
                "in_page_idx": in_page_idx,
                "width": parts[3],
                "height": parts[4],
                "color": parts[5],
                "enc": enc,
                "size": parts[14],
            }
        )

    return image_count, image_formats, image_details


def _pdf_font_metadata(file_path: str) -> tuple[list[str], int]:
    fonts: list[str] = []
    embedded_count = 0
    for line in _run_pdf_info_command(["pdffonts", file_path], 10).splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 5:
            fonts.append(parts[0])
            if parts[3] == "yes":
                embedded_count += 1
    return fonts, embedded_count


def _pdf_attached_files(file_path: str) -> list[str]:
    attached_files: list[str] = []
    for line in _run_pdf_info_command(["pdfdetach", "-list", file_path], 10).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("The following") and ":" in stripped:
            attached_files.append(stripped.split(":", 1)[1].strip())
    return attached_files


def _read_pdf_metadata(
    file_path: str,
) -> tuple[dict[str, str], int, Counter[str], list[dict], list[str], int, list[str]]:
    info = _pdfinfo_fields(file_path)
    image_count, image_formats, image_details = _pdf_image_metadata(file_path)
    fonts, embedded_count = _pdf_font_metadata(file_path)
    attached_files = _pdf_attached_files(file_path)
    return (
        info,
        image_count,
        image_formats,
        image_details,
        fonts,
        embedded_count,
        attached_files,
    )


class SettingsQueueMixin:
    """Mixin providing file queue panel, drag-and-drop, and file actions."""

    def _display_name(self, file_path: str) -> str:
        """Return a user-friendly display name for a queued file."""
        return self.window.settings.display_name(file_path)

    def _create_file_queue_panel(self) -> Gtk.Widget:
        """Create the file queue panel for the right side."""
        self._queue_metadata_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="queue-metadata",
        )
        self._queue_metadata_waiters = {}
        self._queue_metadata_closed = False

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.set_vexpand(True)
        main_box.add_css_class("view")

        # View stack (list / grid)
        self._queue_view_stack = Gtk.Stack()
        self._queue_view_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._queue_view_stack.set_transition_duration(150)

        # ── List view ──
        queue_scroll = Gtk.ScrolledWindow()
        queue_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        queue_scroll.set_vexpand(True)

        self.file_list_box = Gtk.ListBox()
        self.file_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.file_list_box.add_css_class("boxed-list")
        self.file_list_box.set_margin_start(6)
        self.file_list_box.set_margin_end(6)
        self.file_list_box.set_margin_top(3)
        self.file_list_box.set_margin_bottom(6)
        self.file_list_box.connect("row-activated", self._on_list_row_activated)
        set_a11y_label(self.file_list_box, _("File queue"))

        self.placeholder = Adw.StatusPage()
        self.placeholder.set_icon_name("document-open-symbolic")
        self.placeholder.set_title(_("Add your PDFs"))
        self.placeholder.set_description(
            _("Drag PDF files and images here, or use the button in the header bar")
        )
        self.placeholder.set_vexpand(True)
        self.placeholder.set_hexpand(True)
        self.placeholder.set_margin_top(3)
        self.placeholder.set_margin_bottom(6)

        self.file_list_box.set_placeholder(self.placeholder)

        queue_scroll.set_child(self.file_list_box)
        self._queue_view_stack.add_named(queue_scroll, "list")

        # ── Grid view ──
        self._grid_outer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._grid_outer_box.set_vexpand(True)

        grid_scroll = Gtk.ScrolledWindow()
        grid_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        grid_scroll.set_vexpand(True)

        self._file_grid_box = Gtk.FlowBox()
        self._file_grid_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._file_grid_box.set_homogeneous(False)
        self._file_grid_box.set_column_spacing(12)
        self._file_grid_box.set_row_spacing(12)
        self._file_grid_box.set_margin_start(12)
        self._file_grid_box.set_margin_end(12)
        self._file_grid_box.set_margin_top(6)
        self._file_grid_box.set_margin_bottom(12)
        self._file_grid_box.set_min_children_per_line(2)
        self._file_grid_box.set_max_children_per_line(6)
        self._file_grid_box.set_valign(Gtk.Align.START)
        self._file_grid_box.set_activate_on_single_click(True)
        self._file_grid_box.connect("child-activated", self._on_grid_child_activated)
        set_a11y_label(self._file_grid_box, _("File queue"))

        grid_scroll.set_child(self._file_grid_box)

        # Grid placeholder (shown when grid is empty)
        self._grid_placeholder = Adw.StatusPage()
        self._grid_placeholder.set_icon_name("document-open-symbolic")
        self._grid_placeholder.set_title(_("Add your PDFs"))
        self._grid_placeholder.set_description(
            _("Drag PDF files and images here, or use the button in the header bar")
        )
        self._grid_placeholder.set_vexpand(True)
        self._grid_placeholder.set_hexpand(True)

        self._grid_outer_box.append(grid_scroll)
        self._queue_view_stack.add_named(self._grid_outer_box, "grid")

        self._setup_drag_and_drop()

        if self.window.settings.selected_files:
            self._populate_file_list()

        main_box.append(self._queue_view_stack)

        # Bottom options bar
        options_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        options_box.set_spacing(12)
        options_box.set_margin_start(12)
        options_box.set_margin_end(12)
        options_box.set_margin_top(12)
        options_box.set_margin_bottom(12)
        options_box.set_vexpand(False)

        folder_options_store = Gtk.StringList()
        folder_options_store.append(_("Save in the same folder as the original file"))
        folder_options_store.append(_("Custom folder"))

        self.folder_combo = Gtk.DropDown()
        self.folder_combo.set_model(folder_options_store)
        self.folder_combo.set_selected(0 if self.window.settings.save_in_same_folder else 1)
        self.folder_combo.set_valign(Gtk.Align.CENTER)
        self.folder_combo.update_property(
            [Gtk.AccessibleProperty.LABEL], [_("Output folder location")]
        )
        options_box.append(self.folder_combo)

        self.folder_entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.folder_entry_box.set_spacing(4)
        self.folder_entry_box.set_visible(not self.window.settings.save_in_same_folder)
        self.folder_entry_box.set_hexpand(True)

        self.dest_entry = Gtk.Entry()
        self.dest_entry.set_hexpand(True)
        self.dest_entry.set_placeholder_text(_("Select folder"))
        self.dest_entry.set_text(self.window.settings.destination_folder or "")
        self.dest_entry.update_property(
            [Gtk.AccessibleProperty.LABEL], [_("Destination folder path")]
        )
        self.folder_entry_box.append(self.dest_entry)

        folder_button = Gtk.Button()
        folder_button.set_icon_name("folder-symbolic")
        folder_button.set_tooltip_text(_("Browse for folder"))
        set_a11y_label(folder_button, _("Browse for folder"))
        folder_button.connect(
            "clicked", lambda _button: self.window.file_manager.show_folder_selection_dialog()
        )
        folder_button.add_css_class("flat")
        folder_button.add_css_class("circular")
        folder_button.set_valign(Gtk.Align.CENTER)
        self.folder_entry_box.append(folder_button)

        options_box.append(self.folder_entry_box)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        options_box.append(spacer)

        options_button = Gtk.Button(label=_("Output options"))
        options_button.connect("clicked", lambda _: self._show_pdf_options_dialog())
        set_a11y_label(options_button, _("Output options"))
        options_box.append(options_button)

        self.folder_combo.connect("notify::selected", self._on_folder_type_changed)

        main_box.append(options_box)

        return main_box

    def _setup_drag_and_drop(self) -> None:
        """Set up drag and drop functionality for both list and grid views."""
        for widget in (self.file_list_box, self._file_grid_box):
            drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
            drop_target.connect("drop", self._on_drop)
            widget.add_controller(drop_target)

    def _on_view_mode_toggled(self, is_grid: bool) -> None:
        """Switch between list and grid views."""
        if is_grid:
            self._queue_view_stack.set_visible_child_name("grid")
            self._populate_grid()
        else:
            self._queue_view_stack.set_visible_child_name("list")

    def _populate_file_list(self) -> None:
        """Populate the file list box with the selected files."""
        if not self.file_list_box:
            return

        self._dismiss_item_popover()

        while True:
            child = self.file_list_box.get_first_child()
            if child:
                self.file_list_box.remove(child)
            else:
                break

        self.file_list_box.set_placeholder(self.placeholder)

        for idx, file_path in enumerate(self.window.settings.selected_files):
            self._create_file_row(file_path, idx)

        if self._queue_view_stack.get_visible_child_name() == "grid":
            self._populate_grid()

    def _populate_grid(self) -> None:
        """Populate the grid view with thumbnail tiles."""
        grid = self._file_grid_box
        while True:
            child = grid.get_first_child()
            if child:
                grid.remove(child)
            else:
                break

        # Show/hide placeholder
        has_files = bool(self.window.settings.selected_files)
        outer = self._grid_outer_box
        # Ensure placeholder is managed properly
        if has_files:
            if self._grid_placeholder.get_parent() is outer:
                outer.remove(self._grid_placeholder)
        else:
            if self._grid_placeholder.get_parent() is None:
                outer.prepend(self._grid_placeholder)

        for idx, file_path in enumerate(self.window.settings.selected_files):
            tile = self._create_grid_tile(file_path, idx)
            grid.append(tile)

    _GRID_THUMB_SIZE = 180

    def _create_grid_tile(self, file_path: str, idx: int) -> Gtk.Widget:
        """Create a thumbnail tile for the grid view."""

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.START)
        box.set_margin_top(4)
        box.set_margin_bottom(4)

        # Thumbnail image
        image = Gtk.Picture()
        image.set_size_request(self._GRID_THUMB_SIZE, int(self._GRID_THUMB_SIZE * 1.41))
        image.set_content_fit(Gtk.ContentFit.CONTAIN)
        image.set_halign(Gtk.Align.CENTER)
        image.add_css_class("card")
        box.append(image)

        # Filename
        display = self._display_name(file_path)
        label = Gtk.Label(label=display)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_max_width_chars(22)
        label.add_css_class("caption")
        label.set_tooltip_text(display)
        box.append(label)

        info_label = Gtk.Label()
        info_label.add_css_class("caption")
        info_label.add_css_class("dim-label")
        info_label.set_visible(False)
        box.append(info_label)

        # Action buttons row
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        btn_box.set_halign(Gtk.Align.CENTER)

        edit_btn = Gtk.Button.new_from_icon_name("document-edit-symbolic")
        edit_btn.add_css_class("flat")
        edit_btn.add_css_class("circular")
        edit_btn.set_tooltip_text(_("Edit pages of this file"))
        set_a11y_label(edit_btn, _("Edit pages of this file"))
        edit_btn.connect("clicked", lambda _b, fp=file_path: self._on_edit_file(fp))
        btn_box.append(edit_btn)

        remove_btn = Gtk.Button.new_from_icon_name("user-trash-symbolic")
        remove_btn.add_css_class("flat")
        remove_btn.add_css_class("circular")
        remove_btn.set_tooltip_text(_("Remove this file from the list"))
        set_a11y_label(remove_btn, _("Remove this file from the list"))
        remove_btn.connect("clicked", lambda _b, i=idx: self._remove_single_file(i))
        btn_box.append(remove_btn)

        box.append(btn_box)

        box.set_cursor(Gdk.Cursor.new_from_name("pointer", None))

        # Right-click context menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)
        right_click.connect(
            "released",
            lambda g, _n, _x, _y, b=box, i=idx: self._show_item_popover(b, i),
        )
        box.add_controller(right_click)

        # Drag-to-reorder
        drag = Gtk.DragSource()
        drag.set_actions(Gdk.DragAction.MOVE)
        drag.connect(
            "prepare",
            lambda src, x, y, i=idx: Gdk.ContentProvider.new_for_value(
                GObject.Value(GObject.TYPE_STRING, str(i))
            ),
        )
        box.add_controller(drag)

        drop = Gtk.DropTarget.new(cast(type, GObject.TYPE_STRING), Gdk.DragAction.MOVE)
        drop.connect(
            "drop",
            lambda _tgt, val, _x, _y, ti=idx: self._on_reorder_drop(int(val), ti),
        )
        box.add_controller(drop)

        # Load thumbnail asynchronously
        self._load_grid_thumbnail(file_path, image)
        self._load_queue_metadata(
            file_path,
            lambda pages, size: self._update_grid_metadata(
                box,
                info_label,
                file_path,
                pages,
                size,
            ),
        )

        return box

    def _load_queue_metadata(
        self,
        file_path: str,
        callback: Callable[[int, float | None], None],
    ) -> None:
        if self._queue_metadata_closed:
            return

        waiters = self._queue_metadata_waiters.get(file_path)
        if waiters is not None:
            waiters.append(callback)
            return

        self._queue_metadata_waiters[file_path] = [callback]
        pool = self._queue_metadata_pool
        if pool is None:
            return
        future = pool.submit(self._read_queue_metadata, file_path)

        def on_complete(completed: Future[tuple[int, float | None]]) -> None:
            if not self._queue_metadata_closed:
                GLib.idle_add(self._deliver_queue_metadata, file_path, completed)

        future.add_done_callback(on_complete)

    @staticmethod
    def _read_queue_metadata(file_path: str) -> tuple[int, float | None]:
        from bigocrpdf.constants import BYTES_PER_MB
        from bigocrpdf.utils.pdf_utils import get_pdf_page_count

        pages = get_pdf_page_count(file_path)
        try:
            size_mb = os.path.getsize(file_path) / BYTES_PER_MB
        except OSError:
            size_mb = None
        return pages, size_mb

    def _deliver_queue_metadata(
        self,
        file_path: str,
        future: Future[tuple[int, float | None]],
    ) -> bool:
        if self._queue_metadata_closed:
            self._queue_metadata_waiters.pop(file_path, None)
            return GLib.SOURCE_REMOVE

        try:
            metadata = future.result()
        except Exception as error:
            logger.warning("Could not read queue metadata for %s: %s", file_path, error)
            metadata = (0, None)

        for callback in self._queue_metadata_waiters.pop(file_path, []):
            callback(*metadata)
        return GLib.SOURCE_REMOVE

    def _update_grid_metadata(
        self,
        box: Gtk.Box,
        label: Gtk.Label,
        file_path: str,
        pages: int,
        size_mb: float | None,
    ) -> None:
        if box.get_parent() is None or file_path not in self.window.settings.selected_files:
            return
        parts = self._format_queue_metadata(pages, size_mb)
        label.set_label(" · ".join(parts))
        label.set_visible(bool(parts))

    @staticmethod
    def _format_queue_metadata(pages: int, size_mb: float | None) -> list[str]:
        parts: list[str] = []
        if pages > 0:
            parts.append(ngettext("{count} page", "{count} pages", pages).format(count=pages))
        if size_mb is not None:
            parts.append(_("{size} MB").format(size=f"{size_mb:.1f}"))
        return parts

    def _load_grid_thumbnail(self, file_path: str, image_widget: Gtk.Picture) -> None:
        """Load a thumbnail for a grid tile using the existing renderer."""
        from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer

        renderer = get_thumbnail_renderer()

        def _on_loaded(pixbuf) -> None:
            if pixbuf is not None and image_widget.get_parent() is not None:
                texture = Gdk.Texture.new_for_pixbuf(pixbuf)
                image_widget.set_paintable(texture)

        renderer.render_page_thumbnail_async(file_path, 0, _on_loaded, self._GRID_THUMB_SIZE)

    def _create_file_row(self, file_path: str, idx: int) -> None:
        """Create a row for a single file in the list."""
        row = Adw.ActionRow()
        row.set_activatable(True)
        row._file_idx = idx

        file_name = self._display_name(file_path)
        row.set_title(file_name)

        original = self.window.settings.original_file_paths.get(file_path)
        directory = os.path.dirname(original or file_path)
        row.set_subtitle(directory)

        page_label = Gtk.Label()
        page_label.add_css_class("caption")
        page_label.set_visible(False)
        row.add_suffix(page_label)

        # Left side: edit + remove
        edit_button = Gtk.Button.new_from_icon_name("document-edit-symbolic")
        edit_button.set_tooltip_text(_("Edit pages of this file"))
        set_a11y_label(edit_button, _("Edit pages of this file"))
        edit_button.add_css_class("flat")
        edit_button.set_valign(Gtk.Align.CENTER)
        edit_button.connect("clicked", lambda _b, fp=file_path: self._on_edit_file(fp))
        row.add_prefix(edit_button)

        remove_button = Gtk.Button.new_from_icon_name("user-trash-symbolic")
        remove_button.set_tooltip_text(_("Remove this file from the list"))
        set_a11y_label(remove_button, _("Remove this file from the list"))
        remove_button.add_css_class("flat")
        remove_button.set_valign(Gtk.Align.CENTER)
        remove_button.connect("clicked", lambda _b, i=idx: self._remove_single_file(i))
        row.add_prefix(remove_button)

        row.set_cursor(Gdk.Cursor.new_from_name("pointer", None))

        # Right-click context menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)
        right_click.connect(
            "released",
            lambda g, _n, _x, _y, r=row, i=idx: self._show_item_popover(r, i),
        )
        row.add_controller(right_click)

        # Drag-to-reorder
        drag = Gtk.DragSource()
        drag.set_actions(Gdk.DragAction.MOVE)
        drag.connect(
            "prepare",
            lambda src, x, y, i=idx: Gdk.ContentProvider.new_for_value(
                GObject.Value(GObject.TYPE_STRING, str(i))
            ),
        )
        row.add_controller(drag)

        drop = Gtk.DropTarget.new(cast(type, GObject.TYPE_STRING), Gdk.DragAction.MOVE)
        drop.connect(
            "drop",
            lambda _tgt, val, _x, _y, ti=idx: self._on_reorder_drop(int(val), ti),
        )
        row.add_controller(drop)

        self.file_list_box.append(row)
        self._load_queue_metadata(
            file_path,
            lambda pages, size: self._update_row_metadata(
                row,
                page_label,
                file_path,
                directory,
                pages,
                size,
            ),
        )

    def _update_row_metadata(
        self,
        row: Adw.ActionRow,
        page_label: Gtk.Label,
        file_path: str,
        directory: str,
        pages: int,
        size_mb: float | None,
    ) -> None:
        if row.get_parent() is None or file_path not in self.window.settings.selected_files:
            return
        if size_mb is not None:
            row.set_subtitle(f"{directory}  •  {_('{size} MB').format(size=f'{size_mb:.1f}')}")
        if pages > 0:
            page_label.set_label(
                ngettext("{count} page", "{count} pages", pages).format(count=pages)
            )
            page_label.set_visible(True)

    # ── Popover item actions ──

    def _on_list_row_activated(self, _listbox: Gtk.ListBox, row) -> None:
        """Handle list row activation (Enter key) — open file editor."""
        idx = getattr(row, "_file_idx", None)
        if idx is not None and 0 <= idx < len(self.window.settings.selected_files):
            self._on_edit_file(self.window.settings.selected_files[idx])

    def _on_grid_child_activated(self, _flowbox: Gtk.FlowBox, child) -> None:
        """Handle grid tile activation (double-click / Enter key) — open file editor."""
        idx = child.get_index()
        if 0 <= idx < len(self.window.settings.selected_files):
            self._on_edit_file(self.window.settings.selected_files[idx])

    def _show_item_popover(self, widget: Gtk.Widget, idx: int) -> None:
        """Show a contextual popover with open/reveal actions."""
        self._dismiss_item_popover()

        self._selected_file_idx = idx

        group = Gio.SimpleActionGroup()

        open_a = Gio.SimpleAction.new("open", None)
        open_a.connect("activate", lambda *_: self._action_on_selected(self._on_open_file))
        group.add_action(open_a)

        reveal_a = Gio.SimpleAction.new("reveal", None)
        reveal_a.connect(
            "activate", lambda *_: self._action_on_selected(self._reveal_in_file_manager)
        )
        group.add_action(reveal_a)

        info_a = Gio.SimpleAction.new("info", None)
        info_a.connect("activate", lambda *_: self._action_on_selected(self._show_file_info))
        group.add_action(info_a)

        widget.insert_action_group("item", group)

        menu = Gio.Menu()
        file_section = Gio.Menu()
        file_section.append(_("Open file"), "item.open")
        file_section.append(_("Show in file manager"), "item.reveal")
        menu.append_section(None, file_section)

        info_section = Gio.Menu()
        info_section.append(_("File information"), "item.info")
        menu.append_section(None, info_section)

        popover = Gtk.PopoverMenu.new_from_model(menu)
        popover.set_parent(widget)
        popover.connect("closed", self._on_item_popover_closed)
        self._item_popover = popover
        popover.popup()

    def _dismiss_item_popover(self) -> None:
        """Safely close and unparent the popover if active."""
        pop = self._item_popover
        if pop is None:
            return
        self._item_popover = None
        pop.popdown()
        if pop.get_parent() is not None:
            pop.unparent()

    def _on_item_popover_closed(self, popover) -> None:
        """Auto-cleanup when the popover is dismissed by the user."""
        if self._item_popover is popover:
            self._item_popover = None
            GLib.idle_add(self._safe_unparent_popover, popover)

    @staticmethod
    def _safe_unparent_popover(popover) -> bool:
        if popover.get_parent() is not None:
            popover.unparent()
        return GLib.SOURCE_REMOVE

    def _on_reorder_drop(self, source_idx: int, target_idx: int) -> bool:
        """Handle drop to reorder files in the queue."""
        if not self.window.settings._move_file(source_idx, target_idx):
            return False
        self._populate_file_list()
        return True

    def _action_on_selected(self, action_fn) -> None:
        """Run an action function with the currently selected file path."""
        idx = self._selected_file_idx
        files = self.window.settings.selected_files
        if idx is not None and 0 <= idx < len(files):
            action_fn(files[idx])

    def _on_open_file(self, file_path: str) -> None:
        """Open file with the default application."""
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        if not open_file_with_default_app(file_path):
            self.window.ui.show_toast(_("Failed to open file"))

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the system file manager with the given file selected."""
        file_uri = Gio.File.new_for_path(file_path).get_uri()
        try:
            subprocess.Popen(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:{file_uri}",
                    "string:",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            from bigocrpdf.utils.pdf_utils import open_file_with_default_app

            open_file_with_default_app(os.path.dirname(file_path))

    def _show_file_info(self, file_path: str) -> None:
        """Show a dialog with detailed file metadata."""
        if self._queue_metadata_closed:
            return

        pool = self._queue_metadata_pool
        if pool is None:
            return
        future = pool.submit(_read_pdf_metadata, file_path)

        def on_complete(completed) -> None:
            if not self._queue_metadata_closed:
                GLib.idle_add(self._deliver_file_info, file_path, completed)

        future.add_done_callback(on_complete)

    def _deliver_file_info(self, file_path: str, future) -> bool:
        if self._queue_metadata_closed or file_path not in self.window.settings.selected_files:
            return GLib.SOURCE_REMOVE

        try:
            (
                info,
                image_count,
                image_formats,
                image_details,
                fonts,
                embedded_count,
                attached_files,
            ) = future.result()
        except Exception:
            logger.exception("Could not read PDF metadata for %s", file_path)
            return GLib.SOURCE_REMOVE

        all_info = self._build_file_info_rows(
            file_path,
            info,
            image_count,
            image_formats,
            fonts,
            embedded_count,
            attached_files,
        )
        self._show_file_info_dialog(file_path, all_info, image_details)
        return GLib.SOURCE_REMOVE

    def _build_file_info_rows(
        self,
        file_path: str,
        info: dict[str, str],
        image_count: int,
        image_formats: Counter[str],
        fonts: list[str],
        embedded_count: int,
        attached_files: list[str],
    ) -> list[tuple[str, str, str]]:
        all_info: list[tuple[str, str, str]] = []
        self._append_basic_file_info(all_info, file_path)
        self._append_document_info(all_info, info)
        self._append_content_info(
            all_info,
            image_count,
            image_formats,
            fonts,
            embedded_count,
            attached_files,
        )
        self._append_metadata_info(all_info, info)
        return all_info

    def _append_basic_file_info(
        self,
        all_info: list[tuple[str, str, str]],
        file_path: str,
    ) -> None:
        all_info.append(("", _("Name"), self._display_name(file_path)))
        all_info.append(("", _("Path"), os.path.dirname(file_path)))

        try:
            from bigocrpdf.constants import BYTES_PER_MB

            size = os.path.getsize(file_path)
            all_info.append(("", _("Size"), f"{size / BYTES_PER_MB:.2f} MB ({size:,} bytes)"))
        except OSError:
            pass

        try:
            mtime = os.path.getmtime(file_path)
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            all_info.append(("", _("Modified (local)"), modified))
        except OSError:
            pass

    @staticmethod
    def _append_document_info(all_info: list[tuple[str, str, str]], info: dict[str, str]) -> None:
        group_doc = _("Document")
        for key in ("Pages", "Page size", "PDF version", "Encrypted", "Optimized", "Tagged"):
            if key in info:
                all_info.append((group_doc, _(key), info[key]))
        if "Form" in info:
            all_info.append((group_doc, _("Form"), info["Form"]))
        if "JavaScript" in info:
            all_info.append((group_doc, _("JavaScript"), info["JavaScript"]))

    @staticmethod
    def _append_content_info(
        all_info: list[tuple[str, str, str]],
        image_count: int,
        image_formats: Counter[str],
        fonts: list[str],
        embedded_count: int,
        attached_files: list[str],
    ) -> None:
        group_content = _("Content")
        all_info.append((group_content, _("Images"), f"{image_count}"))
        if image_count > 0:
            fmt_str = ", ".join(f"{fmt} ({cnt})" for fmt, cnt in image_formats.most_common())
            all_info.append((group_content, _("Image formats"), fmt_str))

        if fonts:
            unique_fonts = sorted(set(fonts))
            all_info.append((group_content, _("Fonts"), f"{len(unique_fonts)}"))
            all_info.append(
                (group_content, _("Embedded fonts"), f"{embedded_count} / {len(fonts)}")
            )
            all_info.append((group_content, _("Font names"), ", ".join(unique_fonts)))

        if attached_files:
            all_info.append((group_content, _("Attached files"), ", ".join(attached_files)))

    @staticmethod
    def _append_metadata_info(all_info: list[tuple[str, str, str]], info: dict[str, str]) -> None:
        group_meta = _("Metadata")
        for key in ("Title", "Subject", "Author", "Creator", "Producer", "Keywords"):
            if key in info:
                all_info.append((group_meta, _(key), info[key]))
        if "CreationDate" in info:
            all_info.append((group_meta, _("Created"), info["CreationDate"]))
        if "ModDate" in info:
            all_info.append((group_meta, _("Modified"), info["ModDate"]))

    def _show_file_info_dialog(
        self,
        file_path: str,
        all_info: list[tuple[str, str, str]],
        image_details: list[dict],
    ) -> None:
        dialog = Adw.Dialog()
        dialog.set_title(_("File information"))
        dialog.set_content_width(460)
        dialog.set_content_height(520)

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.pack_end(self._file_info_copy_button(all_info))
        toolbar.add_top_bar(header)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(self._file_info_preferences_page(file_path, all_info, image_details))

        toolbar.set_content(scroll)
        dialog.set_child(toolbar)
        dialog.present(self.window)

    def _file_info_copy_button(self, all_info: list[tuple[str, str, str]]) -> Gtk.Button:
        copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        copy_btn.set_tooltip_text(_("Copy all information"))
        set_a11y_label(copy_btn, _("Copy all information"))
        copy_btn.connect("clicked", lambda _b: self._copy_all_info(all_info))
        return copy_btn

    def _file_info_preferences_page(
        self,
        file_path: str,
        all_info: list[tuple[str, str, str]],
        image_details: list[dict],
    ) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage()
        current_group_name = None
        current_group = None

        for group_name, key, value in all_info:
            if current_group is None or group_name != current_group_name:
                current_group_name = group_name
                current_group = Adw.PreferencesGroup()
                if group_name:
                    current_group.set_title(group_name)
                page.add(current_group)
            self._add_info_row(current_group, key, value)

            if group_name == _("Content") and key == _("Images") and image_details:
                current_group.add(self._image_details_expander(file_path, image_details))

        return page

    def _image_details_expander(self, file_path: str, image_details: list[dict]) -> Adw.ExpanderRow:
        expander = Adw.ExpanderRow()
        expander.set_title(_("Image list"))
        image_count = len(image_details)
        expander.set_subtitle(
            ngettext("{count} image", "{count} images", image_count).format(count=image_count)
        )
        for img in image_details:
            expander.add_row(self._image_detail_row(file_path, img))
        return expander

    def _image_detail_row(self, file_path: str, img: dict) -> Adw.ActionRow:
        img_row = Adw.ActionRow()
        img_row.set_title(f"Page {img['page']} — {img['width']}×{img['height']}")
        img_row.set_subtitle(f"{img['enc'].upper()} · {img['size']} · {img['color']}")

        extract_btn = Gtk.Button.new_from_icon_name("document-save-symbolic")
        extract_btn.add_css_class("flat")
        extract_btn.set_valign(Gtk.Align.CENTER)
        extract_btn.set_tooltip_text(_("Extract this image"))
        set_a11y_label(extract_btn, _("Extract this image"))
        extract_btn.connect("clicked", lambda _b, fp=file_path, i=img: self._extract_image(fp, i))
        img_row.add_suffix(extract_btn)
        return img_row

    def _copy_all_info(self, info_rows: list[tuple[str, str, str]]) -> None:
        """Copy all info to clipboard in a readable text format."""
        lines: list[str] = []
        current_group = None
        for group_name, key, value in info_rows:
            if group_name != current_group:
                current_group = group_name
                if lines:
                    lines.append("")
                if group_name:
                    lines.append(f"── {group_name} ──")
            lines.append(f"{key}: {value}")

        clipboard = get_default_clipboard()
        if clipboard is None:
            logger.warning("Clipboard is unavailable because no display is active")
            return
        clipboard.set("\n".join(lines))
        self.window.ui.show_toast(_("Information copied"))

    @staticmethod
    def _add_info_row(group: Adw.PreferencesGroup, title: str, value: str) -> None:
        """Add a read-only info row to a preferences group."""
        row = Adw.ActionRow()
        row.set_title(title)
        row.set_subtitle(value)
        row.set_subtitle_selectable(True)
        group.add(row)

    def _extract_image(self, file_path: str, img_info: dict) -> None:
        """Extract a single image from the PDF via file save dialog."""

        enc = img_info["enc"]
        ext_map = {"jpeg": "jpg", "jpx": "jp2", "ccitt": "tif", "jbig2": "jbig2"}
        ext = ext_map.get(enc, enc if enc else "png")

        name = f"image_p{img_info['page']}_{img_info['in_page_idx']}.{ext}"

        file_dialog = Gtk.FileDialog()
        file_dialog.set_initial_name(name)

        def _on_save(_dialog, result) -> None:
            try:
                gfile = _dialog.save_finish(result)
            except GLib.Error as error:
                if (
                    error.matches(Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED)
                    or error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED)
                    or error.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED)
                ):
                    return
                logger.error("Could not choose an image destination: %s", error)
                self.window.ui.show_toast(_("Failed to extract image"))
                return

            save_path = gfile.get_path()
            if save_path is None:
                logger.warning("Cannot extract an image to a non-local destination")
                self.window.ui.show_toast(_("Failed to extract image"))
                return
            self._do_extract_image(file_path, img_info, save_path)

        file_dialog.save(self.window, None, _on_save)

    def _do_extract_image(self, pdf_path: str, img_info: dict, save_path: str) -> None:
        """Extract image using pdfimages and save to destination."""
        page = str(img_info["page"])
        idx = img_info["in_page_idx"]

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "img")
            try:
                subprocess.run(
                    ["pdfimages", "-all", "-f", page, "-l", page, pdf_path, prefix],
                    check=True,
                    timeout=30,
                    capture_output=True,
                )
            except (OSError, subprocess.SubprocessError) as error:
                logger.error("pdfimages extraction failed: %s", error)
                self.window.ui.show_toast(_("Failed to extract image"))
                return

            files = sorted(glob.glob(f"{prefix}-*"))
            if 0 <= idx < len(files):
                try:
                    shutil.copy2(files[idx], save_path)
                except OSError as error:
                    logger.error("Could not save extracted image: %s", error)
                    self.window.ui.show_toast(_("Failed to extract image"))
                    return
                self.window.ui.show_toast(_("Image extracted"))
                return

            self.window.ui.show_toast(_("Failed to extract image"))

    def _on_edit_file(self, file_path: str) -> None:
        """Open the PDF editor for the file."""
        try:
            from bigocrpdf.ui.pdf_editor import PDFEditorWindow

            initial_state = self.window.settings.file_modifications.get(file_path)

            editor = PDFEditorWindow(
                application=self.window.get_application(),
                pdf_path=file_path,
                on_save_callback=lambda document: self._handle_pdf_editor_save(
                    file_path,
                    document,
                ),
                initial_state=initial_state,
            )
            editor.present()

            logger.info("Opened PDF editor for: %s", file_path)
        except (GLib.Error, OSError, RuntimeError, ValueError) as error:
            logger.error("Failed to open PDF editor: %s", error)
            self.window.ui.show_toast(_("Failed to open PDF editor"))

    def _handle_pdf_editor_save(self, file_path: str, document) -> bool:
        if document.path != file_path:
            if not self.window.settings._replace_file(file_path, document.path):
                logger.warning("Could not persist edited queue replacement for %s", file_path)
                return False
            logger.info("Replaced original file with merged output: %s", document.path)
        else:
            self.window.settings.file_modifications[document.path] = document.to_dict()

        self.window.ui.update_file_info()
        self.refresh_queue_status()

        self.window.ui.show_toast(_("Changes saved"))
        logger.info("Editor saved changes to: %s", document.path)
        return True

    def _on_drop(self, _drop_target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Handle file drop events for both single and multiple files."""
        from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file

        if not isinstance(value, Gdk.FileList):
            logger.warning("Unsupported drop value type: %s", type(value).__name__)
            return False

        file_paths: list[str] = []
        for file in value.get_files():
            file_path = file.get_path()
            if file_path is None:
                logger.warning("Ignoring non-local dropped file: %s", file.get_uri())
                continue
            file_paths.append(file_path)

        valid_file_paths = self._filter_supported_files(file_paths)
        if not valid_file_paths:
            logger.warning("No valid files in drop data")
            return False

        logger.info("%s files dropped", len(valid_file_paths))
        image_files = [path for path in valid_file_paths if is_image_file(path)]
        pdf_files = [path for path in valid_file_paths if not is_image_file(path)]

        if pdf_files:
            self.window.settings.add_files(pdf_files)

        if len(image_files) > 1:
            self._show_drop_image_merge_dialog(image_files)
        elif image_files:
            try:
                pdf_path = images_to_pdf(image_files)
                self.window.settings._add_generated_file(pdf_path, image_files[0])
            except (OSError, RuntimeError, ValueError) as error:
                logger.error("Failed to convert dropped image to PDF: %s", error)

        self.refresh_queue_status()
        return True

    def _show_drop_image_merge_dialog(self, image_files: list[str]) -> None:
        """Show merge dialog for dropped images."""
        self.window.ui.dialogs_manager.show_image_merge_dialog(
            image_files,
            heading=_("Multiple Images Dropped"),
            body=ngettext(
                "You dropped {count} image. How would you like to add it?",
                "You dropped {count} images. How would you like to add them?",
                len(image_files),
            ).format(count=len(image_files)),
            on_complete=self.refresh_queue_status,
        )

    def _filter_supported_files(self, file_paths: list[str]) -> list[str]:
        """Filter file paths to only include valid PDF and image files."""
        from bigocrpdf.utils.pdf_utils import is_image_file

        queued_identities = {
            os.path.realpath(path)
            for path in (
                *self.window.settings.selected_files,
                *self.window.settings.original_file_paths.values(),
            )
        }
        valid_paths: list[str] = []
        for file_path in file_paths:
            if not file_path.lower().endswith(".pdf") and not is_image_file(file_path):
                logger.warning("Ignoring unsupported file: %s", file_path)
                continue

            if not os.path.isfile(file_path):
                logger.warning("Ignoring missing or non-regular file: %s", file_path)
                continue

            identity = os.path.realpath(file_path)
            if identity in queued_identities:
                logger.info("Ignoring file already represented in queue: %s", file_path)
                continue

            valid_paths.append(file_path)
            queued_identities.add(identity)

        return valid_paths

    def _remove_single_file(self, idx: int) -> None:
        """Remove a single file from the list."""
        if idx < 0 or idx >= len(self.window.settings.selected_files):
            return

        file_path = self.window.settings.selected_files[idx]
        if not self.window.settings._remove_file(file_path):
            return
        logger.info("Removed file: %s", file_path)

        self.refresh_queue_status()
        file_count = len(self.window.settings.selected_files)
        self.window.announce_status(
            ngettext(
                "{count} file in queue",
                "{count} files in queue",
                file_count,
            ).format(count=file_count)
        )

    def _remove_all_files(self) -> None:
        """Remove all files from the queue."""
        if not self.window.settings.selected_files:
            return

        logger.info("Removing all %s files from queue", len(self.window.settings.selected_files))
        if not self.window.settings._clear_files():
            return
        self.refresh_queue_status()
        self.window.announce_status(_("All files removed from queue"))

    def _show_pdf_options_dialog(self) -> None:
        """Show PDF output options dialog."""
        self.window.ui.dialogs_manager.show_pdf_options_dialog(lambda _: None)

    def cleanup(self) -> None:
        """Release queue-owned asynchronous resources."""
        self._queue_metadata_closed = True
        self._queue_metadata_waiters.clear()
        self._dismiss_item_popover()
        pool = self._queue_metadata_pool
        self._queue_metadata_pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
