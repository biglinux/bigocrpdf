"""File queue panel, drag-and-drop, and file management for SettingsPageManager."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    pass


class SettingsQueueMixin:
    """Mixin providing file queue panel, drag-and-drop, and file actions."""

    def _display_name(self, file_path: str) -> str:
        """Return a user-friendly display name for a queued file."""
        return self.window.settings.display_name(file_path)

    def _create_file_queue_panel(self) -> Gtk.Widget:
        """Create the file queue panel for the right side."""
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

        self._selected_file_idx: int | None = None
        self._item_popover: Gtk.PopoverMenu | None = None

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
        folder_button.connect("clicked", self.window.on_browse_clicked)
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
            drop_target.set_gtypes([Gdk.FileList])
            drop_target.connect("drop", self._on_drop)
            widget.add_controller(drop_target)

    def _on_view_mode_toggled(self, is_grid: bool) -> None:
        """Switch between list and grid views."""
        if is_grid:
            self._queue_view_stack.set_visible_child_name("grid")
            self._populate_grid()
        else:
            self._queue_view_stack.set_visible_child_name("list")

    def _connect_view_toggles(self) -> None:
        """No-op — view toggle is handled via HeaderBar._on_view_toggle_clicked."""
        pass

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

        if hasattr(self, "placeholder") and self.placeholder:
            self.file_list_box.set_placeholder(self.placeholder)

        for idx, file_path in enumerate(self.window.settings.selected_files):
            self._create_file_row(file_path, idx)

        # Re-populate grid if it is currently visible or was previously populated
        if hasattr(self, "_file_grid_box"):
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
        from bigocrpdf.utils.pdf_utils import get_pdf_page_count

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
        label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        label.set_max_width_chars(22)
        label.add_css_class("caption")
        label.set_tooltip_text(display)
        box.append(label)

        # File info (pages + size)
        info_parts: list[str] = []
        pages = get_pdf_page_count(file_path)
        if pages > 0:
            info_parts.append(_("{pages} pg.").format(pages=pages))
        try:
            from bigocrpdf.constants import BYTES_PER_MB

            file_size = os.path.getsize(file_path) / BYTES_PER_MB
            info_parts.append(_("{size} MB").format(size=f"{file_size:.1f}"))
        except (OSError, FileNotFoundError):
            pass

        if info_parts:
            info_label = Gtk.Label()
            info_label.set_markup(f"<small>{' · '.join(info_parts)}</small>")
            info_label.add_css_class("dim-label")
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

        remove_btn = Gtk.Button.new_from_icon_name("trash-symbolic")
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

        drop = Gtk.DropTarget.new(GObject.TYPE_STRING, Gdk.DragAction.MOVE)
        drop.connect(
            "drop",
            lambda tgt, val, x, y, ti=idx: self._on_reorder_drop(int(val), ti),
        )
        box.add_controller(drop)

        # Load thumbnail asynchronously
        self._load_grid_thumbnail(file_path, image)

        return box

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

        try:
            original = self.window.settings.original_file_paths.get(file_path)
            dir_name = os.path.dirname(original or file_path)
            from bigocrpdf.constants import BYTES_PER_MB

            file_size = os.path.getsize(file_path) / BYTES_PER_MB
            size_str = _("{size} MB").format(size=f"{file_size:.1f}")
            subtitle = f"{dir_name}  •  {size_str}"
            row.set_subtitle(subtitle)
        except (OSError, FileNotFoundError):
            original = self.window.settings.original_file_paths.get(file_path)
            row.set_subtitle(os.path.dirname(original or file_path))

        self._add_page_count_to_row(row, file_path)

        # Left side: edit + remove
        edit_button = Gtk.Button.new_from_icon_name("document-edit-symbolic")
        edit_button.set_tooltip_text(_("Edit pages of this file"))
        set_a11y_label(edit_button, _("Edit pages of this file"))
        edit_button.add_css_class("flat")
        edit_button.set_valign(Gtk.Align.CENTER)
        edit_button.connect("clicked", lambda _b, fp=file_path: self._on_edit_file(fp))
        row.add_prefix(edit_button)

        remove_button = Gtk.Button.new_from_icon_name("trash-symbolic")
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

        drop = Gtk.DropTarget.new(GObject.TYPE_STRING, Gdk.DragAction.MOVE)
        drop.connect(
            "drop",
            lambda tgt, val, x, y, ti=idx: self._on_reorder_drop(int(val), ti),
        )
        row.add_controller(drop)

        self.file_list_box.append(row)

    # ── Popover item actions ──

    def _on_list_row_activated(self, listbox: Gtk.ListBox, row) -> None:
        """Handle list row activation (left-click / Enter key)."""
        pass

    def _on_grid_child_activated(self, flowbox: Gtk.FlowBox, child) -> None:
        """Handle grid tile activation (double-click / Enter key)."""
        pass

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
        try:
            pop.popdown()
            pop.unparent()
        except Exception:
            pass

    def _on_item_popover_closed(self, popover) -> None:
        """Auto-cleanup when the popover is dismissed by the user."""
        if self._item_popover is popover:
            self._item_popover = None
            GLib.idle_add(self._safe_unparent_popover, popover)

    @staticmethod
    def _safe_unparent_popover(popover) -> bool:
        try:
            popover.unparent()
        except Exception:
            pass
        return False

    def _on_reorder_drop(self, source_idx: int, target_idx: int) -> bool:
        """Handle drop to reorder files in the queue."""
        if source_idx == target_idx:
            return False
        files = self.window.settings.selected_files
        if not (0 <= source_idx < len(files)) or not (0 <= target_idx < len(files)):
            return False
        item = files.pop(source_idx)
        files.insert(target_idx, item)
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
            self.window.show_toast(_("Failed to open file"))

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the system file manager with the given file selected."""
        import subprocess

        file_uri = f"file://{os.path.abspath(file_path)}"
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
        import subprocess
        from collections import Counter
        from datetime import datetime

        info: dict[str, str] = {}

        # Parse pdfinfo output
        try:
            result = subprocess.run(
                ["pdfinfo", file_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    value = value.strip()
                    if value:
                        info[key.strip()] = value
        except Exception:
            pass

        # Parse pdfimages -list for image info
        image_count = 0
        image_formats: Counter[str] = Counter()
        image_details: list[dict] = []
        page_img_counts: dict[int, int] = {}
        try:
            result = subprocess.run(
                ["pdfimages", "-list", file_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            for line in result.stdout.splitlines()[2:]:  # skip header lines
                parts = line.split()
                if len(parts) >= 15:
                    image_count += 1
                    enc = parts[8]
                    image_formats[enc] += 1
                    pg = int(parts[0])
                    if pg not in page_img_counts:
                        page_img_counts[pg] = 0
                    in_page_idx = page_img_counts[pg]
                    page_img_counts[pg] += 1
                    image_details.append({
                        "page": pg,
                        "in_page_idx": in_page_idx,
                        "width": parts[3],
                        "height": parts[4],
                        "color": parts[5],
                        "enc": enc,
                        "size": parts[14],
                    })
        except Exception:
            pass

        # Parse pdffonts for font info
        fonts: list[str] = []
        embedded_count = 0
        try:
            result = subprocess.run(
                ["pdffonts", file_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            for line in result.stdout.splitlines()[2:]:  # skip header lines
                parts = line.split()
                if len(parts) >= 5:
                    fonts.append(parts[0])
                    if parts[3] == "yes":
                        embedded_count += 1
        except Exception:
            pass

        # Parse pdfdetach for embedded files
        attached_files: list[str] = []
        try:
            result = subprocess.run(
                ["pdfdetach", "-list", file_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("The following"):
                    # Lines like "1: filename.ext"
                    if ":" in stripped:
                        attached_files.append(stripped.split(":", 1)[1].strip())
        except Exception:
            pass

        # ── Build info dict for copy ──
        all_info: list[tuple[str, str, str]] = []  # (group, key, value)

        # File info
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
            all_info.append((
                "",
                _("Modified (local)"),
                datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            ))
        except OSError:
            pass

        # Document info
        group_doc = _("Document")
        for key in ("Pages", "Page size", "PDF version", "Encrypted", "Optimized", "Tagged"):
            if key in info:
                all_info.append((group_doc, _(key), info[key]))
        if "Form" in info:
            all_info.append((group_doc, _("Form"), info["Form"]))
        if "JavaScript" in info:
            all_info.append((group_doc, _("JavaScript"), info["JavaScript"]))

        # Content info
        group_content = _("Content")
        if image_count > 0:
            fmt_str = ", ".join(f"{fmt} ({cnt})" for fmt, cnt in image_formats.most_common())
            all_info.append((group_content, _("Images"), f"{image_count}"))
            all_info.append((group_content, _("Image formats"), fmt_str))
        else:
            all_info.append((group_content, _("Images"), "0"))

        if fonts:
            unique_fonts = sorted(set(fonts))
            all_info.append((group_content, _("Fonts"), f"{len(unique_fonts)}"))
            all_info.append((
                group_content,
                _("Embedded fonts"),
                f"{embedded_count} / {len(fonts)}",
            ))
            all_info.append((group_content, _("Font names"), ", ".join(unique_fonts)))

        if attached_files:
            all_info.append((group_content, _("Attached files"), ", ".join(attached_files)))

        # Metadata info
        group_meta = _("Metadata")
        for key in ("Title", "Subject", "Author", "Creator", "Producer", "Keywords"):
            if key in info:
                all_info.append((group_meta, _(key), info[key]))
        if "CreationDate" in info:
            all_info.append((group_meta, _("Created"), info["CreationDate"]))
        if "ModDate" in info:
            all_info.append((group_meta, _("Modified"), info["ModDate"]))

        # ── Build dialog ──
        dialog = Adw.Dialog()
        dialog.set_title(_("File information"))
        dialog.set_content_width(460)
        dialog.set_content_height(520)

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()

        # Copy button in header
        copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        copy_btn.set_tooltip_text(_("Copy all information"))
        set_a11y_label(copy_btn, _("Copy all information"))
        copy_btn.connect("clicked", lambda _b: self._copy_all_info(all_info))
        header.pack_end(copy_btn)

        toolbar.add_top_bar(header)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        page = Adw.PreferencesPage()

        current_group_name = None
        current_group = None

        for group_name, key, value in all_info:
            if group_name != current_group_name:
                current_group_name = group_name
                current_group = Adw.PreferencesGroup()
                if group_name:
                    current_group.set_title(group_name)
                page.add(current_group)
            self._add_info_row(current_group, key, value)

            # After the Content group, add the image list expander
            if group_name == group_content and key == _("Images") and image_details:
                expander = Adw.ExpanderRow()
                expander.set_title(_("Image list"))
                expander.set_subtitle(str(len(image_details)) + " " + _("images"))
                for img in image_details:
                    img_row = Adw.ActionRow()
                    img_row.set_title(f"Page {img['page']} — {img['width']}×{img['height']}")
                    img_row.set_subtitle(f"{img['enc'].upper()} · {img['size']} · {img['color']}")

                    extract_btn = Gtk.Button.new_from_icon_name("document-save-symbolic")
                    extract_btn.add_css_class("flat")
                    extract_btn.set_valign(Gtk.Align.CENTER)
                    extract_btn.set_tooltip_text(_("Extract this image"))
                    set_a11y_label(extract_btn, _("Extract this image"))
                    extract_btn.connect(
                        "clicked",
                        lambda _b, fp=file_path, i=img: self._extract_image(fp, i),
                    )
                    img_row.add_suffix(extract_btn)

                    expander.add_row(img_row)
                current_group.add(expander)

        scroll.set_child(page)
        toolbar.set_content(scroll)
        dialog.set_child(toolbar)
        dialog.present(self.window)

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

        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set("\n".join(lines))
        self.window.show_toast(_("Information copied"))

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

        def _on_save(_dialog, result):
            try:
                gfile = _dialog.save_finish(result)
            except GLib.Error:
                return
            if gfile is None:
                return
            save_path = gfile.get_path()
            self._do_extract_image(file_path, img_info, save_path)

        file_dialog.save(self.window, None, _on_save)

    def _do_extract_image(self, pdf_path: str, img_info: dict, save_path: str) -> None:
        """Extract image using pdfimages and save to destination."""
        import glob
        import shutil
        import subprocess
        import tempfile

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
            except Exception as e:
                logger.error(f"pdfimages extraction failed: {e}")
                self.window.show_toast(_("Failed to extract image"))
                return

            files = sorted(glob.glob(f"{prefix}-*"))
            if 0 <= idx < len(files):
                shutil.copy2(files[idx], save_path)
                self.window.show_toast(_("Image extracted"))
            else:
                self.window.show_toast(_("Failed to extract image"))

    def _on_edit_file(self, file_path: str) -> None:
        """Open the PDF editor for the file."""
        try:
            from bigocrpdf.ui.pdf_editor import PDFEditorWindow

            def on_editor_save(document):
                """Handle editor save callback."""
                if document.path != file_path:
                    try:
                        if file_path in self.window.settings.selected_files:
                            idx = self.window.settings.selected_files.index(file_path)
                            self.window.settings.selected_files[idx] = document.path

                            true_original = self.window.settings.original_file_paths.get(
                                file_path, file_path
                            )
                            self.window.settings.original_file_paths[document.path] = true_original

                            if file_path in self.window.settings.original_file_paths:
                                del self.window.settings.original_file_paths[file_path]

                            if file_path in self.window.settings.file_modifications:
                                del self.window.settings.file_modifications[file_path]

                            if file_path != true_original and os.path.exists(file_path):
                                try:
                                    os.remove(file_path)
                                    logger.info(f"Removed previous temp file: {file_path}")
                                except OSError as rm_err:
                                    logger.warning(f"Could not remove temp file: {rm_err}")

                            logger.info(
                                f"Replaced original file with merged output: {document.path}"
                            )
                    except ValueError:
                        logger.debug("File %s not found in queue, skipping update", file_path)
                else:
                    state = document.to_dict()
                    self.window.settings.file_modifications[document.path] = state

                self._populate_file_list()
                self.window.update_file_info()

                if hasattr(self, "refresh_queue_status"):
                    self.refresh_queue_status()

                self.window.show_toast(_("Changes saved"))
                logger.info(f"Editor saved changes to: {document.path}")

            initial_state = self.window.settings.file_modifications.get(file_path)

            editor = PDFEditorWindow(
                application=self.window.get_application(),
                pdf_path=file_path,
                on_save_callback=on_editor_save,
                parent_window=self.window,
                initial_state=initial_state,
            )
            editor.present()

            logger.info(f"Opened PDF editor for: {file_path}")
        except Exception as e:
            logger.error(f"Failed to open PDF editor: {e}")
            self.window.show_toast(_("Failed to open PDF editor"))

    def _add_page_count_to_row(self, row: Adw.ActionRow, file_path: str) -> None:
        """Add page count to a file row if available."""
        from bigocrpdf.utils.pdf_utils import get_pdf_page_count

        pages = get_pdf_page_count(file_path)
        if pages > 0:
            page_label = Gtk.Label()
            page_label.set_markup(f"<small>{_('{pages} pg.').format(pages=pages)}</small>")
            row.add_suffix(page_label)

    def _on_drop(self, drop_target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Handle file drop events for both single and multiple files."""
        from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file

        try:
            file_paths = self._extract_file_paths_from_drop(value)
            if not file_paths:
                return False

            valid_file_paths = self._filter_supported_files(file_paths)
            if not valid_file_paths:
                logger.warning("No valid files in drop data")
                return False

            logger.info(f"{len(valid_file_paths)} files dropped")

            image_files = [p for p in valid_file_paths if is_image_file(p)]
            pdf_files = [p for p in valid_file_paths if not is_image_file(p)]

            if pdf_files:
                self.window.settings.add_files(pdf_files)

            if len(image_files) > 1:
                self._show_drop_image_merge_dialog(image_files)
            elif len(image_files) == 1:
                try:
                    pdf_path = images_to_pdf(image_files)
                    self.window.settings.original_file_paths[pdf_path] = image_files[0]
                    self.window.settings.add_files([pdf_path])
                except Exception as e:
                    logger.error(f"Failed to convert dropped image to PDF: {e}")

            self._populate_file_list()
            file_count = len(self.window.settings.selected_files)
            if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
                self.window.custom_header_bar.update_queue_size(file_count)

            return True
        except Exception as e:
            logger.error(f"Error handling dropped file(s): {e}")
            return False

    def _show_drop_image_merge_dialog(self, image_files: list[str]) -> None:
        """Show merge dialog for dropped images."""

        def _on_complete() -> None:
            self._populate_file_list()
            file_count = len(self.window.settings.selected_files)
            if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
                self.window.custom_header_bar.update_queue_size(file_count)

        self.window.ui.dialogs_manager.show_image_merge_dialog(
            image_files,
            self.window.settings,
            heading=_("Multiple Images Dropped"),
            body=_("You dropped {} images. How would you like to add them?").format(
                len(image_files)
            ),
            on_complete=_on_complete,
        )

    def _extract_file_paths_from_drop(self, value) -> list[str]:
        """Extract file paths from drop value."""
        file_paths = []

        if isinstance(value, Gio.File):
            file_path = value.get_path()
            if file_path:
                file_paths.append(file_path)
        elif isinstance(value, list) or hasattr(value, "__iter__"):
            for file in value:
                if isinstance(file, Gio.File):
                    file_path = file.get_path()
                    if file_path:
                        file_paths.append(file_path)
        else:
            logger.warning(f"Unsupported drop value type: {type(value)}")

        return file_paths

    def _filter_supported_files(self, file_paths: list[str]) -> list[str]:
        """Filter file paths to only include valid PDF and image files."""
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
        valid_paths = []
        for file_path in file_paths:
            if not file_path.lower().endswith(supported_extensions):
                logger.warning(f"Ignoring unsupported file: {file_path}")
                continue

            if not os.path.exists(file_path):
                logger.warning(f"Ignoring nonexistent file: {file_path}")
                continue

            valid_paths.append(file_path)

        return valid_paths

    def _remove_single_file(self, idx: int) -> None:
        """Remove a single file from the list."""
        if idx < 0 or idx >= len(self.window.settings.selected_files):
            return

        file_path = self.window.settings.selected_files.pop(idx)
        logger.info(f"Removed file: {file_path}")

        self._populate_file_list()
        self.refresh_queue_status()
        self.window.announce_status(
            _("{count} files in queue").format(count=len(self.window.settings.selected_files))
        )

    def _remove_all_files(self) -> None:
        """Remove all files from the queue."""
        if not self.window.settings.selected_files:
            return

        logger.info(f"Removing all {len(self.window.settings.selected_files)} files from queue")
        self.window.settings.selected_files.clear()
        self._populate_file_list()
        self.refresh_queue_status()
        self.window.announce_status(_("All files removed from queue"))

    def _show_pdf_options_dialog(self) -> None:
        """Show PDF output options dialog."""
        if hasattr(self.window, "ui") and hasattr(self.window.ui, "show_pdf_options_dialog"):
            self.window.ui.show_pdf_options_dialog(lambda _: None)
        else:
            logger.warning("PDF options dialog not yet implemented")
