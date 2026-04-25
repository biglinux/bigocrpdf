"""Dialog window for choosing how to open multiple PDFs in the editor."""

from __future__ import annotations

import os
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GObject, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.tooltip_helper import get_tooltip_helper


class MultiPdfOpenDialog(Adw.Window):
    """Prompt for opening multiple selected PDFs separately or as one document."""

    def __init__(
        self,
        application: Gtk.Application,
        file_paths: list[str],
        on_open_individual: Callable[[list[str]], None],
        on_open_combined: Callable[[list[str]], None],
    ) -> None:
        super().__init__(application=application)

        self._file_paths = list(file_paths)
        self._on_open_individual = on_open_individual
        self._on_open_combined = on_open_combined
        self._order_list: Gtk.ListBox | None = None

        self.set_title(_("Open selected PDFs"))
        width, height = self._load_dialog_size()
        self.set_default_size(width, height)
        self.set_resizable(True)
        self.set_modal(False)
        self.connect("close-request", self._on_close_request)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)

        self._stack.add_named(self._build_choice_page(), "choice")
        self._stack.add_named(self._build_order_page(), "order")
        self._stack.set_visible_child_name("choice")

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label=_("Open selected PDFs")))
        toolbar.add_top_bar(header)
        toolbar.set_content(self._stack)
        self.set_content(toolbar)

    def _build_choice_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        page.set_margin_top(24)
        page.set_margin_bottom(24)
        page.set_margin_start(24)
        page.set_margin_end(24)

        heading = Gtk.Label()
        heading.set_markup(
            "<span size='large' weight='bold'>"
            + _("How would you like to open these files?")
            + "</span>"
        )
        heading.set_wrap(True)
        heading.set_xalign(0)
        page.append(heading)

        body = Gtk.Label(
            label=_("{count} PDF files were selected.").format(count=len(self._file_paths))
        )
        body.set_wrap(True)
        body.set_xalign(0)
        body.add_css_class("dim-label")
        page.append(body)

        options = Adw.PreferencesGroup()
        options.add(
            self._build_option_row(
                title=_("Open in separate windows"),
                subtitle=_("Edit each PDF in its own editor window."),
                icon_name="window-new-symbolic",
                callback=lambda: self._open_individual(),
            )
        )
        options.add(
            self._build_option_row(
                title=_("Combine in one editor"),
                subtitle=_("Import all pages using the current selection order."),
                icon_name="view-paged-symbolic",
                callback=lambda: self._open_combined(self._file_paths),
            )
        )
        options.add(
            self._build_option_row(
                title=_("Choose order and combine"),
                subtitle=_("Set the document order before importing the pages."),
                icon_name="view-sort-ascending-symbolic",
                callback=self._show_order_page,
            )
        )
        page.append(options)

        return page

    def _build_option_row(
        self,
        title: str,
        subtitle: str,
        icon_name: str,
        callback: Callable[[], None],
    ) -> Adw.ActionRow:
        row = Adw.ActionRow(title=title, subtitle=subtitle)
        row.add_prefix(Gtk.Image.new_from_icon_name(icon_name))
        row.set_activatable(True)
        set_a11y_label(row, f"{title}. {subtitle}")
        row.connect("activated", lambda _row: callback())
        row.add_suffix(Gtk.Image.new_from_icon_name("go-next-symbolic"))
        return row

    def _build_order_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        page.set_margin_top(18)
        page.set_margin_bottom(18)
        page.set_margin_start(18)
        page.set_margin_end(18)

        heading = Gtk.Label()
        heading.set_markup(
            "<span size='large' weight='bold'>" + _("Choose import order") + "</span>"
        )
        heading.set_wrap(True)
        heading.set_xalign(0)
        page.append(heading)

        hint = Gtk.Label(label=_("Pages will be imported from top to bottom."))
        hint.set_wrap(True)
        hint.set_xalign(0)
        hint.add_css_class("dim-label")
        page.append(hint)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_min_content_height(220)

        self._order_list = Gtk.ListBox()
        self._order_list.add_css_class("boxed-list")
        self._order_list.set_selection_mode(Gtk.SelectionMode.NONE)
        scrolled.set_child(self._order_list)
        page.append(scrolled)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        button_box.set_halign(Gtk.Align.END)

        back_button = Gtk.Button(label=_("Back"))
        back_button.connect("clicked", lambda _button: self._show_choice_page())
        button_box.append(back_button)

        open_button = Gtk.Button(label=_("Open combined"))
        open_button.add_css_class("suggested-action")
        open_button.connect("clicked", lambda _button: self._open_combined(self._file_paths))
        button_box.append(open_button)

        page.append(button_box)
        self._refresh_order_list()

        return page

    def _show_choice_page(self) -> None:
        self._stack.set_visible_child_name("choice")

    def _show_order_page(self) -> None:
        self._refresh_order_list()
        self._stack.set_visible_child_name("order")

    def _refresh_order_list(self) -> None:
        if self._order_list is None:
            return

        while row := self._order_list.get_row_at_index(0):
            self._order_list.remove(row)

        for index, path in enumerate(self._file_paths):
            self._order_list.append(self._build_file_row(index, path))

    def _build_file_row(self, index: int, path: str) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.set_activatable(False)
        row.set_selectable(False)
        set_a11y_label(
            row,
            _("{position}. {filename}").format(
                position=index + 1,
                filename=os.path.basename(path),
            ),
        )

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(10)
        box.set_margin_end(10)

        drag_area = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        drag_area.set_hexpand(True)

        number = Gtk.Label(label=str(index + 1))
        number.add_css_class("dim-label")
        number.set_size_request(28, -1)
        number.set_xalign(1)
        drag_area.append(number)

        icon = Gtk.Image.new_from_icon_name("application-pdf-symbolic")
        drag_area.append(icon)

        labels = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        labels.set_hexpand(True)

        name = Gtk.Label(label=os.path.basename(path))
        name.set_xalign(0)
        name.set_ellipsize(3)
        labels.append(name)

        folder = Gtk.Label(label=os.path.dirname(path))
        folder.set_xalign(0)
        folder.set_ellipsize(3)
        folder.add_css_class("caption")
        folder.add_css_class("dim-label")
        labels.append(folder)

        drag_area.append(labels)
        self._setup_file_drag_source(drag_area, index)
        box.append(drag_area)

        up_button = Gtk.Button()
        up_button.set_icon_name("go-up-symbolic")
        up_button.set_sensitive(index > 0)
        up_button.add_css_class("flat")
        get_tooltip_helper().add_tooltip(up_button, _("Move up"))
        set_a11y_label(up_button, _("Move up"))
        up_button.connect("clicked", lambda _button, i=index: self._move_file(i, -1))
        box.append(up_button)

        down_button = Gtk.Button()
        down_button.set_icon_name("go-down-symbolic")
        down_button.set_sensitive(index < len(self._file_paths) - 1)
        down_button.add_css_class("flat")
        get_tooltip_helper().add_tooltip(down_button, _("Move down"))
        set_a11y_label(down_button, _("Move down"))
        down_button.connect("clicked", lambda _button, i=index: self._move_file(i, 1))
        box.append(down_button)

        row.set_child(box)
        self._setup_file_drop_target(row, index)
        return row

    def _setup_file_drag_source(self, widget: Gtk.Widget, index: int) -> None:
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_file_drag_prepare, index)
        widget.add_controller(drag_source)

    def _setup_file_drop_target(self, row: Gtk.ListBoxRow, index: int) -> None:
        drop_target = Gtk.DropTarget.new(GObject.TYPE_INT, Gdk.DragAction.MOVE)
        drop_target.connect("drop", self._on_file_drop, index)
        row.add_controller(drop_target)

    def _on_file_drag_prepare(
        self,
        _source: Gtk.DragSource,
        _x: float,
        _y: float,
        index: int,
    ) -> Gdk.ContentProvider:
        value = GObject.Value(GObject.TYPE_INT, index)
        return Gdk.ContentProvider.new_for_value(value)

    def _on_file_drop(
        self,
        _target: Gtk.DropTarget,
        value: int,
        _x: float,
        _y: float,
        target_index: int,
    ) -> bool:
        return self._move_file_to_position(int(value), target_index)

    def _move_file(self, index: int, direction: int) -> None:
        new_index = index + direction
        if new_index < 0 or new_index >= len(self._file_paths):
            return
        self._file_paths[index], self._file_paths[new_index] = (
            self._file_paths[new_index],
            self._file_paths[index],
        )
        self._refresh_order_list()

    def _move_file_to_position(self, source_index: int, target_index: int) -> bool:
        if (
            source_index == target_index
            or source_index < 0
            or source_index >= len(self._file_paths)
        ):
            return False

        item = self._file_paths.pop(source_index)
        target_index = max(0, min(target_index, len(self._file_paths)))
        self._file_paths.insert(target_index, item)
        self._refresh_order_list()
        return True

    def _open_individual(self) -> None:
        self._save_dialog_size()
        self._on_open_individual(self._file_paths.copy())
        self.close()

    def _open_combined(self, file_paths: list[str]) -> None:
        self._save_dialog_size()
        self._on_open_combined(file_paths.copy())
        self.close()

    def _on_close_request(self, _window: Adw.Window) -> bool:
        self._save_dialog_size()
        return False

    @staticmethod
    def _load_dialog_size() -> tuple[int, int]:
        config = get_config_manager()
        width = config.get("multi_pdf_dialog.width", 560, int)
        height = config.get("multi_pdf_dialog.height", 460, int)
        return max(width, 420), max(height, 320)

    def _save_dialog_size(self) -> None:
        config = get_config_manager()
        width, height = self.get_width(), self.get_height()
        if width > 0 and height > 0:
            config.set("multi_pdf_dialog.width", width, save_immediately=False)
            config.set("multi_pdf_dialog.height", height, save_immediately=True)
