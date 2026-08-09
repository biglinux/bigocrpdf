"""Layout builders for the PDF editor window."""
# Host attributes are supplied by PDFEditorWindow's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GObject, Gtk, Pango

from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _


class EditorWindowLayoutMixin:
    def _setup_ui(self) -> None:
        self._split_view = Adw.OverlaySplitView()
        self._split_view.set_min_sidebar_width(280)
        self._split_view.set_max_sidebar_width(340)
        self._split_view.set_sidebar_width_fraction(0.3)
        self._split_view.set_enable_hide_gesture(True)
        self._split_view.set_enable_show_gesture(True)
        buttons_left = self._window_buttons_on_left()
        self._split_view.set_sidebar(self._create_sidebar(buttons_left))
        self._split_view.set_content(self._create_content_area(buttons_left))
        self.set_content(self._split_view)
        breakpoint = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 600sp"))
        breakpoint.add_setter(self._split_view, "collapsed", True)
        self.add_breakpoint(breakpoint)

    def _create_sidebar(self, buttons_left: bool) -> Adw.ToolbarView:
        sidebar_toolbar = Adw.ToolbarView()
        sidebar_toolbar.add_css_class("sidebar")
        sidebar_toolbar.add_top_bar(self._create_sidebar_header(buttons_left))
        sidebar_scroll = Gtk.ScrolledWindow()
        sidebar_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar_scroll.set_vexpand(True)
        sidebar_scroll.set_child(self._create_sidebar_box())
        sidebar_toolbar.set_content(sidebar_scroll)
        return sidebar_toolbar

    def _create_sidebar_header(self, buttons_left: bool) -> Adw.HeaderBar:
        from bigocrpdf.config import APP_ICON_NAME

        sidebar_header = Adw.HeaderBar()
        sidebar_header.add_css_class("sidebar")
        sidebar_header.set_show_title(True)
        sidebar_header.set_decoration_layout("close,maximize,minimize:menu" if buttons_left else "")
        app_icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        app_icon.set_pixel_size(20)
        title_label = Gtk.Label(label=_("Edit PDF"))
        title_label.add_css_class("heading")
        sidebar_header.pack_start(app_icon)
        if not buttons_left:
            center_box = Gtk.CenterBox()
            center_box.set_hexpand(True)
            center_box.set_center_widget(title_label)
            sidebar_header.set_title_widget(center_box)
        else:
            title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            title_box.append(title_label)
            expander = Gtk.Box()
            expander.set_hexpand(True)
            title_box.append(expander)
            sidebar_header.set_title_widget(title_box)
        return sidebar_header

    def _create_sidebar_box(self) -> Gtk.Box:
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        sidebar_box.set_margin_top(12)
        sidebar_box.set_margin_bottom(24)
        sidebar_box.set_margin_start(12)
        sidebar_box.set_margin_end(12)
        sidebar_box.append(self._create_document_actions_group())
        sidebar_box.append(self._create_page_actions_group())
        return sidebar_box

    def _create_document_actions_group(self) -> Adw.PreferencesGroup:
        doc_group = Adw.PreferencesGroup()
        self._compress_btn = self._document_action_row(
            "Compress PDF", "document-properties-symbolic", "editor.compress"
        )
        self._compress_btn.set_tooltip_text(_("Reduce document file size"))
        doc_group.add(self._compress_btn)
        self._split_pages_btn = self._document_action_row(
            "Split by Page Count", "view-dual-symbolic", "editor.split-pages"
        )
        self._split_pages_btn.set_tooltip_text(_("Split document by number of pages"))
        doc_group.add(self._split_pages_btn)
        self._split_size_btn = self._document_action_row(
            "Split by Size", "view-dual-symbolic", "editor.split-size"
        )
        self._split_size_btn.set_tooltip_text(_("Split document by target file size"))
        doc_group.add(self._split_size_btn)
        self._add_page_layout_row(doc_group)
        return doc_group

    def _document_action_row(self, title: str, icon_name: str, action_name: str) -> Adw.ActionRow:
        row = Adw.ActionRow(title=_(title))
        row.add_prefix(Gtk.Image.new_from_icon_name(icon_name))
        row.set_activatable(True)
        row.set_action_name(action_name)
        return row

    def _add_page_layout_row(self, doc_group: Adw.PreferencesGroup) -> None:
        self._page_layout_values = ["default", "single", "continuous", "two_page"]
        self._page_layout_combo = Adw.ComboRow(title=_("Page Layout"))
        self._page_layout_combo.add_prefix(Gtk.Image.new_from_icon_name("view-paged-symbolic"))
        self._page_layout_combo.set_model(
            Gtk.StringList.new(
                [
                    _("Default (viewer decides)"),
                    _("Single page"),
                    _("Continuous scroll"),
                    _("Two pages"),
                ]
            )
        )
        current_layout = get_config_manager().get("output.page_layout", "default")
        self._page_layout_combo.set_selected(
            self._page_layout_values.index(current_layout)
            if current_layout in self._page_layout_values
            else 0
        )
        self._page_layout_combo.connect("notify::selected", self._on_page_layout_changed)
        self._page_layout_combo.set_tooltip_text(
            _("How PDF viewers arrange pages when opening the file")
        )
        doc_group.add(self._page_layout_combo)

    def _create_page_actions_group(self) -> Adw.PreferencesGroup:
        page_group = Adw.PreferencesGroup()
        self._select_all_btn = self._page_action_row(
            "Select All", "object-select-symbolic", lambda _r: self._on_select_all(None)
        )
        self._select_all_btn.set_tooltip_text(_("Select all pages in the document") + " (Ctrl+A)")
        page_group.add(self._select_all_btn)
        self._deselect_all_btn = self._page_action_row(
            "Deselect All", "edit-clear-all-symbolic", lambda _r: self._on_deselect_all(None)
        )
        self._deselect_all_btn.set_tooltip_text(_("Clear page selection"))
        page_group.add(self._deselect_all_btn)
        self._add_rotate_flip_row(page_group)
        self._reverse_btn = self._document_action_row(
            "Reverse Order", "view-sort-ascending-symbolic", "editor.reverse"
        )
        self._reverse_btn.set_tooltip_text(_("Reverse the order of all pages"))
        page_group.add(self._reverse_btn)
        return page_group

    def _page_action_row(self, title: str, icon_name: str, callback) -> Adw.ActionRow:
        row = Adw.ActionRow(title=_(title))
        row.add_prefix(Gtk.Image.new_from_icon_name(icon_name))
        row.set_activatable(True)
        row.connect("activated", callback)
        return row

    def _add_rotate_flip_row(self, page_group: Adw.PreferencesGroup) -> None:
        self._rotate_flip_btn = Adw.ActionRow(title=_("Rotate / Flip"))
        self._rotate_flip_btn.add_prefix(
            Gtk.Image.new_from_icon_name("object-rotate-right-symbolic")
        )
        self._rotate_flip_btn.set_activatable(True)
        rotate_menu = Gio.Menu()
        rotate_menu.append(_("Rotate Left 90º"), "editor.rotate-left")
        rotate_menu.append(_("Rotate Right 90º"), "editor.rotate-right")
        rotate_menu.append(_("Flip Vertically"), "editor.flip-vertical")
        rotate_menu.append(_("Flip Horizontally"), "editor.flip-horizontal")
        menu_button = Gtk.MenuButton()
        menu_button.set_menu_model(rotate_menu)
        menu_button.set_icon_name("go-next-symbolic")
        menu_button.set_valign(Gtk.Align.CENTER)
        menu_button.add_css_class("flat")
        menu_button.set_tooltip_text(_("Open rotate and flip actions"))
        set_a11y_label(menu_button, _("Open rotate and flip actions"))
        self._rotate_flip_btn.add_suffix(menu_button)
        self._rotate_flip_btn.connect("activated", lambda _r: menu_button.popup())
        self._rotate_flip_btn.set_tooltip_text(
            _("Rotate or flip selected pages") + " (Ctrl+L / Ctrl+R)"
        )
        page_group.add(self._rotate_flip_btn)

    def _create_content_area(self, buttons_left: bool) -> Adw.ToolbarView:
        content_toolbar = Adw.ToolbarView()
        content_toolbar.add_top_bar(self._create_content_header(buttons_left))
        content_toolbar.set_content(self._create_content_overlay())
        self._status_bar = self._create_status_bar()
        content_toolbar.add_bottom_bar(self._status_bar)
        return content_toolbar

    def _create_content_header(self, buttons_left: bool) -> Adw.HeaderBar:
        content_header = Adw.HeaderBar()
        content_header.set_show_end_title_buttons(True)
        content_header.set_show_start_title_buttons(True)
        content_header.set_decoration_layout(
            "close,maximize,minimize:" if buttons_left else "menu:minimize,maximize,close"
        )
        self._add_sidebar_toggle(content_header)
        content_header.set_title_widget(self._create_center_actions())
        content_header.pack_end(self._create_menu_button())
        return content_header

    def _add_sidebar_toggle(self, content_header: Adw.HeaderBar) -> None:
        self.sidebar_toggle = Gtk.ToggleButton()
        self.sidebar_toggle.set_icon_name("sidebar-show-symbolic")
        self.sidebar_toggle.set_valign(Gtk.Align.CENTER)
        self.sidebar_toggle.add_css_class("flat")
        self.sidebar_toggle.set_tooltip_text(_("Show or hide the editor sidebar"))
        self._update_sidebar_toggle_a11y()
        self._split_view.bind_property(
            "show-sidebar",
            self.sidebar_toggle,
            "active",
            GObject.BindingFlags.SYNC_CREATE | GObject.BindingFlags.BIDIRECTIONAL,
        )
        self.sidebar_toggle.connect("notify::active", self._on_sidebar_toggle_active_changed)
        content_header.pack_start(self.sidebar_toggle)
        self._split_view.connect(
            "notify::collapsed", lambda sv, _p: self.sidebar_toggle.set_visible(sv.get_collapsed())
        )
        self.sidebar_toggle.set_visible(self._split_view.get_collapsed())

    def _create_center_actions(self) -> Gtk.Box:
        center_actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        center_actions_box.set_halign(Gtk.Align.CENTER)
        self._add_button_header = Gtk.Button(label=_("Add Files"))
        self._add_button_header.add_css_class("suggested-action")
        self._add_button_header.set_tooltip_text(_("Insert PDF or image files"))
        set_a11y_label(self._add_button_header, _("Add files"))
        self._add_button_header.connect("clicked", self._on_add_files_clicked)
        center_actions_box.append(self._add_button_header)
        center_actions_box.append(self._create_apply_button())
        return center_actions_box

    def _create_apply_button(self) -> Gtk.Button:
        self._apply_button = Gtk.Button(label=_("Save As") if self._standalone else _("Apply"))
        self._apply_button.add_css_class("suggested-action")
        self._apply_button.set_margin_start(12)
        if self._standalone:
            self._apply_button.set_tooltip_text(_("Save PDF to a new file (Ctrl+S)"))
            set_a11y_label(self._apply_button, _("Save PDF to a new file"))
            self._apply_button.connect("clicked", self._on_save_as_clicked)
        else:
            self._apply_button.set_tooltip_text(_("Save changes and go back (Ctrl+S)"))
            set_a11y_label(self._apply_button, _("Save changes and go back"))
            self._apply_button.connect("clicked", self._on_ok_clicked)
        return self._apply_button

    def _create_menu_button(self) -> Gtk.MenuButton:
        self.menu_button = Gtk.MenuButton()
        self.menu_button.set_icon_name("open-menu-symbolic")
        self.menu_button.set_tooltip_text(_("Menu"))
        set_a11y_label(self.menu_button, _("Menu"))
        menu_model = Gio.Menu.new()
        menu_model.append(_("Help"), "editor.help")
        menu_model.append(_("About"), "app.about")
        menu_model.append(_("Quit"), "app.quit")
        self.menu_button.set_menu_model(menu_model)
        return self.menu_button

    def _create_content_overlay(self) -> Gtk.Overlay:
        content_overlay = Gtk.Overlay()
        self._page_grid = PageGrid()
        self._page_grid.on_before_mutate = self._push_undo
        self._page_grid.connect("selection-changed", self._on_selection_changed)
        self._page_grid.connect("page-ocr-toggled", self._on_page_ocr_toggled)
        content_overlay.set_child(self._page_grid)
        self._notification_revealer = Gtk.Revealer()
        self._notification_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._notification_revealer.set_transition_duration(200)
        self._notification_revealer.set_reveal_child(False)
        self._notification_revealer.set_valign(Gtk.Align.START)
        self._notification_revealer.set_halign(Gtk.Align.CENTER)
        self._notification_revealer.set_can_target(False)
        self._notification_revealer.set_child(self._create_notification_box())
        content_overlay.add_overlay(self._notification_revealer)
        return content_overlay

    def _create_notification_box(self) -> Gtk.Box:
        self._notification_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._notification_box.add_css_class("editor-notification")
        self._notification_box.set_margin_top(8)
        self._notification_box.set_accessible_role(Gtk.AccessibleRole.STATUS)
        self._notification_icon = Gtk.Image()
        self._notification_icon.set_icon_size(Gtk.IconSize.NORMAL)
        self._notification_box.append(self._notification_icon)
        self._notification_label = Gtk.Label()
        self._notification_label.set_wrap(True)
        self._notification_label.set_max_width_chars(60)
        self._notification_box.append(self._notification_label)
        return self._notification_box

    def _create_status_bar(self) -> Gtk.Box:
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_box.add_css_class("editor-status-bar")
        status_box.set_margin_start(12)
        status_box.set_margin_end(12)
        status_box.set_margin_top(4)
        status_box.set_margin_bottom(4)
        self._filename_label = Gtk.Label(label=self._pdf_path)
        self._filename_label.add_css_class("dim-label")
        self._filename_label.set_halign(Gtk.Align.START)
        self._filename_label.set_ellipsize(Pango.EllipsizeMode.END)
        set_a11y_label(self._filename_label, _("Current file"))
        status_box.append(self._filename_label)
        filler = Gtk.Box()
        filler.set_hexpand(True)
        status_box.append(filler)
        self._status_label = Gtk.Label()
        self._status_label.add_css_class("dim-label")
        set_a11y_label(self._status_label, _("Document status"))
        status_box.append(self._status_label)
        self._selection_label = Gtk.Label()
        self._selection_label.add_css_class("dim-label")
        self._selection_label.set_halign(Gtk.Align.END)
        set_a11y_label(self._selection_label, _("Selection count"))
        status_box.append(self._selection_label)
        self._add_zoom_dropdown(status_box)
        return status_box

    def _add_zoom_dropdown(self, status_box: Gtk.Box) -> None:
        zoom_levels = Gtk.StringList.new(["50%", "75%", "100%", "150%", "200%", "300%", "400%"])
        self._zoom_dropdown = Gtk.DropDown(model=zoom_levels)
        self._zoom_dropdown.set_selected(2)
        self._zoom_dropdown.set_tooltip_text(_("Change the size of page previews"))
        set_a11y_label(self._zoom_dropdown, _("Zoom level"))
        self._zoom_dropdown.connect("notify::selected", self._on_zoom_dropdown_changed)
        status_box.append(self._zoom_dropdown)

    def _on_sidebar_toggle_active_changed(
        self, _button: Gtk.ToggleButton, _pspec: GObject.ParamSpec
    ) -> None:
        self._update_sidebar_toggle_a11y()

    def _update_sidebar_toggle_a11y(self) -> None:
        if self.sidebar_toggle.get_active():
            set_a11y_label(self.sidebar_toggle, _("Hide editor sidebar"))
        else:
            set_a11y_label(self.sidebar_toggle, _("Show editor sidebar"))
