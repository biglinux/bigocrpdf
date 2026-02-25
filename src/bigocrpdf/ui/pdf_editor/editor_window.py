"""
BigOcrPdf - PDF Editor Window

Main editor window for PDF page manipulation before OCR processing.
Redesigned with a visible action bar for discoverability and accessibility.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from bigocrpdf.ui.pdf_editor.editor_page_actions_mixin import EditorPageActionsMixin
from bigocrpdf.ui.pdf_editor.editor_tools_mixin import EditorToolsMixin
from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.tooltip_helper import get_tooltip_helper

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow


class PDFEditorWindow(EditorToolsMixin, EditorPageActionsMixin, Adw.Window):
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
        on_save_callback: Callable[[PDFDocument], None] | None = None,
        on_close_callback: Callable[[], None] | None = None,
        parent_window: "BigOcrPdfWindow | None" = None,
        initial_state: dict | None = None,
        standalone: bool = False,
    ) -> None:
        """Initialize the PDF editor window.

        Args:
            application: The Gtk Application instance
            pdf_path: Path to the PDF file to edit (None for empty editor)
            on_save_callback: Callback when user saves changes
            on_close_callback: Callback when window is closed
            parent_window: Optional parent window reference
            initial_state: Optional dictionary to restore page states
            standalone: If True, show Save As button instead of Apply
        """
        super().__init__(application=application)

        self._pdf_path = pdf_path
        self._on_save_callback = on_save_callback
        self._on_close_callback = on_close_callback
        self._parent_window = parent_window
        self._initial_state = initial_state
        self._standalone = standalone
        self._document: PDFDocument | None = None
        self._undo_stack: list[list[dict]] = []
        self._notification_timer_id: int | None = None

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
        if self._should_show_editor_help():
            GLib.idle_add(self._on_show_help)

    def _setup_ui(self) -> None:
        """Set up the window UI."""

        # --- Main Layout ---
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
        """Create the sidebar with document and page action groups."""
        from bigocrpdf.config import APP_ICON_NAME

        sidebar_toolbar = Adw.ToolbarView()
        sidebar_toolbar.add_css_class("sidebar")

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

        sidebar_toolbar.add_top_bar(sidebar_header)

        sidebar_scroll = Gtk.ScrolledWindow()
        sidebar_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar_scroll.set_vexpand(True)

        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        sidebar_box.set_margin_top(12)
        sidebar_box.set_margin_bottom(24)
        sidebar_box.set_margin_start(12)
        sidebar_box.set_margin_end(12)

        # Group 1: Document Actions
        doc_group = Adw.PreferencesGroup()

        self._save_copy_btn = Adw.ActionRow(title=_("Save As..."))
        self._save_copy_btn.add_prefix(Gtk.Image.new_from_icon_name("document-save-as-symbolic"))
        self._save_copy_btn.set_activatable(True)
        self._save_copy_btn.set_action_name("editor.save-copy")
        get_tooltip_helper().add_tooltip(self._save_copy_btn, _("Export a copy of the document"))
        doc_group.add(self._save_copy_btn)

        self._compress_btn = Adw.ActionRow(title=_("Compress PDF"))
        self._compress_btn.add_prefix(Gtk.Image.new_from_icon_name("document-properties-symbolic"))
        self._compress_btn.set_activatable(True)
        self._compress_btn.set_action_name("editor.compress")
        get_tooltip_helper().add_tooltip(self._compress_btn, _("Reduce document file size"))
        doc_group.add(self._compress_btn)

        self._split_pages_btn = Adw.ActionRow(title=_("Split by Page Count"))
        self._split_pages_btn.add_prefix(Gtk.Image.new_from_icon_name("view-dual-symbolic"))
        self._split_pages_btn.set_activatable(True)
        self._split_pages_btn.set_action_name("editor.split-pages")
        get_tooltip_helper().add_tooltip(
            self._split_pages_btn, _("Split document by number of pages")
        )
        doc_group.add(self._split_pages_btn)

        self._split_size_btn = Adw.ActionRow(title=_("Split by Size"))
        self._split_size_btn.add_prefix(Gtk.Image.new_from_icon_name("view-dual-symbolic"))
        self._split_size_btn.set_activatable(True)
        self._split_size_btn.set_action_name("editor.split-size")
        get_tooltip_helper().add_tooltip(
            self._split_size_btn, _("Split document by target file size")
        )
        doc_group.add(self._split_size_btn)

        sidebar_box.append(doc_group)

        # Group 2: Page Actions
        page_group = Adw.PreferencesGroup()

        self._select_all_btn = Adw.ActionRow(title=_("Select All"))
        self._select_all_btn.add_prefix(Gtk.Image.new_from_icon_name("object-select-symbolic"))
        self._select_all_btn.set_activatable(True)
        self._select_all_btn.connect("activated", lambda _r: self._on_select_all(None))
        get_tooltip_helper().add_tooltip(
            self._select_all_btn, _("Select all pages in the document") + " (Ctrl+A)"
        )
        page_group.add(self._select_all_btn)

        self._deselect_all_btn = Adw.ActionRow(title=_("Deselect All"))
        self._deselect_all_btn.add_prefix(Gtk.Image.new_from_icon_name("edit-clear-all-symbolic"))
        self._deselect_all_btn.set_activatable(True)
        self._deselect_all_btn.connect("activated", lambda _r: self._on_deselect_all(None))
        get_tooltip_helper().add_tooltip(self._deselect_all_btn, _("Clear page selection"))
        page_group.add(self._deselect_all_btn)

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
        self._rotate_flip_btn.add_suffix(menu_button)
        self._rotate_flip_btn.connect("activated", lambda _r: menu_button.popup())
        get_tooltip_helper().add_tooltip(
            self._rotate_flip_btn,
            _("Rotate or flip selected pages") + " (Ctrl+L / Ctrl+R)",
        )
        page_group.add(self._rotate_flip_btn)

        self._reverse_btn = Adw.ActionRow(title=_("Reverse Order"))
        self._reverse_btn.add_prefix(Gtk.Image.new_from_icon_name("view-sort-ascending-symbolic"))
        self._reverse_btn.set_activatable(True)
        self._reverse_btn.set_action_name("editor.reverse")
        get_tooltip_helper().add_tooltip(self._reverse_btn, _("Reverse the order of all pages"))
        page_group.add(self._reverse_btn)

        sidebar_box.append(page_group)

        sidebar_scroll.set_child(sidebar_box)
        sidebar_toolbar.set_content(sidebar_scroll)

        return sidebar_toolbar

    def _create_content_area(self, buttons_left: bool) -> Adw.ToolbarView:
        """Create the content area with header, page grid, and status bar."""
        content_toolbar = Adw.ToolbarView()

        content_header = Adw.HeaderBar()
        content_header.set_show_end_title_buttons(True)
        content_header.set_show_start_title_buttons(True)
        content_header.set_decoration_layout(
            "close,maximize,minimize:" if buttons_left else "menu:minimize,maximize,close"
        )

        # Sidebar Toggle Button
        self.sidebar_toggle = Gtk.ToggleButton()
        self.sidebar_toggle.set_icon_name("sidebar-show-symbolic")
        self.sidebar_toggle.set_valign(Gtk.Align.CENTER)
        self.sidebar_toggle.add_css_class("flat")
        get_tooltip_helper().add_tooltip(self.sidebar_toggle, _("Toggle Sidebar"))
        set_a11y_label(self.sidebar_toggle, _("Toggle Sidebar"))
        self._split_view.bind_property(
            "show-sidebar",
            self.sidebar_toggle,
            "active",
            GObject.BindingFlags.SYNC_CREATE | GObject.BindingFlags.BIDIRECTIONAL,
        )
        content_header.pack_start(self.sidebar_toggle)

        self._split_view.connect(
            "notify::collapsed", lambda sv, _p: self.sidebar_toggle.set_visible(sv.get_collapsed())
        )
        self.sidebar_toggle.set_visible(self._split_view.get_collapsed())

        # Centered Add Files + Apply buttons
        center_actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        center_actions_box.set_halign(Gtk.Align.CENTER)

        self._add_button_header = Gtk.Button(label=_("Add Files"))
        self._add_button_header.add_css_class("suggested-action")
        get_tooltip_helper().add_tooltip(self._add_button_header, _("Insert PDF or image files"))
        set_a11y_label(self._add_button_header, _("Add files"))
        self._add_button_header.connect("clicked", self._on_add_files_clicked)
        center_actions_box.append(self._add_button_header)

        self._apply_button = Gtk.Button(
            label=_("Save As") if self._standalone else _("Apply"),
        )
        self._apply_button.add_css_class("suggested-action")
        self._apply_button.set_margin_start(12)
        if self._standalone:
            get_tooltip_helper().add_tooltip(
                self._apply_button, _("Save PDF to a new file (Ctrl+S)")
            )
            set_a11y_label(self._apply_button, _("Save PDF to a new file"))
            self._apply_button.connect("clicked", self._on_save_as_clicked)
        else:
            get_tooltip_helper().add_tooltip(
                self._apply_button, _("Save changes and go back (Ctrl+S)")
            )
            set_a11y_label(self._apply_button, _("Save changes and go back"))
            self._apply_button.connect("clicked", self._on_ok_clicked)
        center_actions_box.append(self._apply_button)

        content_header.set_title_widget(center_actions_box)

        # Menu Button
        self.menu_button = Gtk.MenuButton()
        self.menu_button.set_icon_name("open-menu-symbolic")
        get_tooltip_helper().add_tooltip(self.menu_button, _("Menu"))
        set_a11y_label(self.menu_button, _("Menu"))

        menu_model = Gio.Menu.new()
        menu_model.append(_("Help"), "editor.help")
        menu_model.append(_("About"), "app.about")
        menu_model.append(_("Quit"), "app.quit")
        self.menu_button.set_menu_model(menu_model)

        content_header.pack_end(self.menu_button)

        content_toolbar.add_top_bar(content_header)

        # Page Grid + Notification Banner
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

        self._notification_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._notification_box.add_css_class("editor-notification")
        self._notification_box.set_margin_top(8)
        self._notification_box.set_accessible_role(Gtk.AccessibleRole.ALERT)

        self._notification_icon = Gtk.Image()
        self._notification_icon.set_icon_size(Gtk.IconSize.NORMAL)
        self._notification_box.append(self._notification_icon)

        self._notification_label = Gtk.Label()
        self._notification_label.set_wrap(True)
        self._notification_label.set_max_width_chars(60)
        self._notification_box.append(self._notification_label)

        self._notification_revealer.set_child(self._notification_box)
        content_overlay.add_overlay(self._notification_revealer)

        content_toolbar.set_content(content_overlay)

        # Hidden live-region label for screen reader announcements
        self._a11y_status = Gtk.Label(accessible_role=Gtk.AccessibleRole.STATUS)
        self._a11y_status.set_visible(False)

        self._status_bar = self._create_status_bar()
        content_toolbar.add_bottom_bar(self._status_bar)

        return content_toolbar

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
        self._filename_label = Gtk.Label(label=self._pdf_path)
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
        set_a11y_label(self._selection_label, _("Selection count"))
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
            "rotate-left": self._on_rotate_left,
            "rotate-right": self._on_rotate_right,
            "flip-horizontal": self._on_flip_horizontal,
            "flip-vertical": self._on_flip_vertical,
            "help": self._on_show_help,
        }

        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            action_group.add_action(action)

        self.insert_action_group("editor", action_group)

    _EDITOR_HELP_CONFIG = os.path.join(
        os.path.expanduser("~/.config/bigocrpdf"), "show_editor_help"
    )

    def _should_show_editor_help(self) -> bool:
        """Check if the editor help dialog should be shown on open."""
        if not os.path.exists(self._EDITOR_HELP_CONFIG):
            return True
        try:
            with open(self._EDITOR_HELP_CONFIG) as f:
                return f.read().strip().lower() == "true"
        except Exception:
            return True

    def _set_show_editor_help(self, show: bool) -> None:
        """Persist editor help preference."""
        try:
            os.makedirs(os.path.dirname(self._EDITOR_HELP_CONFIG), exist_ok=True)
            with open(self._EDITOR_HELP_CONFIG, "w") as f:
                f.write("true" if show else "false")
        except Exception as e:
            logger.error(f"Error saving editor help preference: {e}")

    def _on_show_help(self, *_args) -> None:
        """Show the PDF editor help dialog."""
        dialog = Adw.Dialog()
        dialog.set_title(_("PDF Editor Help"))
        dialog.set_content_width(520)
        dialog.set_content_height(480)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(36)
        content.set_margin_end(36)

        from bigocrpdf.config import APP_ICON_NAME

        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(48)
        icon.set_margin_bottom(12)
        icon.set_halign(Gtk.Align.CENTER)
        icon.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
        content.append(icon)

        title = Gtk.Label()
        title.set_markup(f"<span size='large' weight='bold'>{_('PDF Editor')}</span>")
        title.set_halign(Gtk.Align.CENTER)
        title.set_margin_bottom(14)
        content.append(title)

        desc = Gtk.Label()
        desc.set_text(
            _(
                "Use the PDF editor to view and organize your documents. "
                "You can rearrange, rotate, or remove pages, compress files, "
                "and split documents by pages or file size."
            )
        )
        desc.set_wrap(True)
        desc.set_justify(Gtk.Justification.LEFT)
        desc.set_halign(Gtk.Align.START)
        desc.set_margin_bottom(16)
        desc.set_max_width_chars(55)
        content.append(desc)

        shortcuts_label = Gtk.Label()
        shortcuts_label.set_markup("<span weight='bold'>" + _("Keyboard shortcuts:") + "</span>")
        shortcuts_label.set_halign(Gtk.Align.START)
        shortcuts_label.set_margin_bottom(8)
        content.append(shortcuts_label)

        shortcuts = [
            ("Ctrl+Z", _("Undo last action")),
            ("Ctrl+A", _("Select all pages")),
            ("Ctrl+S", _("Save and close")),
            ("Delete", _("Remove selected pages")),
            ("Ctrl+L / Ctrl+R", _("Rotate left / right")),
            ("+  /  −", _("Zoom in / out")),
        ]

        shortcuts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        shortcuts_box.set_margin_bottom(16)
        for key, action in shortcuts:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            key_label = Gtk.Label()
            key_label.set_markup(f"<tt>{key}</tt>")
            key_label.set_xalign(0)
            key_label.set_size_request(140, -1)
            row.append(key_label)
            action_label = Gtk.Label(label=action)
            action_label.set_xalign(0)
            row.append(action_label)
            shortcuts_box.append(row)
        content.append(shortcuts_box)

        tips_label = Gtk.Label()
        tips_label.set_markup("<span weight='bold'>" + _("Tips:") + "</span>")
        tips_label.set_halign(Gtk.Align.START)
        tips_label.set_margin_bottom(8)
        content.append(tips_label)

        tips = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        tip_items = [
            _("Drag and drop pages to reorder them"),
            _("Right-click a page to save it as image or PDF"),
            _("Drag external files onto the editor to add them"),
            _("Use the menu to compress or split documents"),
        ]
        for tip in tip_items:
            lbl = Gtk.Label()
            lbl.set_markup(f"• {tip}")
            lbl.set_wrap(True)
            lbl.set_halign(Gtk.Align.START)
            lbl.set_xalign(0)
            tips.append(lbl)
        content.append(tips)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(12)
        separator.set_margin_bottom(16)
        content.append(separator)

        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        startup_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        startup_row.set_halign(Gtk.Align.FILL)
        startup_label = Gtk.Label(label=_("Show when opening the editor"))
        startup_label.set_halign(Gtk.Align.START)
        startup_label.set_hexpand(True)
        startup_switch = Gtk.Switch()
        startup_switch.set_active(self._should_show_editor_help())
        startup_switch.set_valign(Gtk.Align.CENTER)
        startup_switch.set_halign(Gtk.Align.END)
        set_a11y_label(startup_switch, _("Show when opening the editor"))
        startup_switch.connect(
            "notify::active", lambda sw, _p: self._set_show_editor_help(sw.get_active())
        )
        startup_row.append(startup_label)
        startup_row.append(startup_switch)
        bottom.append(startup_row)

        close_btn = Gtk.Button(label=_("Got it"))
        close_btn.add_css_class("suggested-action")
        close_btn.add_css_class("pill")
        close_btn.set_size_request(140, 36)
        close_btn.set_halign(Gtk.Align.CENTER)
        set_a11y_label(close_btn, _("Got it"))
        close_btn.connect("clicked", lambda _: dialog.close())
        bottom.append(close_btn)

        content.append(bottom)

        dialog.set_child(content)
        dialog.set_follows_content_size(True)
        dialog.present(self)

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

    def _on_selection_changed(self, grid: PageGrid) -> None:
        """Handle selection changes in the grid.

        Args:
            grid: The page grid
        """
        self._update_status_bar()
        count = len(grid._selected_indices)
        if count > 0:
            self._a11y_status.set_text(_("{count} pages selected").format(count=count))

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
        n_items = self._zoom_dropdown.get_model().get_n_items()
        new_idx = max(0, min(n_items - 1, current + direction))
        if new_idx != current:
            self._zoom_dropdown.set_selected(new_idx)

    def _on_ok_clicked(self, _button: Gtk.Button) -> None:
        """Handle OK button click - apply changes and close.

        Args:
            _button: The button widget
        """
        self._save_and_callback()
        self._close_window()

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
            dialog.set_initial_folder(Gio.File.new_for_path(os.path.dirname(self._pdf_path)))
        else:
            dialog.set_initial_name(_("document.pdf"))

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

            from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

            if apply_changes_to_pdf(self._document, dest_path):
                self._document.clear_modifications()
                self._show_notification(_("Saved: {}").format(os.path.basename(dest_path)))
                logger.info("Saved PDF via Save As: %s", dest_path)
            else:
                self._show_error(_("Failed to save PDF."))
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error(f"Save As error: {e}")

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
        from bigocrpdf.utils.temp_manager import mkstemp as _mkstemp

        # Tracked temp file — cleaned up after OCR or on exit
        fd, temp_path = _mkstemp(suffix=".pdf", prefix="bigocr_merge_")
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
        """Handle window close request."""
        self._save_editor_window_size()
        if self._document and self._document.modified:
            self._maybe_save_and_close()
            return True
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
