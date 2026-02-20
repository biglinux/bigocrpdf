"""
BigOcrPdf - Header Bar Module

Custom header bar with action buttons for file management and OCR processing.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _


class HeaderBar(Gtk.Box):
    """Custom header bar wrapping Adw.HeaderBar with OCR action buttons."""

    def __init__(self, window, window_buttons_left: bool = False):
        """Initialize the header bar.

        Args:
            window: Reference to the main application window
            window_buttons_left: Whether window buttons are on the left side
        """
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.window = window
        self.window_buttons_left = window_buttons_left
        self.set_hexpand(True)

        # Inner Adw.HeaderBar
        self.header_bar = Adw.HeaderBar()
        self.header_bar.set_hexpand(True)

        if window_buttons_left:
            self.header_bar.set_decoration_layout("")
        else:
            self.header_bar.set_decoration_layout("menu:minimize,maximize,close")

        self.append(self.header_bar)

        # Sidebar toggle button (visible only when split view is collapsed)
        self.sidebar_toggle = Gtk.ToggleButton()
        self.sidebar_toggle.set_icon_name("sidebar-show-symbolic")
        self.sidebar_toggle.set_tooltip_text(_("Toggle sidebar"))
        set_a11y_label(self.sidebar_toggle, _("Toggle sidebar"))
        self.sidebar_toggle.add_css_class("flat")
        self.sidebar_toggle.set_visible(False)
        self.sidebar_toggle.connect("toggled", self._on_sidebar_toggled)
        self.header_bar.pack_start(self.sidebar_toggle)

        # Back button (hidden by default)
        self.back_button = Gtk.Button()
        self.back_button.add_css_class("suggested-action")
        back_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        back_box.append(Gtk.Image.new_from_icon_name("go-previous-symbolic"))
        back_box.append(Gtk.Label(label=_("Back")))
        self.back_button.set_child(back_box)
        self.back_button.connect("clicked", self._on_back_clicked)
        self.back_button.set_visible(False)
        set_a11y_label(self.back_button, _("Back"))
        self.header_bar.pack_start(self.back_button)

        # Queue controls (left side)
        left_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        left_controls.set_margin_start(14)
        left_controls.set_halign(Gtk.Align.START)

        self.clear_queue_button = Gtk.Button()
        self.clear_queue_button.set_icon_name("trash-symbolic")
        self.clear_queue_button.set_tooltip_text(_("Remove all files from the list"))
        set_a11y_label(self.clear_queue_button, _("Remove all files from the list"))
        self.clear_queue_button.add_css_class("circular")
        self.clear_queue_button.add_css_class("destructive-action")
        self.clear_queue_button.connect("clicked", self._on_clear_queue_clicked)
        self.clear_queue_button.set_visible(False)
        left_controls.append(self.clear_queue_button)

        self.queue_size_label = Gtk.Label(label=_("0 files"))
        self.queue_size_label.add_css_class("caption")
        self.queue_size_label.add_css_class("dim-label")
        self.queue_size_label.set_visible(False)
        self.queue_size_label.set_margin_start(4)
        self.queue_size_label.set_margin_end(8)
        self.queue_size_label.set_valign(Gtk.Align.CENTER)
        left_controls.append(self.queue_size_label)

        self.header_bar.pack_start(left_controls)

        # Action buttons in the title area
        self.action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.action_box.set_halign(Gtk.Align.CENTER)
        self.action_box.set_spacing(6)

        self.add_button = Gtk.Button(label=_("Add Files"))
        self.add_button.add_css_class("suggested-action")
        self.add_button.connect("clicked", self._on_add_files_clicked)
        set_a11y_label(self.add_button, _("Add Files"))
        self.action_box.append(self.add_button)

        self.start_button = Gtk.Button(label=_("Start OCR"))
        self.start_button.add_css_class("suggested-action")
        self.start_button.set_margin_start(12)
        self.start_button.connect("clicked", self._on_start_clicked)
        self.start_button.set_visible(False)
        set_a11y_label(self.start_button, _("Start OCR"))
        self.action_box.append(self.start_button)

        self.start_current_button = Gtk.Button(label=_("Start OCR"))
        self.start_current_button.add_css_class("suggested-action")
        self.start_current_button.connect("clicked", self._on_start_current_clicked)
        self.start_current_button.set_visible(False)
        set_a11y_label(self.start_current_button, _("Start OCR"))
        self.action_box.append(self.start_current_button)

        self.header_bar.set_title_widget(self.action_box)

        # Menu button
        self.menu_button = Gtk.MenuButton()
        self.menu_button.set_icon_name("open-menu-symbolic")
        self.menu_button.set_tooltip_text(_("Menu"))
        set_a11y_label(self.menu_button, _("Menu"))

        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Help"), "win.help")
        menu.append(_("About"), "app.about")
        menu.append(_("Quit"), "app.quit")
        self.menu_button.set_menu_model(menu)

        if self.window_buttons_left:
            icon_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
            icon_box.set_halign(Gtk.Align.END)
            icon_box.set_valign(Gtk.Align.CENTER)
            icon_box.append(self.menu_button)
            app_icon = Gtk.Image.new_from_icon_name("big-ocr-pdf")
            app_icon.set_pixel_size(20)
            app_icon.set_halign(Gtk.Align.END)
            app_icon.set_valign(Gtk.Align.CENTER)
            icon_box.append(app_icon)
            self.header_bar.pack_end(icon_box)
        else:
            self.header_bar.pack_end(self.menu_button)

    # --- Event Handlers ---

    def _on_add_files_clicked(self, button: Gtk.Button) -> None:
        """Handle Add Files button click."""
        if hasattr(self.window, "on_add_file_clicked"):
            self.window.on_add_file_clicked(button)

    def _on_back_clicked(self, button: Gtk.Button) -> None:
        """Handle Back button click."""
        if hasattr(self.window, "on_back_clicked"):
            self.window.on_back_clicked(button)

    def _on_clear_queue_clicked(self, button: Gtk.Button) -> None:
        """Handle Clear Queue button click."""
        if hasattr(self.window, "clear_file_queue"):
            self.window.clear_file_queue()

    def _on_start_clicked(self, button: Gtk.Button) -> None:
        """Handle Start OCR button click."""
        button.set_sensitive(False)
        if hasattr(self.window, "on_apply_clicked"):
            self.window.on_apply_clicked(button)

    def _on_start_current_clicked(self, button: Gtk.Button) -> None:
        """Handle Start OCR for current file button click."""
        button.set_sensitive(False)
        if hasattr(self.window, "start_ocr_current_file"):
            self.window.start_ocr_current_file()

    # --- Public API ---

    def update_queue_size(self, count: int) -> None:
        """Update queue size label and button visibility.

        Args:
            count: Number of files in the queue
        """
        if count == 1:
            text = _("1 file")
        else:
            text = _("{} files").format(count)
        self.queue_size_label.set_text(text)

        has_multiple_files = count >= 2
        self.clear_queue_button.set_visible(has_multiple_files)
        self.queue_size_label.set_visible(has_multiple_files)

        has_files = count > 0
        self.start_button.set_visible(has_files)

    def set_view(self, view_name: str) -> None:
        """Set the header bar context based on current view.

        Args:
            view_name: 'queue', 'editor', 'processing', or 'complete'
        """
        if view_name == "queue":
            self.add_button.set_visible(True)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(False)

            if hasattr(self.window, "settings") and self.window.settings:
                queue_count = len(self.window.settings.selected_files)
                has_multiple_files = queue_count >= 2
                has_files = queue_count > 0
                self.clear_queue_button.set_visible(has_multiple_files)
                self.queue_size_label.set_visible(has_multiple_files)
                self.start_button.set_visible(has_files)
                self.start_button.set_sensitive(True)
                self.start_button.set_label(_("Start OCR"))
            else:
                self.start_button.set_visible(False)

        elif view_name == "editor":
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            self.start_current_button.set_visible(True)
            self.back_button.set_visible(True)
            self.start_current_button.set_sensitive(True)

        elif view_name == "processing":
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(False)

        elif view_name == "complete":
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(True)

    def bind_split_view(self, split_view: Adw.OverlaySplitView) -> None:
        """Bind the sidebar toggle button to an OverlaySplitView.

        Shows the toggle when the split view is collapsed and keeps
        the button state in sync with sidebar visibility.

        Args:
            split_view: The OverlaySplitView to control
        """
        self._split_view = split_view

        split_view.connect("notify::collapsed", self._on_split_view_collapsed_changed)
        split_view.connect("notify::show-sidebar", self._on_split_view_show_sidebar_changed)

    def _on_sidebar_toggled(self, button: Gtk.ToggleButton) -> None:
        """Handle sidebar toggle button click."""
        if hasattr(self, "_split_view"):
            self._split_view.set_show_sidebar(button.get_active())

    def _on_split_view_collapsed_changed(self, split_view: Adw.OverlaySplitView, _param) -> None:
        """Show/hide the sidebar toggle based on collapsed state."""
        collapsed = split_view.get_collapsed()
        self.sidebar_toggle.set_visible(collapsed)
        if not collapsed:
            self.sidebar_toggle.set_active(False)

    def _on_split_view_show_sidebar_changed(self, split_view: Adw.OverlaySplitView, _param) -> None:
        """Keep toggle button in sync with sidebar visibility."""
        showing = split_view.get_show_sidebar()
        if self.sidebar_toggle.get_active() != showing:
            self.sidebar_toggle.set_active(showing)
