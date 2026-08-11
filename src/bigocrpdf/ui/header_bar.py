"""
BigOcrPdf - Header Bar Module

Custom header bar with action buttons for file management and OCR processing.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _, ngettext


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

        # Queue controls (left side)
        left_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        left_controls.set_margin_start(14)
        left_controls.set_halign(Gtk.Align.START)

        self.clear_queue_button = Gtk.Button()
        self.clear_queue_button.set_icon_name("user-trash-symbolic")
        self.clear_queue_button.set_tooltip_text(_("Remove all files from the list"))
        set_a11y_label(self.clear_queue_button, _("Remove all files from the list"))
        self.clear_queue_button.add_css_class("circular")
        self.clear_queue_button.add_css_class("destructive-action")
        self.clear_queue_button.connect("clicked", self._on_clear_queue_clicked)
        self.clear_queue_button.set_visible(False)
        left_controls.append(self.clear_queue_button)

        self.queue_size_label = Gtk.Label(
            label=ngettext("{count} file", "{count} files", 0).format(count=0)
        )
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
        self._apply_ocr_availability_to_button(self.start_button)
        self.action_box.append(self.start_button)

        self.header_bar.set_title_widget(self.action_box)

        # View toggle button (single icon, switches between list/grid)
        self._is_grid_view = False
        self.view_toggle_button = Gtk.Button()
        self.view_toggle_button.set_icon_name("view-list-symbolic")
        self.view_toggle_button.set_tooltip_text(_("List view (click for grid)"))
        set_a11y_label(self.view_toggle_button, _("List view (click for grid)"))
        self.view_toggle_button.add_css_class("flat")
        self.view_toggle_button.set_visible(False)
        self.view_toggle_button.connect("clicked", self._on_view_toggle_clicked)

        # Menu button
        self.menu_button = Gtk.MenuButton()
        self.menu_button.set_icon_name("open-menu-symbolic")
        self.menu_button.set_tooltip_text(_("Menu"))
        set_a11y_label(self.menu_button, _("Menu"))

        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Keyboard Shortcuts"), "app.shortcuts")
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

        # Pack view toggle next to menu (pack_end adds right-to-left)
        self.header_bar.pack_end(self.view_toggle_button)

    # --- Event Handlers ---

    def _on_add_files_clicked(self, button: Gtk.Button) -> None:
        """Handle Add Files button click."""
        self.window.file_manager.show_open_files_dialog()

    def _on_clear_queue_clicked(self, button: Gtk.Button) -> None:
        """Handle Clear Queue button click with confirmation dialog."""
        n_files = len(self.window.settings.selected_files)
        if n_files == 0:
            return

        dialog = Adw.AlertDialog(
            heading=_("Clear file queue?"),
            body=ngettext(
                "This will remove {count} file from the queue.",
                "This will remove {count} files from the queue.",
                n_files,
            ).format(count=n_files),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("clear", _("Clear"))
        dialog.set_response_appearance("clear", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_clear_queue_response)
        dialog.present(self.window)

    def _on_clear_queue_response(self, dialog: Adw.AlertDialog, response: str) -> None:
        """Handle clear queue confirmation response."""
        if response == "clear":
            self.window.actions.clear_file_queue()

    def _on_view_toggle_clicked(self, button: Gtk.Button) -> None:
        """Toggle between list and grid view."""
        self._is_grid_view = not self._is_grid_view
        if self._is_grid_view:
            button.set_icon_name("view-grid-symbolic")
            button.set_tooltip_text(_("Grid view (click for list)"))
        else:
            button.set_icon_name("view-list-symbolic")
            button.set_tooltip_text(_("List view (click for grid)"))

        # Notify the queue panel
        self.window.ui.settings_page_manager._on_view_mode_toggled(self._is_grid_view)

    def _on_start_clicked(self, button: Gtk.Button) -> None:
        """Handle Start OCR button click."""
        self.window.processing.start(button)

    # --- Public API ---

    def update_queue_size(self, count: int) -> None:
        """Update queue size label and button visibility.

        Args:
            count: Number of files in the queue
        """
        text = ngettext("{count} file", "{count} files", count).format(count=count)
        self.queue_size_label.set_text(text)

        has_multiple_files = count >= 2
        self.clear_queue_button.set_visible(has_multiple_files)
        self.queue_size_label.set_visible(has_multiple_files)

        has_files = count > 0
        self.view_toggle_button.set_visible(has_files)
        self.start_button.set_visible(has_files)
        # The label belongs with the rest of the button's idle state. Restoring
        # only visibility and sensitivity left a hidden button still reading
        # "Starting…" from the run that just ended, which came back into view
        # with that text the moment files were added again.
        self.start_button.set_label(_("Start OCR"))
        self._apply_ocr_availability_to_button(self.start_button)

    def _apply_ocr_availability_to_button(self, button: Gtk.Button) -> None:
        is_available = self.window.ocr_dependency.is_available
        button.set_sensitive(is_available)
        if is_available:
            button.set_tooltip_text(_("Start OCR processing"))
        else:
            button.set_tooltip_text(
                _("OCR is unavailable. Install the required engine and restart the application.")
            )

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
        self._on_split_view_collapsed_changed(split_view, None)
        self._on_split_view_show_sidebar_changed(split_view, None)

    def _on_sidebar_toggled(self, button: Gtk.ToggleButton) -> None:
        """Handle sidebar toggle button click."""
        if hasattr(self, "_split_view"):
            self._split_view.set_show_sidebar(button.get_active())

    def _on_split_view_collapsed_changed(self, split_view: Adw.OverlaySplitView, _param) -> None:
        """Show/hide the sidebar toggle based on collapsed state."""
        collapsed = split_view.get_collapsed()
        self.sidebar_toggle.set_visible(collapsed)
        if collapsed:
            split_view.set_show_sidebar(False)
        else:
            self.sidebar_toggle.set_active(False)

    def _on_split_view_show_sidebar_changed(self, split_view: Adw.OverlaySplitView, _param) -> None:
        """Keep toggle button in sync with sidebar visibility."""
        showing = split_view.get_show_sidebar()
        if self.sidebar_toggle.get_active() != showing:
            self.sidebar_toggle.set_active(showing)
