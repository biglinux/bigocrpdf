"""
BigOcrPdf - Header Bar Module

Custom header bar with action buttons for file management and OCR processing.
Design inspired by Big Video Converter for consistency.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.i18n import _


class HeaderBar(Gtk.Box):
    """
    Custom header bar with action buttons for file management and OCR conversion.
    Follows the design pattern from Big Video Converter for consistency.
    """

    def __init__(self, window, window_buttons_left: bool = False):
        """Initialize the header bar.

        Args:
            window: Reference to the main application window
            window_buttons_left: Whether window buttons are on the left side
        """
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.window = window
        self.window_buttons_left = window_buttons_left

        # Ensure the Box fills the entire width
        self.set_hexpand(True)

        # Create the header bar
        self.header_bar = Adw.HeaderBar()
        self.header_bar.set_hexpand(True)

        # Configure decoration layout based on window button position
        if window_buttons_left:
            self.header_bar.set_decoration_layout("")
        else:
            self.header_bar.set_decoration_layout("menu:minimize,maximize,close")

        self.append(self.header_bar)

        # Add Back button at the start (hidden by default)
        self.back_button = Gtk.Button()
        self.back_button.add_css_class("suggested-action")
        back_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        back_icon = Gtk.Image.new_from_icon_name("go-previous-symbolic")
        back_label = Gtk.Label(label=_("Back"))
        back_box.append(back_icon)
        back_box.append(back_label)
        self.back_button.set_child(back_box)
        self.back_button.connect("clicked", self._on_back_clicked)
        self.back_button.set_visible(False)
        self.header_bar.pack_start(self.back_button)

        # Create left controls box for queue info
        left_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        left_controls.set_margin_start(14)
        left_controls.set_halign(Gtk.Align.START)

        # Clear queue icon button (left side)
        self.clear_queue_button = Gtk.Button()
        self.clear_queue_button.set_icon_name("trash-symbolic")
        self.clear_queue_button.set_tooltip_text(_("Remove all files from the list"))
        self.clear_queue_button.add_css_class("circular")
        self.clear_queue_button.add_css_class("destructive-action")
        self.clear_queue_button.connect("clicked", self._on_clear_queue_clicked)
        self.clear_queue_button.set_visible(False)
        left_controls.append(self.clear_queue_button)

        # Queue size label (left side)
        self.queue_size_label = Gtk.Label(label=_("0 files"))
        self.queue_size_label.add_css_class("caption")
        self.queue_size_label.add_css_class("dim-label")
        self.queue_size_label.set_visible(False)
        self.queue_size_label.set_margin_start(4)
        self.queue_size_label.set_margin_end(8)
        self.queue_size_label.set_valign(Gtk.Align.CENTER)
        left_controls.append(self.queue_size_label)

        self.header_bar.pack_start(left_controls)

        # Create action buttons container for title area
        self.action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.action_box.set_halign(Gtk.Align.CENTER)
        self.action_box.set_spacing(6)

        # Add Files button (simple button — no folder option)
        self.add_button = Gtk.Button(label=_("Add Files"))
        self.add_button.add_css_class("suggested-action")
        self.add_button.connect("clicked", self._on_add_files_clicked)

        self.action_box.append(self.add_button)

        # Start OCR button (shown when files exist)
        self.start_button = Gtk.Button(label=_("Start OCR"))
        self.start_button.add_css_class("suggested-action")
        self.start_button.set_margin_start(12)
        self.start_button.connect("clicked", self._on_start_clicked)
        self.start_button.set_visible(False)
        self.action_box.append(self.start_button)

        # Start OCR for Current File button (for editor view)
        self.start_current_button = Gtk.Button(label=_("Start OCR"))
        self.start_current_button.add_css_class("suggested-action")
        self.start_current_button.connect("clicked", self._on_start_current_clicked)
        self.start_current_button.set_visible(False)
        self.action_box.append(self.start_current_button)

        # Set action box as title widget
        self.header_bar.set_title_widget(self.action_box)

        # Add menu button (three dots) at the end
        self.menu_button = Gtk.MenuButton()
        self.menu_button.set_icon_name("open-menu-symbolic")
        self.menu_button.set_tooltip_text(_("Menu"))

        # Create menu model
        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Help"), "win.help")
        menu.append(_("About"), "app.about")
        menu.append(_("Quit"), "app.quit")

        self.menu_button.set_menu_model(menu)

        # Add app icon if window buttons on left
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

    def update_queue_size(self, count: int) -> None:
        """Update queue size label and button visibility.

        Args:
            count: Number of files in the queue
        """
        # Update label text
        if count == 1:
            text = _("1 file")
        else:
            text = _("{} files").format(count)
        self.queue_size_label.set_text(text)

        # Show clear button and label only when there are 2+ files
        has_multiple_files = count >= 2
        self.clear_queue_button.set_visible(has_multiple_files)
        self.queue_size_label.set_visible(has_multiple_files)

        # Show Start OCR button when there are files
        has_files = count > 0
        self.start_button.set_visible(has_files)

    def set_view(self, view_name: str) -> None:
        """Set the header bar context based on current view.

        Args:
            view_name: 'queue', 'editor', 'processing', or 'complete'
        """
        if view_name == "queue":
            # Show file management buttons
            self.add_button.set_visible(True)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(False)

            # Restore button visibility based on queue size
            if hasattr(self.window, "settings") and self.window.settings:
                queue_count = len(self.window.settings.selected_files)
                has_multiple_files = queue_count >= 2
                has_files = queue_count > 0
                self.clear_queue_button.set_visible(has_multiple_files)
                self.queue_size_label.set_visible(has_multiple_files)
                self.start_button.set_visible(has_files)
                self.start_button.set_sensitive(True)
            else:
                self.start_button.set_visible(False)

        elif view_name == "editor":
            # Hide file management buttons
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            # Show editor buttons
            self.start_current_button.set_visible(True)
            self.back_button.set_visible(True)
            self.start_current_button.set_sensitive(True)

        elif view_name == "processing":
            # Hide all action buttons during processing
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(False)

        elif view_name == "complete":
            # Show back button to return to settings
            self.add_button.set_visible(False)
            self.clear_queue_button.set_visible(False)
            self.queue_size_label.set_visible(False)
            self.start_button.set_visible(False)
            self.start_current_button.set_visible(False)
            self.back_button.set_visible(True)
