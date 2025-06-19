"""
BigOcrPdf - Welcome Dialog

This module contains the welcome dialog implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from config import APP_ICON_NAME, CONFIG_DIR
from utils.logger import logger
from utils.i18n import _


class WelcomeDialog:
    """Welcome dialog for BigOcrPdf application"""

    # Configuration file to store welcome dialog preference
    WELCOME_DIALOG_CONFIG = os.path.join(CONFIG_DIR, "show_welcome_dialog")

    def __init__(self, parent_window: "BigOcrPdfWindow"):
        """Initialize welcome dialog

        Args:
            parent_window: The parent window
        """
        self.parent_window = parent_window
        self.dialog = None

    def should_show_welcome_dialog(self) -> bool:
        """Return True if the welcome dialog should be shown at startup."""
        if not os.path.exists(self.WELCOME_DIALOG_CONFIG):
            with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                f.write("true")
            return True
        with open(self.WELCOME_DIALOG_CONFIG, "r") as f:
            return f.read().strip().lower() == "true"

    def set_show_welcome_dialog(self, show: bool) -> None:
        """Set whether to show the welcome dialog at startup

        Args:
            show: True to show the dialog, False to hide it
        """
        try:
            with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                f.write("true" if show else "false")
            logger.info(f"Set show welcome dialog: {show}")
        except Exception as e:
            logger.error(f"Error setting welcome dialog config: {e}")

    def show_welcome_dialog(self) -> None:
        """Show the welcome dialog with application information"""
        # Create the welcome dialog using Adw.Dialog
        self.dialog = Adw.Dialog()
        self.dialog.set_content_width(640)
        self.dialog.set_content_height(400)

        # Create main container with headerbar
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Create custom headerbar with close button
        headerbar = Adw.HeaderBar()
        headerbar.add_css_class("flat")
        main_box.append(headerbar)

        # Create content box
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content_box.set_margin_bottom(24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)

        # Header with icon and title
        header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        header_box.set_halign(Gtk.Align.CENTER)

        app_icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        app_icon.set_pixel_size(64)
        header_box.append(app_icon)

        title = Gtk.Label()
        title.set_markup(
            "<span size='x-large' weight='bold'>"
            + _("Welcome to Big OCR PDF")
            + "</span>"
        )
        header_box.append(title)
        content_box.append(header_box)

        # Explanation text
        explanation = Gtk.Label()
        explanation.set_wrap(True)
        explanation.set_max_width_chars(60)
        explanation.set_margin_top(12)
        explanation.set_markup(
            _(
                "Big OCR PDF adds optical character recognition to your PDF files, "
                "making them searchable and allowing you to select and copy text "
                "from scanned documents.\n\n"
                "<b>Benefits of using Big OCR PDF:</b>\n\n"
                "• <b>Search</b>: Find text within your scanned PDF documents\n"
                "• <b>Copy Text</b>: Select and copy text from images and scanned documents\n"
                "• <b>Batch Processing</b>: Process multiple files at once for efficiency\n"
                "• <b>Auto Correction</b>: Correct page alignment and rotation automatically\n"
            )
        )
        explanation.set_halign(Gtk.Align.START)
        content_box.append(explanation)

        # Separator before switch
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        content_box.append(separator)

        # Don't show again switch
        switch_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        show_at_startup_switch = Gtk.Switch()
        show_at_startup_switch.set_active(self.should_show_welcome_dialog())
        show_at_startup_switch.set_valign(Gtk.Align.CENTER)

        switch_label = Gtk.Label(label=_("Show dialog on startup"))
        switch_label.set_xalign(0)
        switch_label.set_hexpand(True)

        switch_box.append(switch_label)
        switch_box.append(show_at_startup_switch)

        content_box.append(switch_box)

        # Close button
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        button_box.set_halign(Gtk.Align.CENTER)

        start_button = Gtk.Button(label=_("Let's Get Started"))
        start_button.add_css_class("suggested-action")

        button_box.append(start_button)
        content_box.append(button_box)

        # Add content to main box
        main_box.append(content_box)

        # Set the main box as dialog content
        self.dialog.set_child(main_box)

        # Connect signals
        def on_switch_toggle(switch, _param):
            self.set_show_welcome_dialog(switch.get_active())

        show_at_startup_switch.connect("notify::active", on_switch_toggle)
        start_button.connect("clicked", lambda _: self.dialog.close())

        # Show the dialog
        self.dialog.present(self.parent_window)
