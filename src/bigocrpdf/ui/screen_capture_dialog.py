"""
BigOcrPdf - Screen Capture Dialog Module

This module provides a dialog for displaying extracted text from screen captures.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ScreenCaptureResultDialog(Adw.Window):
    """Dialog to display OCR results from screen capture."""

    def __init__(self, parent: Gtk.Window, text: str | None = None) -> None:
        """Initialize the screen capture result dialog.

        Args:
            parent: Parent window
            text: The extracted text to display, or None for loading state
        """
        super().__init__()

        self.set_title(_("Extracted Text"))
        self.set_default_size(550, 450)
        self.set_modal(True)
        self.set_transient_for(parent)
        self.set_resizable(True)

        self._text = text
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        # Use ToolbarView for better layout management
        toolbar_view = Adw.ToolbarView()

        # Header bar
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        toolbar_view.add_top_bar(header)

        # Content container
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(8)
        content_box.set_margin_bottom(24)
        content_box.set_vexpand(True)
        content_box.set_hexpand(True)

        # Loading Stack
        self._stack = Gtk.Stack()
        self._stack.set_vexpand(True)
        self._stack.set_hexpand(True)
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        content_box.append(self._stack)

        # --- Loading Page ---
        loading_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        loading_box.set_halign(Gtk.Align.CENTER)
        loading_box.set_valign(Gtk.Align.CENTER)
        loading_box.set_vexpand(True)
        loading_box.set_hexpand(True)

        spinner = Gtk.Spinner()
        spinner.set_size_request(48, 48)
        spinner.start()
        loading_box.append(spinner)

        loading_label = Gtk.Label(label=_("Processing extracted text..."))
        loading_label.add_css_class("title-4")
        loading_box.append(loading_label)

        self._stack.add_named(loading_box, "loading")

        # --- Results Page ---
        results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        results_box.set_vexpand(True)
        results_box.set_hexpand(True)

        # Title
        title_label = Gtk.Label()
        title_label.set_markup(
            f"<span size='large' weight='bold'>{_('Text Extracted from Screen')}</span>"
        )
        title_label.set_halign(Gtk.Align.START)
        results_box.append(title_label)

        # Subtitle
        subtitle_label = Gtk.Label()
        subtitle_label.set_text(_("The following text was extracted from the selected region:"))
        subtitle_label.set_halign(Gtk.Align.START)
        subtitle_label.add_css_class("dim-label")
        results_box.append(subtitle_label)

        # Text view in a scrolled window with frame
        frame = Gtk.Frame()
        frame.set_vexpand(True)
        frame.set_hexpand(True)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(200)
        scroll.set_vexpand(True)
        scroll.set_hexpand(True)

        self._text_view = Gtk.TextView()
        self._text_view.set_editable(False)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._text_view.set_left_margin(12)
        self._text_view.set_right_margin(12)
        self._text_view.set_top_margin(12)
        self._text_view.set_bottom_margin(12)
        self._text_view.set_monospace(True)

        # Buffer will be set when text is available
        self._text_buffer = self._text_view.get_buffer()

        scroll.set_child(self._text_view)
        frame.set_child(scroll)
        results_box.append(frame)

        # Button box (part of results page)
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        button_box.set_halign(Gtk.Align.END)
        button_box.set_margin_top(8)

        # Copy button
        copy_button = Gtk.Button()
        copy_button.set_label(_("Copy to Clipboard"))
        copy_button.set_icon_name("edit-copy-symbolic")
        copy_button.add_css_class("suggested-action")
        copy_button.connect("clicked", self._on_copy_clicked)
        button_box.append(copy_button)

        # New capture button
        new_capture_button = Gtk.Button()
        new_capture_button.set_label(_("New Capture"))
        new_capture_button.set_icon_name("camera-photo-symbolic")
        new_capture_button.connect("clicked", self._on_new_capture_clicked)
        button_box.append(new_capture_button)

        results_box.append(button_box)

        self._stack.add_named(results_box, "results")

        toolbar_view.set_content(content_box)
        self.set_content(toolbar_view)

        # Set text if available
        if self._text:
            self._text_buffer.set_text(self._text)
            self._stack.set_visible_child_name("results")
        else:
            self._stack.set_visible_child_name("loading")

    def set_text(self, text: str) -> None:
        """Update the displayed text and switch to results view.

        Args:
            text: Extracted text
        """
        self._text = text
        self._text_buffer.set_text(text)
        self._stack.set_visible_child_name("results")

    def _on_copy_clicked(self, button: Gtk.Button) -> None:
        """Handle copy button click - copies text to clipboard.

        Args:
            button: The button that was clicked
        """
        try:
            clipboard = Gdk.Display.get_default().get_clipboard()
            clipboard.set_text(self._text)

            # Show feedback by temporarily changing button text
            original_label = button.get_label()
            button.set_label(_("Copied!"))
            button.set_sensitive(False)

            def restore_button():
                button.set_label(original_label)
                button.set_sensitive(True)
                return False  # Don't repeat

            GLib.timeout_add(1500, restore_button)

            logger.info("Text copied to clipboard")

        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")

    def _on_new_capture_clicked(self, button: Gtk.Button) -> None:
        """Handle new capture button click.

        Args:
            button: The button that was clicked
        """
        self.close()

        # Emit a custom signal to trigger new capture
        # The parent window should handle this
        if self.get_transient_for():
            parent = self.get_transient_for()
            if hasattr(parent, "on_screen_capture_clicked"):
                # Small delay to allow dialog to close
                GLib.timeout_add(300, parent.on_screen_capture_clicked, None)


class ScreenCaptureErrorDialog(Adw.AlertDialog):
    """Dialog to display screen capture errors."""

    def __init__(self, error_message: str) -> None:
        """Initialize the error dialog.

        Args:
            error_message: The error message to display
        """
        super().__init__()

        self.set_heading(_("Screen Capture Failed"))
        self.set_body(error_message)
        self.add_response("close", _("Close"))
        self.set_default_response("close")
        self.set_close_response("close")


def show_screen_capture_result(parent: Gtk.Window, text: str | None, error: str | None) -> None:
    """Show the appropriate dialog based on the capture result.

    Args:
        parent: Parent window
        text: Extracted text or None
        error: Error message or None
    """
    if error:
        if "cancelled" in error.lower():
            # User cancelled - no dialog needed
            logger.info("Screen capture cancelled by user")
            return

        dialog = ScreenCaptureErrorDialog(error)
        dialog.present(parent)
    elif text:
        dialog = ScreenCaptureResultDialog(parent, text)
        dialog.present()
    else:
        dialog = ScreenCaptureErrorDialog(_("No text could be extracted from the image"))
        dialog.present(parent)
