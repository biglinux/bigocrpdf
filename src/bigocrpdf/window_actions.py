"""Window Actions, Drag-Drop, Signals and Dialogs Mixin."""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, Gtk

from bigocrpdf.config import APP_ICON_NAME
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class WindowActionsSignalsMixin:
    """Mixin providing window actions, drag-drop, signal management and dialogs."""

    def do_realize(self) -> None:
        """Called when the window is being realized."""
        Adw.ApplicationWindow.do_realize(self)

        if hasattr(self, "stack") and self.stack:
            self.stack.set_visible_child_name("settings")

    def _setup_window_actions(self) -> None:
        """Set up window-level actions for keyboard shortcuts."""
        # Add files action (Ctrl+O)
        add_files_action = Gio.SimpleAction.new("add-files", None)
        add_files_action.connect("activate", self._on_add_files_action)
        self.add_action(add_files_action)

        # Start processing action (Ctrl+Enter)
        start_action = Gio.SimpleAction.new("start-processing", None)
        start_action.connect("activate", self._on_start_processing_action)
        self.add_action(start_action)

        # Cancel processing action (Escape)
        cancel_action = Gio.SimpleAction.new("cancel-processing", None)
        cancel_action.connect("activate", self._on_cancel_processing_action)
        self.add_action(cancel_action)

        # Remove all files action (Ctrl+R)
        remove_all_action = Gio.SimpleAction.new("remove-all-files", None)
        remove_all_action.connect("activate", self._on_remove_all_files_action)
        self.add_action(remove_all_action)

        logger.info("Window actions set up for keyboard shortcuts")

    def _on_add_files_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle add files shortcut (Ctrl+O)."""
        # Only work on settings page
        if self.stack.get_visible_child_name() == "settings":
            self.on_add_file_clicked(None)

    def _on_start_processing_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle start processing shortcut (Ctrl+Enter)."""
        # Only work on settings page with files selected
        if self.stack.get_visible_child_name() == "settings":
            if self.settings.selected_files:
                self.on_apply_clicked(None)

    def _on_cancel_processing_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle cancel processing shortcut (Escape)."""
        current_page = self.main_stack.get_visible_child_name()
        if current_page == "terminal":
            self.on_cancel_clicked()

    def _on_remove_all_files_action(self, _action: Gio.SimpleAction, _param) -> None:
        """Handle remove all files shortcut (Ctrl+R)."""
        # Only work on settings page
        if self.stack.get_visible_child_name() == "settings":
            if hasattr(self.ui, "settings_page_manager") and self.ui.settings_page_manager:
                self.ui.settings_page_manager._remove_all_files()

    def _setup_global_drag_drop(self) -> None:
        """Set up global drag and drop for the entire window."""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList])
        drop_target.connect("drop", self._on_global_drop)
        drop_target.connect("enter", self._on_global_drop_enter)
        drop_target.connect("leave", self._on_global_drop_leave)
        self.add_controller(drop_target)
        logger.info("Global drag & drop enabled")

    def _on_global_drop(self, _drop_target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Handle global file drop on the window.

        Args:
            _drop_target: The drop target controller
            value: The dropped file(s)
            _x: X coordinate
            _y: Y coordinate

        Returns:
            True if drop was handled
        """
        # Only accept drops on the settings page
        if self.stack.get_visible_child_name() != "settings":
            return False

        # Delegate to settings page if it exists
        if hasattr(self.ui, "settings_page_manager") and self.ui.settings_page_manager:
            return self.ui.settings_page_manager._on_drop(_drop_target, value, _x, _y)
        return False

    def _on_global_drop_enter(
        self, _drop_target: Gtk.DropTarget, _x: float, _y: float
    ) -> Gdk.DragAction:
        """Handle drag enter on the window."""
        if self.stack.get_visible_child_name() == "settings":
            return Gdk.DragAction.COPY
        return Gdk.DragAction(0)  # Reject if not on settings page

    def _on_global_drop_leave(self, _drop_target: Gtk.DropTarget) -> None:
        """Handle drag leave from the window."""
        pass  # No visual feedback needed

    def _window_buttons_on_left(self) -> bool:
        """Detect if window buttons (close/min/max) are on the left side."""
        try:
            settings = Gio.Settings.new("org.gnome.desktop.wm.preferences")
            layout = settings.get_string("button-layout")
            if layout and ":" in layout:
                left, right = layout.split(":", 1)
                if "close" in left:
                    return True
                if "close" in right:
                    return False
            elif layout:
                if "close" in layout:
                    return False
        except Exception as e:
            logger.debug(f"Could not detect window button layout: {e}")
        return False

    def clear_file_queue(self) -> None:
        """Clear all files from the queue."""
        self.settings.selected_files.clear()
        self.ui.refresh_queue_status()
        self.custom_header_bar.update_queue_size(0)

    def should_show_welcome_dialog(self) -> bool:
        """Check if the welcome dialog should be shown at startup.

        Returns:
            True if dialog should be shown
        """
        if not os.path.exists(self.WELCOME_DIALOG_CONFIG):
            try:
                with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                    f.write("true")
                return True
            except Exception as e:
                logger.error(f"Error creating welcome dialog config: {e}")
                return True

        try:
            with open(self.WELCOME_DIALOG_CONFIG) as f:
                content = f.read().strip()
                return content.lower() == "true"
        except Exception as e:
            logger.error(f"Error reading welcome dialog config: {e}")
            return True

    def set_show_welcome_dialog(self, show: bool) -> None:
        """Set whether to show the welcome dialog at startup.

        Args:
            show: Whether to show the dialog
        """
        try:
            with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                f.write("true" if show else "false")
            logger.info(f"Set show welcome dialog: {show}")
        except Exception as e:
            logger.error(f"Error setting welcome dialog config: {e}")

    def show_welcome_dialog(self) -> None:
        """Show the welcome dialog as a centered modal."""
        dialog = Adw.Dialog()
        dialog.set_title("Big OCR PDF")
        dialog.set_content_width(650)
        dialog.set_content_height(500)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content_box.set_margin_top(24)
        content_box.set_margin_bottom(24)
        content_box.set_margin_start(36)
        content_box.set_margin_end(36)

        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(64)
        icon.set_margin_bottom(16)
        icon.set_halign(Gtk.Align.CENTER)
        content_box.append(icon)

        what_is = Gtk.Label()
        what_is.set_markup(f"<span size='large' weight='bold'>{_('What is')} Big OCR PDF?</span>")
        what_is.set_halign(Gtk.Align.CENTER)
        what_is.set_margin_bottom(14)
        content_box.append(what_is)

        what_is_desc = Gtk.Label()
        what_is_desc.set_markup(
            _(
                "Big OCR PDF adds optical character recognition to your PDF files, "
                "making them searchable and allowing you to select and copy text "
                "from scanned documents."
            )
        )
        what_is_desc.set_wrap(True)
        what_is_desc.set_justify(Gtk.Justification.LEFT)
        what_is_desc.set_halign(Gtk.Align.START)
        what_is_desc.set_margin_bottom(16)
        what_is_desc.set_max_width_chars(65)
        content_box.append(what_is_desc)

        benefits = Gtk.Label()
        benefits.set_markup(
            "<span weight='bold'>" + _("Benefits of using Big OCR PDF:") + "</span>"
        )
        benefits.set_halign(Gtk.Align.START)
        benefits.set_margin_bottom(8)
        content_box.append(benefits)

        benefits_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        benefits_list.set_halign(Gtk.Align.START)
        benefits_list.set_margin_bottom(16)

        benefit_items = [
            (_("Search"), _("Search through your scanned PDF documents")),
            (_("Copy text"), _("Copy text from images and scanned documents")),
            (_("Images"), _("Add image files and generate new PDFs from them")),
            (_("Edit PDF"), _("Rearrange, rotate or remove pages before processing")),
            (_("Batch processing"), _("Process multiple files at once")),
            (
                _("Auto-correction"),
                _("Automatically correct page alignment and rotation"),
            ),
        ]

        for title, description in benefit_items:
            benefit_label = Gtk.Label()
            benefit_label.set_markup(f"<span>• <b>{title}:</b> {description}</span>")
            benefit_label.set_wrap(True)
            benefit_label.set_halign(Gtk.Align.START)
            benefit_label.set_xalign(0)
            benefit_label.set_margin_bottom(3)

            benefits_list.append(benefit_label)

        content_box.append(benefits_list)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(4)
        separator.set_margin_bottom(16)
        content_box.append(separator)

        bottom_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        show_at_startup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        show_at_startup_box.set_halign(Gtk.Align.FILL)

        show_at_startup_label = Gtk.Label(label=_("Show this dialog at startup"))
        show_at_startup_label.set_halign(Gtk.Align.START)
        show_at_startup_label.set_hexpand(True)

        show_at_startup_switch = Gtk.Switch()
        show_at_startup_switch.set_active(self.should_show_welcome_dialog())
        show_at_startup_switch.set_valign(Gtk.Align.CENTER)
        show_at_startup_switch.set_halign(Gtk.Align.END)
        set_a11y_label(show_at_startup_switch, _("Show this dialog at startup"))

        show_at_startup_box.append(show_at_startup_label)
        show_at_startup_box.append(show_at_startup_switch)
        bottom_section.append(show_at_startup_box)

        start_button = Gtk.Button()
        start_button.set_label(_("Let's Get Started"))
        start_button.add_css_class("suggested-action")
        start_button.add_css_class("pill")
        start_button.set_size_request(160, 36)
        start_button.set_halign(Gtk.Align.CENTER)
        set_a11y_label(start_button, _("Let's Get Started"))
        bottom_section.append(start_button)

        content_box.append(bottom_section)

        # Adw.Dialog provides its own header bar, just set content directly
        dialog.set_child(content_box)
        dialog.set_follows_content_size(True)

        def on_switch_toggle(switch, _param):
            self.set_show_welcome_dialog(switch.get_active())

        show_at_startup_switch.connect("notify::active", on_switch_toggle)
        start_button.connect("clicked", lambda _: dialog.close())

        dialog.present(self)

    def _confirm_reset_settings(self) -> None:
        """Show a confirmation dialog before resetting all settings to defaults."""
        dialog = Adw.AlertDialog(
            heading=_("Reset All Settings?"),
            body=_(
                "This will restore all options to their default values. "
                "Your file queue will not be affected."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("reset", _("Reset"))
        dialog.set_response_appearance("reset", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_reset_settings_response)
        dialog.present(self)

    def _on_reset_settings_response(self, _dialog: Adw.AlertDialog, response: str) -> None:
        """Handle the reset settings confirmation response."""
        if response != "reset":
            return

        # Save current file queue before reset
        current_files = list(self.settings.selected_files)
        original_paths = dict(self.settings.original_file_paths)

        # Reset all settings
        self.settings.reset_to_defaults()

        # Restore file queue
        self.settings.selected_files = current_files
        self.settings.original_file_paths = original_paths

        # Refresh the UI to reflect new defaults
        if hasattr(self, "ui") and hasattr(self.ui, "settings_page"):
            self.ui.settings_page.sync_ui_to_settings()

        logger.info("Settings reset to defaults via menu")
        self.show_toast(_("Settings restored to defaults"))

    def check_resumable_session(self) -> None:
        """Check for and offer to resume an incomplete processing session.

        This should be called after the window is presented to check if
        there's a previous session that was interrupted.
        """
        if self.ocr_processor.has_resumable_session():
            session_info = self.ocr_processor.get_resumable_session_info()
            if session_info:
                logger.info(
                    f"Found incomplete session with {session_info.get('pending_files', 0)} "
                    "pending files"
                )
                self.ui.dialogs_manager.show_resume_session_dialog(
                    session_info,
                    on_resume=self._on_resume_session,
                    on_discard=self._on_discard_session,
                )

    def _on_resume_session(self) -> None:
        """Handle user choosing to resume a previous session."""
        if self.ocr_processor.resume_previous_session():
            # Update UI with the resumed files
            if hasattr(self.ui, "settings_page_manager") and self.ui.settings_page_manager:
                self.ui.settings_page_manager._populate_file_list()
            # Show Start OCR button by updating header bar queue count
            file_count = len(self.settings.selected_files)
            if hasattr(self, "custom_header_bar") and self.custom_header_bar:
                self.custom_header_bar.update_queue_size(file_count)
            self.show_toast(_("Session resumed with {0} files").format(file_count))
            logger.info("User chose to resume previous session")
        else:
            logger.warning("Failed to resume session")
            self.show_toast(_("Could not resume session"))

    def _on_discard_session(self) -> None:
        """Handle user choosing to discard a previous session."""
        self.ocr_processor.discard_previous_session()
        logger.info("User discarded previous incomplete session")
        self.show_toast(_("Previous session discarded"))
