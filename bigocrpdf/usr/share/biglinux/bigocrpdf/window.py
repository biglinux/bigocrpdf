"""
BigOcrPdf - Window Module

This module contains the main application window implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, GLib

import os
import time

from config import APP_ICON_NAME, CONFIG_DIR
from utils.logger import logger
from services.settings import OcrSettings
from services.processor import OcrProcessor
from ui.ui_manager import BigOcrPdfUI
from utils.i18n import _
from utils.timer import safe_remove_source


class BigOcrPdfWindow(Adw.ApplicationWindow):
    """Main application window for BigOcrPdf"""

    # Configuration dictionaries for dropdowns
    LANGUAGES_CONFIG = "LANGUAGES"  # This will be populated dynamically
    
    QUALITY_CONFIG = [
        ("normal", _("Normal")),
        ("economic", _("Economic")),
        ("economicplus", _("More economic")),
    ]
    
    ALIGNMENT_CONFIG = [
        ("none", _("Don't change")),
        ("align", _("Align")),
        ("rotate", _("Auto rotate")),
        ("alignrotate", _("Align and auto rotate")),
    ]

    # Configuration file to store welcome dialog preference
    WELCOME_DIALOG_CONFIG = os.path.join(CONFIG_DIR, "show_welcome_dialog")

    def __init__(self, app: Adw.Application):
        """Initialize application window

        Args:
            app: The parent Adw.Application instance
        """
        super().__init__(application=app, title=_("Big OCR PDF"))
        self.set_default_size(820, 600)

        # Set up the window icon (either by name or path)
        self.set_icon_name(APP_ICON_NAME)

        # Initialize components
        self.settings = OcrSettings()
        self.ocr_processor = OcrProcessor(self.settings)
        self.ui = BigOcrPdfUI(self)
        
        # Initialize state variables
        self.process_pid = None  # Process ID for OCR operation
        self.processed_files = []  # List to store processed output files
        self.process_start_time = 0
        self.progress_timer_id = None
        self.conclusion_timer_id = None
        
        # Animation state variables
        self._animation_progress = 0.05
        self._animation_direction = 0.01

        # Initialize UI components
        self.stack = None  # ViewStack for main UI transitions
        self.toast_overlay = None  # Toast overlay for notifications
        self.step_label = None  # Step indicator
        self.back_button = None  # Back button
        self.next_button = None  # Next button
        self.header_bar = None  # Header bar reference

        # Signal handler tracking
        self.signal_handlers = {}  # Dictionary to track signal handlers

        # Create the main layout
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the main user interface"""
        # Create the toast overlay for notifications
        self.toast_overlay = Adw.ToastOverlay()

        # Create main containers
        self.setup_stack()
        self.setup_header_bar()
        self.setup_content_area()
        self.setup_action_bar()
        
        # Add pages to the stack
        self.setup_pages()
        
        # Connect signals
        self.stack.connect("notify::visible-child", self._on_page_changed)

        # Set initial view
        self.stack.set_visible_child_name("settings")

        # Set up content with toast overlay for notifications
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(self.toolbar_view)
        main_box.append(self.action_bar)
        
        self.toast_overlay.set_child(main_box)
        self.set_content(self.toast_overlay)

    def setup_stack(self) -> None:
        """Set up the main view stack"""
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        self.stack.set_transition_duration(300)  # 300ms animation duration

    def setup_header_bar(self) -> None:
        """Set up the application header bar"""
        self.header_bar = Adw.HeaderBar()
        self.header_bar.set_show_end_title_buttons(True)
        self.header_bar.set_show_start_title_buttons(True)
        self.header_bar.add_css_class("flat")  # Make header flat for modern look

        # Use a simpler title for better space management
        title_label = Gtk.Label(label=_("BigOcrPdf - Scanned PDFs with search support"))
        self.header_bar.set_title_widget(title_label)

        # Create a menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_tooltip_text(_("Menu"))

        # Create the app menu
        menu = Gio.Menu()
        menu.append(_("Help"), "win.help")
        menu.append(_("About the application"), "app.about")
        menu_button.set_menu_model(menu)
        self.header_bar.pack_end(menu_button)

        # Add help action for the window
        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", lambda a, p: self.show_welcome_dialog())
        self.add_action(help_action)

    def setup_content_area(self) -> None:
        """Set up the main content area with scrolling"""
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(450)  # Reasonable minimum height
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        # Create adaptive layout structure
        self.toolbar_view = Adw.ToolbarView()
        self.toolbar_view.add_top_bar(self.header_bar)
        self.toolbar_view.set_content(content_scroll)

    def setup_action_bar(self) -> None:
        """Set up the bottom action bar for navigation"""
        self.action_bar = Gtk.ActionBar()

        # Create step indicator
        self.step_label = Gtk.Label()
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
        )
        self.step_label.add_css_class("dim-label")
        self.step_label.set_margin_start(12)
        self.step_label.set_hexpand(True)
        self.step_label.set_halign(Gtk.Align.CENTER)
        self.action_bar.set_center_widget(self.step_label)

        # Create back button (initially hidden)
        self.back_button = Gtk.Button()
        self.back_button.set_label(_("Back"))
        self.back_button.set_icon_name("go-previous-symbolic")
        self.connect_signal(self.back_button, "clicked", self.on_back_clicked)
        self.back_button.set_sensitive(False)
        self.back_button.set_visible(False)

        # Create next/finish button
        self.next_button = Gtk.Button()
        self.next_button.set_label(_("Start"))
        self.next_button.add_css_class("suggested-action")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)

        # Add buttons to action bar
        self.action_bar.pack_start(self.back_button)
        self.action_bar.pack_end(self.next_button)

    def setup_pages(self) -> None:
        """Set up the application pages"""
        # Create pages for the stack
        settings_page = self.ui.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        terminal_page = self.ui.create_terminal_page()
        self.stack.add_named(terminal_page, "terminal")

        conclusion_page = self.ui.create_conclusion_page()
        self.stack.add_named(conclusion_page, "conclusion")

    def connect_signal(self, widget, signal, callback, *args) -> int:
        """Connect a signal and store the handler ID
        
        Args:
            widget: The widget to connect the signal to
            signal: The signal name
            callback: The callback function
            args: Additional arguments to pass to the callback
            
        Returns:
            The handler ID
        """
        handler_id = widget.connect(signal, callback, *args)
        
        # Store the handler ID
        if widget not in self.signal_handlers:
            self.signal_handlers[widget] = {}
        
        self.signal_handlers[widget][signal] = handler_id
        
        return handler_id

    def disconnect_signal(self, widget, signal) -> bool:
        """Disconnect a signal if it exists
        
        Args:
            widget: The widget to disconnect the signal from
            signal: The signal name
            
        Returns:
            True if the signal was disconnected, False otherwise
        """
        if widget in self.signal_handlers and signal in self.signal_handlers[widget]:
            widget.disconnect(self.signal_handlers[widget][signal])
            del self.signal_handlers[widget][signal]
            return True
        return False

    def should_show_welcome_dialog(self) -> bool:
        """Check if the welcome dialog should be shown at startup
        
        Returns:
            True if the dialog should be shown, False otherwise
        """
        if not os.path.exists(self.WELCOME_DIALOG_CONFIG):
            # File doesn't exist, create it with default value (show dialog)
            try:
                with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                    f.write("true")
                return True
            except Exception as e:
                logger.error(f"Error creating welcome dialog config: {e}")
                return True
        
        try:
            with open(self.WELCOME_DIALOG_CONFIG, "r") as f:
                content = f.read().strip()
                return content.lower() == "true"
        except Exception as e:
            logger.error(f"Error reading welcome dialog config: {e}")
            return True

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
        # Create the welcome dialog
        dialog = Gtk.Window()
        dialog.set_title(_("Welcome to Big OCR PDF"))
        dialog.set_default_size(640, 670)
        dialog.set_modal(True)
        dialog.set_transient_for(self)
        
        # Create a toolbar view for the content
        toolbar_view = Adw.ToolbarView()
        
        # Create a box for the content
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        content_box.set_margin_top(24)
        content_box.set_margin_bottom(24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        
        # Add logo/icon
        icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        icon.set_pixel_size(64)
        icon.set_margin_bottom(16)
        content_box.append(icon)
        
        # Add title
        title = Gtk.Label()
        title.set_markup("<span size='x-large' weight='bold'>" + _("Welcome to Big OCR PDF") + "</span>")
        title.set_margin_bottom(16)
        content_box.append(title)
        
        # Add "What is Big OCR PDF?" section
        what_is = Gtk.Label()
        what_is.set_markup("<span weight='bold'>" + _("What is Big OCR PDF?") + "</span>")
        what_is.set_halign(Gtk.Align.START)
        content_box.append(what_is)
        
        what_is_desc = Gtk.Label()
        what_is_desc.set_markup(_(
            "Big OCR PDF adds optical character recognition to your PDF files, "
            "making them searchable and allowing you to select and copy text "
            "from scanned documents."
        ))
        what_is_desc.set_wrap(True)
        what_is_desc.set_halign(Gtk.Align.START)
        what_is_desc.set_margin_bottom(16)
        content_box.append(what_is_desc)
        
        # Add "Benefits of using Big OCR PDF" section
        benefits = Gtk.Label()
        benefits.set_markup("<span weight='bold'>" + _("Benefits of using Big OCR PDF:") + "</span>")
        benefits.set_halign(Gtk.Align.START)
        content_box.append(benefits)
        
        # Create a list of benefits
        benefits_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        benefits_list.set_margin_bottom(16)
        
        benefit_items = [
            _("• Search through your scanned PDF documents"),
            _("• Copy text from images and scanned documents"),
            _("• Process multiple files at once"),
            _("• Correct page alignment and rotation automatically"),
            _("• Choose optimal quality settings for your needs")
        ]
        
        for item in benefit_items:
            benefit_label = Gtk.Label()
            benefit_label.set_markup(item)
            benefit_label.set_halign(Gtk.Align.START)
            benefits_list.append(benefit_label)
        
        content_box.append(benefits_list)
        
        # Add "Show dialog at startup" switch
        show_at_startup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        show_at_startup_box.set_margin_top(16)
        
        show_at_startup_label = Gtk.Label(label=_("Show this dialog at startup"))
        show_at_startup_label.set_halign(Gtk.Align.START)
        show_at_startup_label.set_hexpand(True)
        
        show_at_startup_switch = Gtk.Switch()
        show_at_startup_switch.set_active(self.should_show_welcome_dialog())
        show_at_startup_switch.set_valign(Gtk.Align.CENTER)
        
        show_at_startup_box.append(show_at_startup_label)
        show_at_startup_box.append(show_at_startup_switch)
        content_box.append(show_at_startup_box)
        
        # Add "Let's Get Started" button
        start_button = Gtk.Button()
        start_button.set_label(_("Let's Get Started"))
        start_button.add_css_class("suggested-action")
        start_button.set_margin_top(16)
        start_button.set_halign(Gtk.Align.CENTER)
        content_box.append(start_button)
        
        # Create a scrolled window for the content
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_child(content_box)
        
        # Add the scrolled window to the toolbar view
        toolbar_view.set_content(scrolled_window)
        
        # Set the toolbar view as the dialog content
        dialog.set_child(toolbar_view)
        
        # Connect signals
        def on_switch_toggle(switch, _param):
            self.set_show_welcome_dialog(switch.get_active())
        
        show_at_startup_switch.connect("notify::active", on_switch_toggle)
        start_button.connect("clicked", lambda _: dialog.close())
        
        # Show the dialog
        dialog.present()

    def on_back_clicked(self, button: Gtk.Button) -> None:
        """Handle back button navigation

        Args:
            button: The button that triggered the event
        """
        current_page = self.stack.get_visible_child_name()

        if current_page == "terminal":
            # Go back to settings page
            self.stack.set_visible_child_name("settings")
            self.step_label.set_markup(
                "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
            )
            self.back_button.set_sensitive(False)
            self.back_button.set_visible(False)
            self.next_button.set_label(_("Start Processing"))
            self.next_button.set_sensitive(True)
            self.next_button.set_visible(True)

            # Update the file queue to reflect any changes
            self.update_file_info()

        elif current_page == "conclusion":
            # Go back to settings for a new conversion
            self.reset_and_go_to_settings()

    def on_next_clicked(self, button: Gtk.Button) -> None:
        """Handle next button navigation

        Args:
            button: The button that triggered the event
        """
        current_page = self.stack.get_visible_child_name()

        if current_page == "settings":
            # Start OCR processing (same as "Process PDF" button)
            self.on_apply_clicked(button)

        elif current_page == "conclusion":
            # Start a new conversion
            self.reset_and_go_to_settings()

    def show_help(self, button: Gtk.Button = None) -> None:
        """Show help dialog with usage instructions

        Args:
            button: The button that triggered the event (optional)
        """
        # Now this method redirects to the welcome dialog
        self.show_welcome_dialog()

    def show_about_dialog(self) -> None:
        """Show about dialog with application information"""
        # Get the application instance
        app = self.get_application()
        if app:
            app.on_about_action(None, None)

    def on_add_file_clicked(self, button: Gtk.Button) -> None:
        """Handle add file button click

        Args:
            button: The button that triggered the event
        """
        # Use the native file chooser portal for better desktop integration
        file_chooser = Gtk.FileDialog.new()
        file_chooser.set_title(_("Select PDF Files"))

        # Configure for multiple selection
        file_chooser.set_modal(True)

        # Create filter for PDF files
        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_mime_type("application/pdf")
        pdf_filter.add_pattern("*.pdf")

        # Create filter collection
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(pdf_filter)
        file_chooser.set_filters(filters)

        # Open the file chooser
        file_chooser.open_multiple(
            parent=self, cancellable=None, callback=self._on_open_multiple_callback
        )

    def _on_open_multiple_callback(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        """Handle file chooser completion for multiple files

        Args:
            dialog: The file dialog
            result: The async result
        """
        try:
            # Get the selected files
            files = dialog.open_multiple_finish(result)

            if files and files.get_n_items() > 0:
                # Convert to paths
                file_paths = []
                for i in range(files.get_n_items()):
                    file = files.get_item(i)
                    if isinstance(file, Gio.File):
                        file_paths.append(file.get_path())

                # Add files to settings
                added = self.settings.add_files(file_paths)

                if added > 0:
                    # Update UI
                    self.update_file_info()
                else:
                    # No valid files found, log the issue and show toast
                    logger.warning(_("No valid files were selected"))
                    self.show_toast(_("No valid PDF files were selected"))
        except Exception as e:
            logger.error(f"Error adding files: {e}")
            self.show_toast(_("Error adding files"))

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a toast notification
        
        Args:
            message: The message to display
            timeout: The timeout in seconds
        """
        toast = Adw.Toast.new(message)
        toast.set_timeout(timeout)
        self.toast_overlay.add_toast(toast)

    def on_browse_clicked(self, button: Gtk.Button) -> None:
        """Handle the browse button click for selecting a destination folder

        Args:
            button: The button that triggered the event
        """
        # Create a folder chooser dialog
        folder_chooser = Gtk.FileDialog.new()
        folder_chooser.set_title(_("Select destination folder"))
        folder_chooser.set_modal(True)

        # Set initial folder if we have one already selected
        self.set_initial_folder(folder_chooser)

        # Show the folder selection dialog
        folder_chooser.select_folder(
            parent=self, cancellable=None, callback=self._on_select_folder_callback
        )

    def set_initial_folder(self, folder_chooser: Gtk.FileDialog) -> None:
        """Set the initial folder for a folder chooser dialog
        
        Args:
            folder_chooser: The folder chooser dialog
        """
        if self.settings.destination_folder:
            # Try to use the directory from existing destination path
            if os.path.isdir(self.settings.destination_folder):
                initial_folder = self.settings.destination_folder
            else:
                initial_folder = os.path.dirname(self.settings.destination_folder)

            # Set initial folder if it exists
            if os.path.exists(initial_folder):
                folder = Gio.File.new_for_path(initial_folder)
                folder_chooser.set_initial_folder(folder)

    def _on_select_folder_callback(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        """Handle the folder selection dialog response

        Args:
            dialog: The file dialog
            result: The async result
        """
        try:
            # Get the selected folder
            folder = dialog.select_folder_finish(result)
            if folder:
                path = folder.get_path()

                # Update the UI and settings
                self.settings.destination_folder = path
                if hasattr(self.ui, "dest_entry") and self.ui.dest_entry:
                    self.ui.dest_entry.set_text(path)

                # Log destination selection and show toast
                logger.info(f"Destination folder selected: {path}")
                self.show_toast(_("Destination folder selected"))

        except Exception as e:
            logger.error(f"Error selecting save location: {e}")
            self.show_toast(_("Error selecting destination folder"))

    def _validate_ocr_settings(self) -> bool:
        """Validate OCR settings before processing

        Returns:
            True if settings are valid, False otherwise
        """
        # Validate there are files to process
        if not self.settings.selected_files:
            logger.warning(_("No files selected for processing"))
            self.show_toast(_("No files selected for processing"))
            return False

        # Get "save in same folder" option
        save_in_same_folder = self.get_save_in_same_folder()

        # Validate destination folder if not saving in same folder
        if not save_in_same_folder and not self.settings.destination_folder:
            logger.warning(_("No destination folder selected"))
            self.show_toast(_("Please select a destination folder"))
            return False

        return True

    def get_save_in_same_folder(self) -> bool:
        """Get the value of the save in same folder switch
        
        Returns:
            True if saving in the same folder, False otherwise
        """
        if hasattr(self.ui, "same_folder_switch_row") and self.ui.same_folder_switch_row:
            return self.ui.same_folder_switch_row.get_active()
        return False

    def _get_settings_from_ui(self) -> None:
        """Get settings from UI components"""
        # Get language setting
        if hasattr(self.ui, "lang_dropdown") and self.ui.lang_dropdown is not None:
            lang_index = self.ui.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        # Get quality setting
        if hasattr(self.ui, "quality_dropdown") and self.ui.quality_dropdown is not None:
            quality_index = self.ui.quality_dropdown.get_selected()
            if 0 <= quality_index < len(self.QUALITY_CONFIG):
                self.settings.quality = self.QUALITY_CONFIG[quality_index][0]

        # Get alignment setting
        if hasattr(self.ui, "alignment_dropdown") and self.ui.alignment_dropdown is not None:
            align_index = self.ui.alignment_dropdown.get_selected()
            if 0 <= align_index < len(self.ALIGNMENT_CONFIG):
                self.settings.align = self.ALIGNMENT_CONFIG[align_index][0]

        # Get "save in same folder" option
        save_in_same_folder = self.get_save_in_same_folder()

        # Get destination folder
        if hasattr(self.ui, "dest_entry") and self.ui.dest_entry is not None:
            dest_folder = self.ui.dest_entry.get_text()
            if dest_folder:
                self.settings.destination_folder = dest_folder

        # Save all settings
        self.settings.save_settings(
            self.settings.lang,
            self.settings.quality,
            self.settings.align,
            self.settings.destination_folder,
            save_in_same_folder,
        )

    def on_apply_clicked(self, button: Gtk.Button) -> None:
        """Process the selected files with OCR

        Args:
            button: The button that triggered the event
        """
        # Validate settings
        if not self._validate_ocr_settings():
            return

        # Get settings from UI
        self._get_settings_from_ui()

        # Set up for processing
        self.processed_files = []  # Reset processed files list
        self.process_start_time = time.time()

        # Register callbacks for OCR processing events
        self.ocr_processor.register_callbacks(
            on_file_complete=self._on_file_processed,
            on_all_complete=self._on_processing_complete,
        )

        # Start OCR processing using Python API
        success = self.ocr_processor.process_with_api()
        if not success:
            logger.error(_("Failed to start OCR processing"))
            self.show_toast(_("Failed to start OCR processing"))
            return

        # Switch to terminal page and update UI
        self.stack.set_visible_child_name("terminal")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 2/3: Processing") + "</span>"
        )
        self.back_button.set_visible(False)
        self.next_button.set_visible(False)

        # Start progress updates
        self._start_progress_monitor()

        # Processing has started - log and show toast
        logger.info(_("OCR processing started using Python API"))
        self.show_toast(_("OCR processing started"))

    def reset_and_go_to_settings(self) -> None:
        """Reset the application state and return to the settings page"""
        # Clear processed files list
        self.processed_files = []

        # Reset UI elements
        self.next_button.set_label(_("Start"))
        self.next_button.set_sensitive(True)
        self.next_button.set_visible(True)

        # Disconnect any previous handlers and connect the standard handler
        self.disconnect_signal(self.next_button, "clicked")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)

        # Return to the settings page
        self.stack.set_visible_child_name("settings")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
        )

        # Update the file queue interface to reflect any changes
        self.update_file_info()

    def show_conclusion_page(self) -> None:
        """Show the conclusion page after OCR processing completes"""
        # Switch to the conclusion page
        self.stack.set_visible_child_name("conclusion")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 3/3: Completed") + "</span>"
        )

        # Update UI for conclusion page
        self.back_button.set_visible(False)

        # Configure the next button as a "Back" button
        self.next_button.set_label(_("Back"))
        self.next_button.set_visible(True)
        self.next_button.set_sensitive(True)

        # Disconnect any existing signal handlers and reconnect
        self.disconnect_signal(self.next_button, "clicked")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)

    def update_file_info(self) -> None:
        """Update the file information UI after files have been added or removed"""
        current_page = self.stack.get_visible_child_name()
        
        if current_page != "settings":
            return  # Skip updates when not on settings page
            
        if hasattr(self.ui, "refresh_queue_status"):
            self.ui.refresh_queue_status()
            logger.info(f"Queue status refreshed with {len(self.settings.selected_files)} files")
            return
            
        # Full UI rebuild only when necessary
        old_page = self.stack.get_visible_child()
        new_page = self.ui.create_settings_page()
        self.stack.remove(old_page)
        self.stack.add_named(new_page, "settings")
        self.stack.set_visible_child_name("settings")
        logger.info(f"UI updated with {len(self.settings.selected_files)} files in queue")

    def _on_page_changed(self, stack, _param) -> None:
        """Handle page change events to update the headerbar styling

        Args:
            stack: The stack widget that changed
            _param: The property that changed (unused)
        """
        current_page = stack.get_visible_child_name()

        # Apply or remove styling based on the current page
        if current_page == "settings":
            # On settings page, match the headerbar to sidebar color
            self._apply_headerbar_sidebar_style()
        else:
            # On other pages, use default headerbar styling
            self._remove_headerbar_sidebar_style()

    def _apply_headerbar_sidebar_style(self) -> None:
        """Apply a style to the headerbar for the settings page (optional)"""
        if self.header_bar:
            self.header_bar.add_css_class("settings-page-header")
            self.header_bar.remove_css_class("default-header")

    def _remove_headerbar_sidebar_style(self) -> None:
        """Remove style from the headerbar for non-settings pages"""
        if self.header_bar:
            # Toggle the classes to reset to default style
            self.header_bar.remove_css_class("settings-page-header")
            self.header_bar.add_css_class("default-header")

    def _on_file_processed(
        self, input_file: str, output_file: str, extracted_text: str = ""
    ) -> None:
        """Callback when a file is processed with OCR

        Args:
            input_file: Path to the input file
            output_file: Path to the output file
            extracted_text: The text extracted from the processed PDF
        """
        # Remove the processed file from the queue
        self.ocr_processor.remove_processed_file(input_file)

        # Store the extracted text for this output file
        logger.info(
            f"Received extracted_text in _on_file_processed: length={len(extracted_text)}"
        )

        # Make sure output_file is in processed_files list
        if output_file not in self.settings.processed_files:
            self.settings.processed_files.append(output_file)
            logger.info(f"Added {output_file} to processed_files list")

        # Store the extracted text, ensuring we have the dictionary initialized
        if not hasattr(self.settings, "extracted_text"):
            self.settings.extracted_text = {}

        # Store the text with the output file as key
        self.settings.extracted_text[output_file] = extracted_text
        logger.info(
            f"Stored {len(extracted_text)} characters of extracted text for {os.path.basename(output_file)}"
        )
        
        # Update the status bar with current progress
        self._update_processing_status(input_file)

    def _update_processing_status(self, input_file: str = None) -> None:
        """Update the status bar with current processing information"""
        file_count = self.ocr_processor.get_processed_count()
        total_files = self.ocr_processor.get_total_count()

        if hasattr(self.ui, "terminal_status_bar") and self.ui.terminal_status_bar:
            remaining = total_files - file_count
            self.ui.terminal_status_bar.set_markup(
                _("<b>Processing: {current}/{total}:</b> {completed} file(s) completed • <b>{remaining}</b> remaining").format(
                    current=file_count,
                    total=total_files,
                    completed=file_count,
                    remaining=remaining
                )
            )

        # Log processing complete for this file
        if input_file:
            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=file_count,
                    total=total_files,
                    filename=os.path.basename(input_file)
                )
            )

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR"""
        logger.info(_("OCR processing complete callback triggered"))

        # Check if we're still on the terminal page (might have been cancelled)
        if self.stack.get_visible_child_name() != "terminal":
            logger.info(
                _("Processing complete but no longer on terminal page, likely cancelled")
            )
            return

        # Update the progress display to show 100%
        self._update_terminal_progress(1.0, "100%")

        # Update the status text
        self._update_terminal_status_complete()

        # Stop the spinner if it's still spinning
        self._stop_terminal_spinner()

        # First update the conclusion page with actual values
        if hasattr(self.ui, "update_conclusion_page"):
            self.ui.update_conclusion_page()

        # Show conclusion page with a short delay
        self.conclusion_timer_id = GLib.timeout_add(
            1000, lambda: self.show_conclusion_page()
        )

        # Log completion
        logger.info(
            _("OCR processing completed successfully for {count} files").format(
                count=self.ocr_processor.get_processed_count()
            )
        )

        # Show toast notification
        self.show_toast(_("OCR processing complete"))

    def _update_terminal_progress(self, fraction: float, text: str = None) -> None:
        """Update the terminal progress bar
        
        Args:
            fraction: Progress fraction (0.0-1.0)
            text: Optional text to display
        """
        if hasattr(self.ui, "terminal_progress_bar") and self.ui.terminal_progress_bar:
            self.ui.terminal_progress_bar.set_fraction(fraction)
            if text:
                self.ui.terminal_progress_bar.set_text(text)

    def _update_terminal_status_complete(self) -> None:
        """Update terminal status to show completion"""
        if hasattr(self.ui, "terminal_status_bar") and self.ui.terminal_status_bar:
            # Get the total files processed from the OCR queue
            total_files = self.ocr_processor.get_processed_count()
            self.ui.terminal_status_bar.set_markup(
                _("<b>OCR processing complete!</b> {total} file(s) processed").format(
                    total=total_files
                )
            )

    def _stop_terminal_spinner(self) -> None:
        """Stop the terminal spinner"""
        if hasattr(self.ui, "terminal_spinner") and self.ui.terminal_spinner:
            self.ui.terminal_spinner.set_spinning(False)

    def _start_progress_monitor(self) -> None:
        """Start monitoring the OCR progress"""
        # Set up a timer to update the progress
        self.progress_timer_id = GLib.timeout_add(
            250, lambda: self._update_ocr_progress()
        )

    def _is_on_terminal_page(self) -> bool:
        """Check if we're on the terminal page and handle timer if not

        Returns:
            True if on terminal page, False otherwise
        """
        if self.stack.get_visible_child_name() != "terminal":
            # If we've left the terminal page, stop the timer
            if self.progress_timer_id is not None:
                safe_remove_source(self.progress_timer_id)
                self.progress_timer_id = None
            return False
        return True

    def _calculate_progress(self) -> float:
        """Calculate the current progress value

        Returns:
            Progress value between 0.0 and 1.0
        """
        progress = self.ocr_processor.get_progress()

        # Use a timebased animation if the progress is still at 0
        # This gives feedback to the user that processing is happening
        if progress == 0:
            # Oscillate between 0.05 and 0.15 to show activity
            if self._animation_progress >= 0.15:
                self._animation_direction = -0.01
            elif self._animation_progress <= 0.05:
                self._animation_direction = 0.01

            self._animation_progress += self._animation_direction
            
            # Use animation progress during active processing
            return self._animation_progress
        else:
            # Use real progress
            return progress

    def _update_ocr_progress(self) -> bool:
        """Update the OCR progress in the UI

        Returns:
            True to continue updating, False to stop
        """
        # Check if we're still on the terminal page
        if not self._is_on_terminal_page():
            return False

        # Calculate and update the progress
        progress = self._calculate_progress()
        
        # Update the UI with progress information
        self._update_progress_ui(progress)
        
        return True

    def _update_progress_ui(self, progress: float) -> None:
        """Update all progress UI elements
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        # Update progress bar
        if hasattr(self.ui, "terminal_progress_bar") and self.ui.terminal_progress_bar:
            self.ui.terminal_progress_bar.set_fraction(progress)
            progress_percent = int(progress * 100)
            self.ui.terminal_progress_bar.set_text(f"{progress_percent}%")

        # Update spinner to indicate activity
        if hasattr(self.ui, "terminal_spinner") and self.ui.terminal_spinner:
            if not self.ui.terminal_spinner.get_spinning():
                self.ui.terminal_spinner.set_spinning(True)
                
        # Update status text
        self._update_progress_status()

    def _update_progress_status(self) -> None:
        """Update the status text with current progress information"""
        if hasattr(self.ui, "terminal_status_bar") and self.ui.terminal_status_bar:
            # Get file counts from OCR processor
            files_processed = self.ocr_processor.get_processed_count()
            total_files = self.ocr_processor.get_total_count()
            elapsed_time = 0

            if hasattr(self, "process_start_time"):
                elapsed_time = int(time.time() - self.process_start_time)

            # Format the elapsed time
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

            remaining = total_files - files_processed
            progress = self.ocr_processor.get_progress()

            # Show the current status
            if progress >= 1.0 or files_processed >= total_files:
                # Process is complete
                self.ui.terminal_status_bar.set_markup(
                    _(
                        "<b>OCR processing complete!</b> {total} file(s) processed • Total time: {time}"
                    ).format(total=total_files, time=time_str)
                )

                # Stop the spinner
                self._stop_terminal_spinner()

                # Stop the timer
                if self.progress_timer_id is not None:
                    safe_remove_source(self.progress_timer_id)
                    self.progress_timer_id = None

            elif files_processed > 0:
                current_file = (
                    files_processed + 1
                    if files_processed < total_files
                    else total_files
                )
                self.ui.terminal_status_bar.set_markup(
                    _(
                        "<b>Processing file {cur}/{total}:</b> {done} file(s) completed • <b>{rem}</b> remaining • Time: {time}"
                    ).format(
                        cur=current_file,
                        total=total_files,
                        done=files_processed,
                        rem=remaining,
                        time=time_str,
                    )
                )
            else:
                # Show different stages of processing based on elapsed time
                stage = self._get_processing_stage(elapsed_time)
                
                self.ui.terminal_status_bar.set_markup(
                    _(
                        "<b>Processing file 1/{total}:</b> {stage} • Time: {time}"
                    ).format(total=total_files, stage=stage, time=time_str)
                )

    def _get_processing_stage(self, elapsed_time: int) -> str:
        """Get the current processing stage based on elapsed time
        
        Args:
            elapsed_time: Elapsed time in seconds
            
        Returns:
            String describing the current stage
        """
        if elapsed_time < 5:
            return _("starting conversion")
        elif elapsed_time < 10:
            return _("analyzing document")
        elif elapsed_time < 15:
            return _("applying OCR")
        elif elapsed_time < 20:
            return _("processing texts")
        else:
            return _("finalizing processing")

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click during OCR processing"""
        # Stop the OCR queue
        if self.ocr_processor.ocr_queue:
            self.ocr_processor.ocr_queue.stop()
            logger.info(_("OCR processing cancelled by user"))

            # Stop the progress monitor
            if self.progress_timer_id is not None:
                safe_remove_source(self.progress_timer_id)
                self.progress_timer_id = None

            # Show a toast notification
            self.show_toast(_("OCR processing cancelled"))

            # Reset the OCR processor status
            self.ocr_processor._processed_files = 0
            self.ocr_processor._progress = 0.0
            self.ocr_processor.ocr_queue = None

            # Go back to the settings page
            self.reset_and_go_to_settings()