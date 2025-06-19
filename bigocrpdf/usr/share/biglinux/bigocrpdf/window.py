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

from config import APP_ICON_NAME
from utils.logger import logger
from services.settings import OcrSettings
from services.processor import OcrProcessor
from ui.welcome import WelcomeDialog
from ui.first_window import FirstWindow
from ui.progress_window import ProgressWindow
from ui.final_window import FinalWindow
from ui.settings import SettingsDialog
from utils.i18n import _


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
        ("alignrotate", _("Align + rotate")),
    ]

    ALIGNMENT_TOOLTIPS = [
        _("Keep pages as they are without any modifications"),
        _("Align page content to improve readability"),
        _("Automatically rotate pages to correct orientation"),
        _("Align page content and automatically rotate pages to correct orientation"),
    ]

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

        # Initialize UI components
        self.welcome_dialog = WelcomeDialog(self)
        self.first_window = FirstWindow(self)
        self.progress_window = ProgressWindow(self)
        self.final_window = FinalWindow(self)
        self.settings_dialog = SettingsDialog(self)

        # Initialize state variables
        self.processed_files = []  # List to store processed output files
        self.process_start_time = 0
        self.progress_timer_id = None
        self.conclusion_timer_id = None
        self.current_processing_file_name = ""
        self.last_progress_update = 0  # Track last progress change
        self.last_progress_value = 0.0  # Track last progress value
        self.stall_check_count = 0  # Counter for stall detection

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
        menu.append(_("Show Welcome Screen"), "win.help")
        menu.append(_("About the application"), "app.about")
        menu_button.set_menu_model(menu)
        self.header_bar.pack_end(menu_button)

        # Add welcome screen action for the window
        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", self.on_help_action)
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
        settings_page = self.first_window.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        terminal_page = self.progress_window.create_terminal_page()
        self.stack.add_named(terminal_page, "terminal")

        conclusion_page = self.final_window.create_conclusion_page()
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
        """Return True if the welcome dialog should be shown at startup."""
        return self.welcome_dialog.should_show_welcome_dialog()

    def set_show_welcome_dialog(self, show: bool) -> None:
        """Set whether to show the welcome dialog at startup

        Args:
            show: True to show the dialog, False to hide it
        """
        self.welcome_dialog.set_show_welcome_dialog(show)

    def show_welcome_dialog(self) -> None:
        """Show the welcome dialog with application information"""
        self.welcome_dialog.show_welcome_dialog()

    def on_help_action(self, action: Gio.SimpleAction, param) -> None:
        """Handle help menu action - show welcome dialog

        Args:
            action: The action that triggered the event
            param: Additional parameters
        """
        self.show_welcome_dialog()

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
                    # No valid files found, log the issue
                    logger.warning(_("No valid files were selected"))

        except Exception as e:
            logger.error(f"Error adding files: {e}")
            self.show_toast(_("Error adding files"))

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a toast notification, but not on the terminal page.

        Args:
            message: The message to display
            timeout: The timeout in seconds
        """
        # Do not show toasts on the processing page to avoid distraction
        if self.stack and self.stack.get_visible_child_name() == "terminal":
            logger.info(f"Toast suppressed (on terminal page): {message}")
            return

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
                if (
                    hasattr(self.first_window, "dest_entry")
                    and self.first_window.dest_entry
                ):
                    self.first_window.dest_entry.set_text(path)

                # Log destination selection
                logger.info(f"Destination folder selected: {path}")

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
        if (
            hasattr(self.first_window, "same_folder_switch_row")
            and self.first_window.same_folder_switch_row
        ):
            return self.first_window.same_folder_switch_row.get_active()
        return False

    def _get_settings_from_ui(self) -> None:
        """Get settings from UI components"""
        # Get language setting
        if (
            hasattr(self.first_window, "lang_dropdown")
            and self.first_window.lang_dropdown is not None
        ):
            lang_index = self.first_window.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        # Get quality setting
        if (
            hasattr(self.first_window, "quality_dropdown")
            and self.first_window.quality_dropdown is not None
        ):
            quality_index = self.first_window.quality_dropdown.get_selected()
            if 0 <= quality_index < len(self.QUALITY_CONFIG):
                self.settings.quality = self.QUALITY_CONFIG[quality_index][0]

        # Get alignment setting
        if (
            hasattr(self.first_window, "alignment_dropdown")
            and self.first_window.alignment_dropdown is not None
        ):
            align_index = self.first_window.alignment_dropdown.get_selected()
            if 0 <= align_index < len(self.ALIGNMENT_CONFIG):
                self.settings.align = self.ALIGNMENT_CONFIG[align_index][0]

        # Get "save in same folder" option
        save_in_same_folder = self.get_save_in_same_folder()

        # Get destination folder
        if (
            hasattr(self.first_window, "dest_entry")
            and self.first_window.dest_entry is not None
        ):
            dest_folder = self.first_window.dest_entry.get_text()
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
        try:
            # Validate settings
            if not self._validate_ocr_settings():
                return

            # Get settings from UI
            self._get_settings_from_ui()

            # Additional validation of files and settings
            if not self.settings.selected_files:
                logger.error("No files selected for processing")
                self.show_toast(_("Please select files to process"))
                return

            # Check if all files still exist
            missing_files = []
            for file_path in self.settings.selected_files:
                if not os.path.exists(file_path):
                    missing_files.append(os.path.basename(file_path))

            if missing_files:
                error_msg = _("Some files are no longer available: {0}").format(
                    ", ".join(missing_files)
                )
                logger.error(error_msg)
                self.show_toast(error_msg)
                return

            # Set up for processing
            self.processed_files = []  # Reset processed files list
            self.process_start_time = time.time()
            self.last_progress_update = time.time()  # Initialize stall detection
            self.last_progress_value = 0.0
            self.stall_check_count = 0

            self.current_processing_file_name = (
                os.path.basename(self.settings.selected_files[0])
                if self.settings.selected_files
                else ""
            )

            # Register callbacks for OCR processing events
            self.ocr_processor.register_callbacks(
                on_file_complete=self._on_file_processed,
                on_all_complete=self._on_processing_complete,
            )

            logger.info(
                f"Starting OCR processing for {len(self.settings.selected_files)} files"
            )

            # Start OCR processing using Python API
            success = self.ocr_processor.process_with_api()
            if not success:
                error_msg = _("Failed to start OCR processing")
                logger.error(error_msg)
                self.show_toast(error_msg)
                return

            # Additional validation: Check if OCR queue actually has processes
            if not self.ocr_processor.ocr_queue:
                error_msg = _("OCR queue was not created properly")
                logger.error(error_msg)
                self.show_toast(error_msg)
                return

            # Brief check to ensure processes are starting
            time.sleep(0.2)  # Give processes a moment to start
            with self.ocr_processor.ocr_queue.lock:
                total_processes = len(self.ocr_processor.ocr_queue.queue) + len(
                    self.ocr_processor.ocr_queue.running
                )
                if total_processes == 0:
                    error_msg = _("No OCR processes were started")
                    logger.error(error_msg)
                    self.show_toast(error_msg)
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

            # Processing has started - log
            logger.info(_("OCR processing started using Python API"))

        except Exception as e:
            error_msg = _("Error starting OCR processing: {0}").format(str(e))
            logger.error(error_msg)
            self.show_toast(error_msg)

            # Ensure we don't get stuck on the processing screen
            if self.stack.get_visible_child_name() == "terminal":
                self.reset_and_go_to_settings()

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

        if hasattr(self.first_window, "refresh_queue_status"):
            self.first_window.refresh_queue_status()
            logger.info(
                f"Queue status refreshed with {len(self.settings.selected_files)} files"
            )
            return

        # Full UI rebuild only when necessary
        old_page = self.stack.get_visible_child()
        new_page = self.first_window.create_settings_page()
        self.stack.remove(old_page)
        self.stack.add_named(new_page, "settings")
        self.stack.set_visible_child_name("settings")
        logger.info(
            f"UI updated with {len(self.settings.selected_files)} files in queue"
        )

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
        """Callback when a file is processed with OCR. Dispatches to the main thread."""
        GLib.idle_add(
            self._on_file_processed_ui, input_file, output_file, extracted_text
        )

    def _on_file_processed_ui(
        self, input_file: str, output_file: str, extracted_text: str
    ) -> None:
        """UI portion of the file processed callback."""
        # Remove the processed file from the internal list to stop tracking it
        if input_file in self.settings.selected_files:
            self.settings.selected_files.remove(input_file)

        # Update the current file name for the status display
        if self.settings.selected_files:
            self.current_processing_file_name = os.path.basename(
                self.settings.selected_files[0]
            )
        else:
            self.current_processing_file_name = ""  # No more files

        # Log and store results
        logger.info(f"File completed: {os.path.basename(input_file)}")

        if output_file not in self.processed_files:
            self.processed_files.append(output_file)

        if not hasattr(self.settings, "extracted_text"):
            self.settings.extracted_text = {}
        self.settings.extracted_text[output_file] = extracted_text

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR. Dispatches to the main thread."""
        GLib.idle_add(self._on_processing_complete_ui)

    def _on_processing_complete_ui(self) -> None:
        """UI portion of the processing complete callback. Ensures it runs safely in the main thread."""
        logger.info(
            "OCR processing complete callback triggered. Forcing final UI state."
        )

        # 1. Immediately stop the progress monitor to prevent it from overwriting the final state.
        if self.progress_timer_id is not None:
            self.progress_timer_id = None
            logger.info("Progress monitor stopped.")

        # Check if we're still on the terminal page (might have been cancelled)
        if self.stack.get_visible_child_name() != "terminal":
            logger.warning(
                "Processing complete but no longer on terminal page. Likely cancelled."
            )
            return

        # 2. Force the progress bar and status to their 100% completed state.
        self._update_terminal_progress(1.0)
        self._update_terminal_status_complete()

        # 3. Update the data for the conclusion page.
        if hasattr(self.final_window, "update_conclusion_page"):
            # Pass the final list of generated files
            self.settings.processed_files = self.processed_files
            self.final_window.update_conclusion_page()

        # 4. Schedule the transition to the conclusion page after a short delay.
        if self.conclusion_timer_id is not None:
            self.conclusion_timer_id = None
        self.conclusion_timer_id = GLib.timeout_add(
            1000, self._transition_to_conclusion
        )

        logger.info(
            _("OCR processing completed successfully for {count} files").format(
                count=self.ocr_processor.get_processed_count()
            )
        )

    def _transition_to_conclusion(self) -> bool:
        """Helper to transition to the conclusion page, run by a timer."""
        self.conclusion_timer_id = None  # Clear the ID
        self.show_conclusion_page()
        return GLib.SOURCE_REMOVE  # Timer should not repeat

    def _update_terminal_progress(self, fraction: float) -> None:
        """Update the terminal progress bar (visual only)."""
        if (
            hasattr(self.progress_window, "terminal_progress_bar")
            and self.progress_window.terminal_progress_bar
        ):
            self.progress_window.terminal_progress_bar.set_fraction(fraction)

    def _update_terminal_status_complete(self) -> None:
        """Update terminal status labels to show completion."""
        if self.progress_window.page_status_label:
            total_pages = self.ocr_processor.get_total_page_count()
            self.progress_window.page_status_label.set_markup(
                f"<span size='xx-large'>{_('Page {p} of {t}').format(p=total_pages, t=total_pages)}</span>"
            )
        if self.progress_window.file_status_label:
            self.progress_window.file_status_label.set_description(
                _("Processing complete!")
            )

        if self.progress_window.elapsed_time_label and self.process_start_time > 0:
            elapsed = time.time() - self.process_start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            self.progress_window.elapsed_time_label.set_text(
                f"{_('Elapsed time:')} {minutes:02d}:{seconds:02d}"
            )

    def _start_progress_monitor(self) -> None:
        """Start monitoring the OCR progress"""
        self.progress_timer_id = GLib.timeout_add(250, self._update_ocr_progress)

    def _is_on_terminal_page(self) -> bool:
        """Check if we're on the terminal page and handle timer if not."""
        if self.stack.get_visible_child_name() != "terminal":
            if self.progress_timer_id is not None:
                self.progress_timer_id = None
            return False
        return True

    def _update_ocr_progress(self) -> bool:
        """Update the OCR progress in the UI."""
        if not self._is_on_terminal_page():
            return False  # Stop the timer

        current_time = time.time()

        # Check for failed processes first
        if self.ocr_processor.has_failed_processes():
            errors = self.ocr_processor.get_failed_process_errors()
            error_message = "\n".join(errors[:3])  # Show first 3 errors
            if len(errors) > 3:
                error_message += f"\n... and {len(errors) - 3} more errors"

            logger.error(f"OCR processing failed: {error_message}")

            # Show error in progress window
            self.progress_window.set_processing_error(error_message)

            # Clean up and stop processing
            if self.progress_timer_id is not None:
                self.progress_timer_id = None

            # Stop the OCR queue
            if self.ocr_processor.ocr_queue:
                self.ocr_processor.ocr_queue.stop()
                self.ocr_processor.ocr_queue = None

            # After a short delay, return to settings
            def return_to_settings():
                self.reset_and_go_to_settings()

            GLib.timeout_add_seconds(3, return_to_settings)
            return False  # Stop the timer

        # Get current progress
        progress = self.ocr_processor.get_progress()

        # Enhanced stall detection with more robust monitoring
        if progress != self.last_progress_value:
            # Progress changed - update tracking
            self.last_progress_update = current_time
            self.last_progress_value = progress
            self.stall_check_count = 0
        else:
            # No progress change - check for stall
            time_since_progress = current_time - self.last_progress_update

            if time_since_progress > 45:  # Increased from 30 seconds
                self.stall_check_count += 1

                if self.stall_check_count == 1:  # First warning
                    logger.warning(
                        f"OCR processing appears stalled - no progress for {time_since_progress:.1f} seconds"
                    )

                    # Check if processes are actually running
                    if self.ocr_processor.ocr_queue:
                        with self.ocr_processor.ocr_queue.lock:
                            running_count = len(self.ocr_processor.ocr_queue.running)
                            queue_count = len(self.ocr_processor.ocr_queue.queue)
                            logger.info(
                                f"Queue status - Running: {running_count}, Queued: {queue_count}"
                            )

                elif (
                    self.stall_check_count >= 15  # Increased threshold
                ):  # More time before declaring complete stall
                    logger.error(
                        "OCR processing completely stalled - attempting recovery"
                    )

                    # Try to restart the processing
                    if self.ocr_processor.ocr_queue:
                        logger.info("Attempting to recover stalled OCR processing")
                        try:
                            # Stop current processing
                            self.ocr_processor.ocr_queue.stop()

                            # Show error message
                            self.progress_window.set_processing_error(
                                _("Processing appears to be stalled. Please try again.")
                            )

                            # Clean up
                            if self.progress_timer_id is not None:
                                self.progress_timer_id = None
                            self.ocr_processor.ocr_queue = None

                            # Return to settings after delay
                            def return_to_settings():
                                self.reset_and_go_to_settings()

                            GLib.timeout_add_seconds(3, return_to_settings)
                            return False  # Stop the timer

                        except Exception as e:
                            logger.error(f"Error during stall recovery: {e}")

        # Update UI with current progress
        self._update_terminal_progress(progress)
        self._update_progress_status()

        return True  # Continue the timer

    def _update_progress_status(self) -> None:
        """Update the status labels on the processing page."""
        if (
            not self.progress_window.page_status_label
            or not self.progress_window.file_status_label
        ):
            return

        processed_pages = self.ocr_processor.get_processed_page_count()
        total_pages = self.ocr_processor.get_total_page_count()

        # Update the main page counter
        self.progress_window.page_status_label.set_markup(
            f"<span size='xx-large'>{_('Page {p} of {t}').format(p=processed_pages, t=total_pages)}</span>"
        )

        # Update elapsed time
        if self.progress_window.elapsed_time_label and self.process_start_time > 0:
            elapsed = time.time() - self.process_start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            self.progress_window.elapsed_time_label.set_text(
                f"{_('Elapsed time:')} {minutes:02d}:{seconds:02d}"
            )

        # Update the file status
        files_processed = self.ocr_processor.get_processed_count()
        total_files = self.ocr_processor.get_total_count()

        if self.current_processing_file_name:
            description_text = _("File {c} of {t}: {name}").format(
                c=files_processed + 1,
                t=total_files,
                name=self.current_processing_file_name,
            )
            self.progress_window.file_status_label.set_description(description_text)
        else:
            self.progress_window.file_status_label.set_description(_("Finalizing..."))

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click during OCR processing"""
        if self.ocr_processor.ocr_queue:
            self.ocr_processor.ocr_queue.stop()
            logger.info(_("OCR processing cancelled by user"))

            if self.progress_timer_id is not None:
                self.progress_timer_id = None

            self.ocr_processor.ocr_queue = None
            self.reset_and_go_to_settings()
