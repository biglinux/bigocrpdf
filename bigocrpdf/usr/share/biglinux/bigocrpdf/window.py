"""
BigOcrPdf - Window Module

This module contains the main applica        # Create left header (will match sidebar color)
        left_header = Adw.HeaderBar()
        left_header.add_css_class("sidebar-header")
        left_header.set_show_end_title_buttons(False)
        left_header.set_show_start_title_buttons(False)
        left_header.set_hexpand(False)

        # Set a fixed width for the left header to match sidebar
        left_header.set_size_request(400, -1)dow implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio

import os
import time

from .config import ICON_PATH
from .utils.logger import logger
from .services.settings import OcrSettings
from .services.processor import OcrProcessor
from .ui.ui_manager import BigOcrPdfUI
from .utils.i18n import _


class BigOcrPdfWindow(Adw.ApplicationWindow):
    """Main application window for BigOcrPdf"""

    def __init__(self, app: Adw.Application):
        """Initialize application window

        Args:
            app: The parent Adw.Application instance
        """
        super().__init__(application=app, title=_("Big OCR PDF"))
        self.set_default_size(820, 600)

        # Set up the window icon (either by name or path)
        if os.path.exists(ICON_PATH):
            self.set_icon_name(ICON_PATH)
        else:
            self.set_icon_name("document-edit-symbolic")

        # Initialize components
        self.settings = OcrSettings()
        self.ocr_processor = OcrProcessor(self.settings)
        self.ui = BigOcrPdfUI(self)
        self.process_pid = None  # Process ID for OCR operation
        self.processed_files = []  # List to store processed output files

        # Initialize UI components
        self.stack = None  # ViewStack for main UI transitions
        self.toast_overlay = None  # Toast overlay for notifications
        self.step_label = None  # Step indicator
        self.back_button = None  # Back button
        self.next_button = None  # Next button

        # Create the main layout
        self.setup_ui()

    def setup_ui(self) -> None:
        # Create the toast overlay for notifications
        self.toast_overlay = Adw.ToastOverlay()

        # Create a view stack with smooth transitions
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        self.stack.set_transition_duration(300)  # 300ms animation duration

        # Create a unified headerbar
        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(True)
        header_bar.set_show_start_title_buttons(True)
        header_bar.add_css_class("flat")  # Make header flat for modern look

        # Use a simpler title for better space management
        title_label = Gtk.Label(label=_("BigOcrPdf - Scanned PDFs with search support"))
        header_bar.set_title_widget(title_label)

        # Create step indicator (will be added to the footer later)
        self.step_label = Gtk.Label()
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
        )
        self.step_label.add_css_class("dim-label")
        self.step_label.set_margin_start(12)

        # Create a help button
        help_button = Gtk.Button()
        help_button.set_icon_name("help-about-symbolic")
        help_button.set_tooltip_text(_("Help"))
        help_button.add_css_class("flat")
        help_button.connect("clicked", self.show_help)
        header_bar.pack_end(help_button)

        # Create a menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_tooltip_text(_("Menu"))

        # Create the app menu
        menu = Gio.Menu()
        menu.append(_("About the application"), "app.about")
        menu_button.set_menu_model(menu)
        header_bar.pack_end(menu_button)

        # Add about action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", lambda a, p: self.show_about_dialog())
        self.get_application().add_action(about_action)

        # Create a content container with scrolling support
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(450)  # Reasonable minimum height
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        # Create adaptive layout structure
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header_bar)
        toolbar_view.set_content(content_scroll)

        # Bottom navigation bar for wizard-style navigation
        action_bar = Gtk.ActionBar()

        # Move step indicator from header to action bar (footer) and center it
        self.step_label.set_hexpand(True)
        self.step_label.set_halign(Gtk.Align.CENTER)
        action_bar.set_center_widget(self.step_label)

        # Create back button (initially hidden)
        self.back_button = Gtk.Button()
        self.back_button.set_label(_("Back"))
        self.back_button.set_icon_name("go-previous-symbolic")
        self.back_button.connect("clicked", self.on_back_clicked)
        self.back_button.set_sensitive(False)
        self.back_button.set_visible(False)

        # Create next/finish button
        self.next_button = Gtk.Button()
        self.next_button.set_label(_("Start"))
        self.next_button.add_css_class("suggested-action")
        self.next_button.connect("clicked", self.on_next_clicked)

        # Add buttons to action bar
        action_bar.pack_start(self.back_button)
        action_bar.pack_end(self.next_button)

        # Add content with navigation controls
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(toolbar_view)
        main_box.append(action_bar)

        # Create pages - each in its own scrolled window for better space management
        settings_page = self.ui.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        terminal_page = self.ui.create_terminal_page()
        self.stack.add_named(terminal_page, "terminal")

        conclusion_page = self.ui.create_conclusion_page()
        self.stack.add_named(conclusion_page, "conclusion")

        # Store reference to the unified header for consistent access
        self.header_bar = header_bar

        self.stack.connect("notify::visible-child", self._on_page_changed)

        # Set initial view
        self.stack.set_visible_child_name("settings")

        # Set up content with toast overlay for notifications
        self.toast_overlay.set_child(main_box)
        self.set_content(self.toast_overlay)

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
                "<span font_size='small'>Step 1/3: Settings</span>"
            )
            self.back_button.set_sensitive(False)
            self.back_button.set_visible(False)
            self.next_button.set_label("Start Processing")
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

    def show_help(self, button: Gtk.Button) -> None:
        """Show help dialog with usage instructions

        Args:
            button: The button that triggered the event
        """
        help_dialog = Adw.MessageDialog(transient_for=self)
        help_dialog.set_heading(_("How to use Big OCR PDF"))

        help_content = _(
            "This application adds OCR to PDF files, making it possible to search for text.\n\n"
            "<b>Steps:</b>\n"
            "1. Configure the main language of the text in the documents\n"
            "2. Choose the desired quality and alignment\n"
            "3. Define where to save the processed files\n"
            "4. Click on 'Start Processing' to begin\n\n"
            "<b>Tips:</b>\n"
            "• Use 'Economic' for smaller files with good quality\n"
            "• The 'Auto rotate' option corrects pages in wrong orientation\n"
            "• You can process multiple files at once"
        )

        help_dialog.set_body(help_content)
        help_dialog.add_response("close", _("Close"))
        help_dialog.set_response_appearance("close", Adw.ResponseAppearance.DEFAULT)
        help_dialog.present()

    def show_about_dialog(self) -> None:
        """Show about dialog with application information"""
        # Get the application instance
        app = self.get_application()
        if app:
            app.on_about_action(None, None)  # File management methods

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
                    # Files added successfully, update the UI only
                    self.update_file_info()
                else:
                    # No valid files found, log the issue
                    logger.warning("No valid files were selected")
        except Exception as e:
            # Use the already imported logger instead of trying to import it again
            logger.error(f"Error adding files: {e}")

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

        # Show the folder selection dialog
        folder_chooser.select_folder(
            parent=self, cancellable=None, callback=self._on_select_folder_callback
        )

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

                # Log destination selection instead of showing toast
                logger.info(f"Destination folder selected: {path}")

        except Exception as e:
            logger.error(f"Error selecting save location: {e}")

    def on_apply_clicked(self, button: Gtk.Button) -> None:
        """Process the selected files with OCR

        Args:
            button: The button that triggered the event
        """
        # Validate there are files to process
        if not self.settings.selected_files:
            # Log error instead of showing toast
            logger.warning("No files selected for processing")
            return

        # Get the settings from UI
        if hasattr(self.ui, "lang_dropdown") and self.ui.lang_dropdown is not None:
            # Get the language code
            lang_index = self.ui.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        if (
            hasattr(self.ui, "quality_dropdown")
            and self.ui.quality_dropdown is not None
        ):
            # Get the quality setting
            quality_index = self.ui.quality_dropdown.get_selected()
            qualities = [
                ("normal", "Normal"),
                ("economic", "Economic"),
                ("economicplus", "More economic"),
            ]
            if 0 <= quality_index < len(qualities):
                self.settings.quality = qualities[quality_index][0]

        if (
            hasattr(self.ui, "alignment_dropdown")
            and self.ui.alignment_dropdown is not None
        ):
            # Get the alignment setting
            align_index = self.ui.alignment_dropdown.get_selected()
            alignments = [
                ("none", "Não alterar"),
                ("align", "Alinhar"),
                ("rotate", "Auto rotacionar"),
                ("alignrotate", "Alinhar e auto rotacionar"),
            ]
            if 0 <= align_index < len(alignments):
                self.settings.align = alignments[align_index][
                    0
                ]  # Check for "save in same folder" option
        save_in_same_folder = False
        if (
            hasattr(self.ui, "same_folder_switch_row")
            and self.ui.same_folder_switch_row
        ):
            save_in_same_folder = self.ui.same_folder_switch_row.get_active()

        # Get the destination folder
        if hasattr(self.ui, "dest_entry") and self.ui.dest_entry is not None:
            dest_folder = self.ui.dest_entry.get_text()
            if dest_folder:
                self.settings.destination_folder = dest_folder

        # Validate destination folder - skip validation if save_in_same_folder is enabled
        if not save_in_same_folder and not self.settings.destination_folder:
            # Log warning instead of showing toast
            logger.warning("No destination folder selected")
            return

        # Save all settings
        save_in_same_folder = False
        if (
            hasattr(self.ui, "same_folder_switch_row")
            and self.ui.same_folder_switch_row
        ):
            save_in_same_folder = self.ui.same_folder_switch_row.get_active()

        self.settings.save_settings(
            self.settings.lang,
            self.settings.quality,
            self.settings.align,
            self.settings.destination_folder,
            save_in_same_folder,
        )

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
            # Log error instead of showing toast
            logger.error("Failed to start OCR processing")
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

        # Processing has started - log instead of showing toast
        logger.info("OCR processing started using Python API")

    def reset_and_go_to_settings(self) -> None:
        """Reset the application state and return to the settings page"""
        # Clear processed files list
        self.processed_files = []

        # Reset UI elements
        self.next_button.set_label("Start")
        self.next_button.set_sensitive(True)
        self.next_button.set_visible(True)

        # Disconnect any previous handlers from the next button
        for handler_id in getattr(self.next_button, "handler_ids", []):
            self.next_button.disconnect(handler_id)

        # Connect the standard handler
        handler_id = self.next_button.connect("clicked", self.on_next_clicked)
        self.next_button.handler_ids = [handler_id]

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
        self.next_button.set_label("Back")
        self.next_button.set_visible(True)
        self.next_button.set_sensitive(True)

        # Disconnect any existing signal handlers
        if hasattr(self.next_button, "_handler_id") and self.next_button._handler_id:
            self.next_button.disconnect(self.next_button._handler_id)

    def update_file_info(self) -> None:
        """Update the file information UI after files have been added or removed"""
        # Store the current visible child name to restore later
        current_page = self.stack.get_visible_child_name()

        # Check if we can use the refresh method or need a full rebuild
        if current_page == "settings" and hasattr(self.ui, "refresh_queue_status"):
            # Use the lightweight refresh method if available
            self.ui.refresh_queue_status()
            logger.info(
                f"Queue status refreshed with {len(self.settings.selected_files)} files"
            )
        # Fallback to completely rebuilding the settings page
        elif current_page == "settings":
            # Get the current settings page
            old_page = self.stack.get_visible_child()

            # Create a new settings page with updated file list
            new_page = self.ui.create_settings_page()

            # Replace the old page with the new one
            self.stack.remove(old_page)
            self.stack.add_named(new_page, "settings")
            self.stack.set_visible_child_name("settings")

            # Log that the UI was updated using the already imported logger
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
        # With a unified headerbar, this styling could be optional
        # but we'll keep it for potential future styling needs
        if hasattr(self, "header_bar") and self.header_bar:
            self.header_bar.add_css_class("settings-page-header")
            self.header_bar.remove_css_class("default-header")

            # Apply a specific style if needed
            # This is now optional since we have a unified headerbar
            '''
            css_provider = Gtk.CssProvider()
            css_provider.load_from_string(b"""
                headerbar.settings-page-header {
                    /* Apply any special styling for settings page */
                }
            """)
            
            # Add the provider to the display (application-wide)
            Gtk.StyleContext.add_provider_for_display(
                self.header_bar.get_display(),
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 10,  # Higher priority
            )
            '''

    def _remove_headerbar_sidebar_style(self) -> None:
        """Remove style from the headerbar for non-settings pages"""
        # This effectively restores the default headerbar styling
        if hasattr(self, "header_bar") and self.header_bar:
            # Toggle the classes to reset to default style
            self.header_bar.remove_css_class("settings-page-header")
            self.header_bar.add_css_class("default-header")

            # No timeout needed as we're just toggling classes

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
        # Print the keys in the extracted_text dictionary for debugging
        logger.info(
            f"Current extracted_text keys: {list(self.settings.extracted_text.keys())}"
        )

        # Update the status bar with current progress
        file_count = self.ocr_processor.get_processed_count()
        total_files = self.ocr_processor.get_total_count()

        if hasattr(self.ui, "terminal_status_bar") and self.ui.terminal_status_bar:
            remaining = total_files - file_count
            self.ui.terminal_status_bar.set_markup(
                f"<b>Processing: {file_count}/{total_files}:</b> "
                + f"{file_count} file(s) completed • "
                + f"<b>{remaining}</b> remaining"
            )

        # Log processing complete for this file
        logger.info(
            f"Processed file {file_count}/{total_files}: {os.path.basename(input_file)}"
        )

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR"""
        logger.info("OCR processing complete callback triggered")

        # Check if we're still on the terminal page (might have been cancelled)
        if self.stack.get_visible_child_name() != "terminal":
            logger.info(
                "Processing complete but no longer on terminal page, likely cancelled"
            )
            return

        # Update the progress display to show 100%
        if hasattr(self.ui, "terminal_progress_bar") and self.ui.terminal_progress_bar:
            self.ui.terminal_progress_bar.set_fraction(1.0)
            self.ui.terminal_progress_bar.set_text("100%")

        # Update the status text
        if hasattr(self.ui, "terminal_status_bar") and self.ui.terminal_status_bar:
            # Get the total files processed from the OCR queue
            total_files = self.ocr_processor.get_processed_count()
            self.ui.terminal_status_bar.set_markup(
                f"<b>OCR processing complete!</b> {total_files} file(s) processed"
            )

        # Stop the spinner if it's still spinning
        if hasattr(self.ui, "terminal_spinner") and self.ui.terminal_spinner:
            self.ui.terminal_spinner.set_spinning(False)

        # First update the conclusion page with actual values
        if hasattr(self.ui, "update_conclusion_page"):
            self.ui.update_conclusion_page()

        # Show conclusion page with a short delay
        from gi.repository import GLib

        self.conclusion_timer_id = GLib.timeout_add(
            1000, lambda: self.show_conclusion_page()
        )

        # Log completion
        logger.info(
            f"OCR processing completed successfully for {self.ocr_processor.get_processed_count()} files"
        )

    def _start_progress_monitor(self) -> None:
        """Start monitoring the OCR progress"""
        # Set up a timer to update the progress
        from gi.repository import GLib

        self.progress_timer_id = GLib.timeout_add(
            250, lambda: self._update_ocr_progress()
        )

    def _update_ocr_progress(self) -> bool:
        """Update the OCR progress in the UI

        Returns:
            True to continue updating, False to stop
        """
        # Check if we're still on the terminal page
        if self.stack.get_visible_child_name() != "terminal":
            # If we've left the terminal page, stop the timer
            if hasattr(self, "progress_timer_id") and self.progress_timer_id:
                from .utils.timer import safe_remove_source

                safe_remove_source(self.progress_timer_id)
                self.progress_timer_id = None
            return False

        # Update the terminal UI with current progress
        progress = self.ocr_processor.get_progress()

        # Use a timebased animation if the progress is still at 0
        # This gives feedback to the user that processing is happening
        if progress == 0:
            if not hasattr(self, "_animation_progress"):
                self._animation_progress = 0.05
            else:
                # Oscillate between 0.05 and 0.15 to show activity
                if self._animation_progress >= 0.15:
                    self._animation_direction = -0.01
                elif self._animation_progress <= 0.05:
                    self._animation_direction = 0.01

                if hasattr(self, "_animation_direction"):
                    self._animation_progress += self._animation_direction
                else:
                    self._animation_direction = 0.01
                    self._animation_progress += self._animation_direction

            # Use animation progress during active processing
            animation_progress = self._animation_progress
        else:
            # Use real progress
            animation_progress = progress

        # Update progress bar
        if hasattr(self.ui, "terminal_progress_bar") and self.ui.terminal_progress_bar:
            self.ui.terminal_progress_bar.set_fraction(animation_progress)
            progress_percent = int(animation_progress * 100)
            self.ui.terminal_progress_bar.set_text(f"{progress_percent}%")

        # Update spinner to indicate activity
        if hasattr(self.ui, "terminal_spinner") and self.ui.terminal_spinner:
            if not self.ui.terminal_spinner.get_spinning():
                self.ui.terminal_spinner.set_spinning(True)

        # Update status text if terminal_status_bar exists
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

            # Show the current status
            if progress >= 1.0 or files_processed >= total_files:
                # Process is complete
                self.ui.terminal_status_bar.set_markup(
                    _(
                        "<b>OCR processing complete!</b> {total} file(s) processed • Total time: {time}"
                    ).format(total=total_files, time=time_str)
                )

                # Stop the spinner
                if hasattr(self.ui, "terminal_spinner") and self.ui.terminal_spinner:
                    self.ui.terminal_spinner.set_spinning(False)

                # Stop the timer
                if hasattr(self, "progress_timer_id") and self.progress_timer_id:
                    from .utils.timer import safe_remove_source

                    safe_remove_source(self.progress_timer_id)
                    self.progress_timer_id = None

                return False
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
                if elapsed_time < 5:
                    stage = _("starting conversion")
                elif elapsed_time < 10:
                    stage = _("analyzing document")
                elif elapsed_time < 15:
                    stage = _("applying OCR")
                elif elapsed_time < 20:
                    stage = _("processing texts")
                else:
                    stage = _("finalizing processing")

                self.ui.terminal_status_bar.set_markup(
                    _(
                        "<b>Processing file 1/{total}:</b> {stage} • Time: {time}"
                    ).format(total=total_files, stage=stage, time=time_str)
                )

        return True

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click during OCR processing"""
        # Stop the OCR queue
        if self.ocr_processor.ocr_queue:
            self.ocr_processor.ocr_queue.stop()
            logger.info("OCR processing cancelled by user")

            # Stop the progress monitor
            if hasattr(self, "progress_timer_id") and self.progress_timer_id:
                from .utils.timer import safe_remove_source

                safe_remove_source(self.progress_timer_id)
                self.progress_timer_id = None

            # Show a toast notification
            toast = Adw.Toast.new(_("OCR processing cancelled"))
            toast.set_timeout(3)
            self.toast_overlay.add_toast(toast)

            # Any partial output files would be inconsistent, so we don't save them
            # We could remove them here if needed

            # Reset the OCR processor status
            self.ocr_processor._processed_files = 0
            self.ocr_processor._progress = 0.0
            self.ocr_processor.ocr_queue = None

            # Go back to the settings page
            self.reset_and_go_to_settings()
