"""
BigOcrPdf - UI Manager

This module manages the creation of UI components and handles UI interactions.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, Gdk, Pango

import os
import time
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _


class BigOcrPdfUI:
    """Class to manage UI creation and interaction"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize UI manager

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI components references
        self.lang_dropdown = None
        self.quality_dropdown = None
        self.alignment_dropdown = None
        self.dest_entry = None
        self.terminal = None
        self.terminal_status_bar = None
        self.terminal_progress_bar = None
        self.terminal_spinner = None
        self.file_list_box = None
        self.remove_button = None
        self.drop_label = None

        # Results page components
        self.result_file_count = None
        self.result_page_count = None
        self.result_time = None
        self.result_file_size = None
        self.files_list_box = None

    def create_settings_page(self) -> Gtk.Widget:
        """Create the settings page for the application using a horizontal box layout

        Returns:
            A widget containing the settings UI with side-by-side layout
        """
        # Create a horizontal box layout for a unified appearance
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)

        # Create the configuration panel (left side)
        config_panel = self._create_config_panel()
        config_panel.set_size_request(400, -1)  # Set minimum width for config panel
        config_panel.set_margin_start(8)
        config_panel.set_hexpand(False)

        # Create the file queue panel (right side)
        file_queue_panel = self._create_file_queue_panel()
        file_queue_panel.set_hexpand(True)  # Allow file queue to expand horizontally

        # Add some spacing between the panels instead of a separator
        file_queue_panel.set_margin_end(8)

        # Add panels to the box without separator between them
        main_box.append(config_panel)
        main_box.append(file_queue_panel)

        return main_box

    def _create_config_panel(self) -> Gtk.Widget:
        """Create the configuration panel for the left sidebar

        Returns:
            A widget containing OCR settings and destination configuration
        """
        # OCR Settings section with outer container
        settings_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        settings_card.set_margin_top(8)
        settings_card.set_margin_bottom(8)
        settings_card.set_margin_start(8)
        settings_card.set_margin_end(8)
        settings_card.add_css_class("card")  # Add a header for destination
        dest_header = Gtk.Label(label=_("Configuration"))
        dest_header.add_css_class("heading")
        dest_header.set_margin_top(12)
        dest_header.set_margin_start(12)
        dest_header.set_halign(Gtk.Align.START)
        settings_card.append(dest_header)

        # Add "Save in same folder" using SwitchRow for better appearance
        self.same_folder_switch_row = Adw.SwitchRow(
            title=_("Save in same folder as original file")
        )
        self.same_folder_switch_row.add_css_class("action-row-config")
        self.same_folder_switch_row.set_margin_top(8)
        self.same_folder_switch_row.set_active(self.window.settings.save_in_same_folder)
        self.same_folder_switch_row.connect(
            "notify::active", self._on_same_folder_toggled
        )

        # Add a folder icon
        same_folder_icon = Gtk.Image.new_from_icon_name("folder-symbolic")
        self.same_folder_switch_row.add_prefix(same_folder_icon)

        settings_card.append(self.same_folder_switch_row)

        # Create a container for the destination input - will be shown/hidden based on switch state
        self.dest_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Add a subtitle for the destination folder
        dest_subtitle = Gtk.Label(label=_("Select the destination folder"))
        dest_subtitle.add_css_class("caption")
        dest_subtitle.add_css_class("dim-label")
        dest_subtitle.set_margin_top(8)
        dest_subtitle.set_margin_start(12)
        dest_subtitle.set_margin_end(12)
        self.dest_container.append(dest_subtitle)

        # Create a container for the file chooser
        dest_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dest_box.set_margin_bottom(12)
        dest_box.set_margin_start(12)
        dest_box.set_margin_end(12)

        # Create an entry with modern styling
        self.dest_entry = Gtk.Entry()
        self.dest_entry.set_text(self.window.settings.destination_folder)
        self.dest_entry.set_hexpand(True)

        # Add a folder icon to the entry using the correct method for GTK4
        folder_icon = Gio.ThemedIcon.new("folder-symbolic")
        self.dest_entry.set_icon_from_gicon(Gtk.EntryIconPosition.PRIMARY, folder_icon)

        # Browse button with text instead of icon
        self.browse_button = Gtk.Button()
        self.browse_button.set_label(_("Browse"))
        self.browse_button.connect("clicked", self.window.on_browse_clicked)

        # Add entry and button to container
        dest_box.append(self.dest_entry)
        dest_box.append(self.browse_button)
        self.dest_container.append(dest_box)
        settings_card.append(self.dest_container)

        # Initially hide or show the destination input based on switch state
        self.dest_container.set_visible(not self.window.settings.save_in_same_folder)

        # Add a button to configure PDF output options
        output_options_row = Adw.ActionRow(title=_("Output File Options"))
        output_options_row.add_css_class("action-row-config")

        # Add an icon for visual clarity
        options_icon = Gtk.Image.new_from_icon_name("document-edit-symbolic")
        output_options_row.add_prefix(options_icon)

        # Add a button to open the options dialog
        options_button = Gtk.Button()
        options_button.set_label(_("Configure"))
        options_button.connect(
            "clicked", lambda b: self.show_pdf_options_dialog(lambda _: None)
        )
        options_button.set_valign(Gtk.Align.CENTER)
        output_options_row.add_suffix(options_button)
        output_options_row.set_activatable_widget(options_button)

        settings_card.append(output_options_row)

        # Language selection with icon
        self.lang_dropdown = self._create_language_dropdown()
        lang_row = Adw.ActionRow(title=_("Text Language"))
        lang_row.add_css_class("action-row-config")
        lang_icon = Gtk.Image.new_from_icon_name("preferences-desktop-locale-symbolic")
        lang_row.add_prefix(lang_icon)
        lang_row.add_suffix(self.lang_dropdown)
        lang_row.set_activatable(True)
        settings_card.append(lang_row)

        # Quality selection with icon
        self.quality_dropdown = self._create_quality_dropdown()
        quality_row = Adw.ActionRow(title=_("Quality"))
        quality_row.add_css_class("action-row-config")
        quality_icon = Gtk.Image.new_from_icon_name("preferences-system-symbolic")
        quality_row.add_prefix(quality_icon)
        quality_row.add_suffix(self.quality_dropdown)
        quality_row.set_activatable(True)
        settings_card.append(quality_row)

        # Alignment selection with icon
        self.alignment_dropdown = self._create_alignment_dropdown()
        align_row = Adw.ActionRow(title=_("Alignment"))
        align_row.add_css_class("action-row-config-last")
        align_icon = Gtk.Image.new_from_icon_name("format-justify-fill-symbolic")
        align_row.add_prefix(align_icon)
        align_row.add_suffix(self.alignment_dropdown)
        align_row.set_activatable(True)
        settings_card.append(align_row)

        # Wrap in a scrolled window for better handling of different screen sizes
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_child(settings_card)

        return scrolled_window

    def _create_file_queue_panel(self) -> Gtk.Widget:
        """Create the file queue panel for the right side

        Returns:
            A widget containing the file queue UI using card-style layout
        """
        # Create a container for the file queue
        queue_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        queue_box.set_margin_top(8)
        queue_box.set_margin_bottom(8)
        queue_box.set_margin_start(8)
        queue_box.set_margin_end(8)
        queue_box.add_css_class("card")

        # Add a header for PDF Files Queue
        queue_header = Gtk.Label(label=_("PDF Files Queue"))
        queue_header.add_css_class("heading")
        queue_header.set_margin_top(12)
        queue_header.set_margin_start(12)
        queue_header.set_halign(Gtk.Align.START)
        queue_box.append(queue_header)

        # File info and controls in a consistent layout
        info_controls_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        info_controls_box.set_margin_start(12)
        info_controls_box.set_margin_end(12)
        info_controls_box.set_margin_top(8)

        # Display document counts and page information
        # Create status label (will be updated by _update_queue_status)
        self.status_label = Gtk.Label()
        self.status_label.add_css_class("caption")
        self.status_label.add_css_class("dim-label")
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        info_controls_box.append(self.status_label)

        # Initialize with current data
        self._update_queue_status()

        # Add Files button
        add_button = Gtk.Button()
        add_button.set_label(_("Add"))
        # add  css class for button important
        add_button.add_css_class("suggested-action")
        add_button.connect("clicked", self.window.on_add_file_clicked)
        info_controls_box.append(add_button)

        # Remove All Files button
        remove_button = Gtk.Button()
        remove_button.set_label(_("Remove All"))
        remove_button.connect("clicked", lambda _b: self._remove_all_files())
        remove_button.set_sensitive(len(self.window.settings.selected_files) > 0)
        self.remove_button = remove_button  # Store reference for later updates
        info_controls_box.append(remove_button)

        queue_box.append(info_controls_box)

        # Container for the files list
        file_list_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        file_list_container.set_margin_start(12)
        file_list_container.set_margin_end(12)
        file_list_container.set_margin_top(8)
        file_list_container.set_margin_bottom(12)

        # Create the list box with queue styling
        self.file_list_box = Gtk.ListBox()
        self.file_list_box.set_selection_mode(Gtk.SelectionMode.MULTIPLE)
        self.file_list_box.add_css_class("boxed-list")

        # Set up drag-and-drop target
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.set_gtypes([
            Gdk.FileList
        ])  # Focus on handling file lists for multiple file drop
        drop_target.connect("drop", self._on_drop)
        drop_target.connect("enter", self._on_drop_enter)
        drop_target.connect("leave", self._on_drop_leave)
        self.file_list_box.add_controller(drop_target)

        # Add files to the list box if there are any
        if self.window.settings.selected_files:
            self._populate_file_list()
        else:
            # Add a placeholder row when no files are selected
            placeholder_row = Adw.ActionRow()
            placeholder_row.set_title(_("No files selected"))
            placeholder_row.set_subtitle(_("Add PDF files for OCR processing"))
            placeholder_icon = Gtk.Image.new_from_icon_name("document-open-symbolic")
            placeholder_row.add_prefix(placeholder_icon)
            self.file_list_box.append(placeholder_row)

        # Create a scrolled window for the file list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)  # Allow vertical expansion to fill space

        # Add list box to scrolled window
        scrolled.set_child(self.file_list_box)
        file_list_container.append(scrolled)

        # Add the file list container to the queue box
        queue_box.append(file_list_container)

        # Add drop indicator label (hidden by default)
        self.drop_label = Gtk.Label(label=_("Drop PDF files here"))
        self.drop_label.add_css_class("drop-target")
        self.drop_label.add_css_class("dim-label")
        self.drop_label.set_margin_start(12)
        self.drop_label.set_margin_end(12)
        self.drop_label.set_margin_bottom(12)
        self.drop_label.set_visible(False)
        queue_box.append(self.drop_label)

        # Wrap in a scrolled window for better handling of different screen sizes
        queue_scrolled = Gtk.ScrolledWindow()
        queue_scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        queue_scrolled.set_child(queue_box)

        return queue_scrolled

    def create_terminal_page(self) -> Gtk.Box:
        """Create the processing page with progress display

        Returns:
            A Gtk.Box containing the processing UI
        """
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Create a card for progress visualization
        progress_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        progress_card.set_margin_bottom(8)
        progress_card.add_css_class("card")
        progress_card.set_vexpand(True)

        # Create a centered progress area
        progress_area = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        progress_area.set_valign(Gtk.Align.CENTER)
        progress_area.set_vexpand(True)
        progress_area.set_margin_start(24)
        progress_area.set_margin_end(24)
        progress_area.set_margin_bottom(24)

        # Add a large progress indicator
        self.terminal_progress_bar = Gtk.ProgressBar()
        self.terminal_progress_bar.set_show_text(True)
        self.terminal_progress_bar.set_text("0%")
        self.terminal_progress_bar.set_fraction(0)
        self.terminal_progress_bar.set_margin_bottom(8)

        # Add file icon and progress visualization
        pdf_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        pdf_icon.set_pixel_size(48)
        pdf_icon.set_margin_bottom(16)
        pdf_icon.set_halign(Gtk.Align.CENTER)

        # Add current file label
        current_file_label = Gtk.Label()
        current_file_label.set_markup("<big>" + _("Processing PDF files...") + "</big>")
        current_file_label.set_halign(Gtk.Align.CENTER)
        current_file_label.set_margin_bottom(24)

        # Create status label with file counter
        self.terminal_status_bar = Gtk.Label(label=_("Preparing processing..."))
        self.terminal_status_bar.add_css_class("body")
        self.terminal_status_bar.set_halign(Gtk.Align.CENTER)
        self.terminal_status_bar.set_margin_bottom(8)

        # Add all elements to the progress area
        progress_area.append(pdf_icon)
        progress_area.append(current_file_label)
        progress_area.append(self.terminal_progress_bar)
        progress_area.append(self.terminal_status_bar)

        # Add a cancel button
        cancel_button = Gtk.Button()
        cancel_button.set_label(_("Cancel"))
        cancel_button.add_css_class("destructive-action")
        cancel_button.set_halign(Gtk.Align.CENTER)
        cancel_button.set_margin_top(16)
        cancel_button.connect("clicked", lambda b: self.window.on_cancel_clicked())
        progress_area.append(cancel_button)

        # Add to the progress card
        progress_card.append(progress_area)

        # Add to the main box
        main_box.append(progress_card)

        return main_box

    def create_conclusion_page(self) -> Gtk.Box:
        """Create the conclusion page showing OCR processing results"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        # Set individual margins
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Add summary card
        summary_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        summary_card.add_css_class("card")
        # Card header
        card_header = Gtk.Label(label=_("Processing Summary"))
        card_header.add_css_class("heading")
        card_header.set_halign(Gtk.Align.START)
        card_header.set_margin_top(16)
        card_header.set_margin_start(16)
        summary_card.append(card_header)

        # Create a horizontal box to contain two columns
        columns_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=32)
        columns_box.set_margin_start(16)
        columns_box.set_margin_end(16)
        columns_box.set_margin_bottom(16)
        columns_box.set_margin_top(8)

        # Left column - Files and Pages
        left_grid = Gtk.Grid()
        left_grid.set_column_spacing(16)
        left_grid.set_row_spacing(16)
        left_grid.set_hexpand(True)
        left_grid.set_halign(Gtk.Align.START)

        # Files processed
        files_icon = Gtk.Image.new_from_icon_name("document-multiple-symbolic")
        files_icon.set_pixel_size(16)
        left_grid.attach(files_icon, 0, 0, 1, 1)

        files_label = Gtk.Label(label=_("Files processed:"))
        files_label.set_halign(Gtk.Align.START)
        files_label.set_margin_start(8)
        left_grid.attach(files_label, 1, 0, 1, 1)

        self.result_file_count = Gtk.Label(label="0")
        self.result_file_count.set_halign(Gtk.Align.END)
        self.result_file_count.add_css_class("heading")
        self.result_file_count.set_hexpand(True)
        left_grid.attach(self.result_file_count, 2, 0, 1, 1)

        # Total pages
        pages_icon = Gtk.Image.new_from_icon_name("view-paged-symbolic")
        pages_icon.set_pixel_size(16)
        left_grid.attach(pages_icon, 0, 1, 1, 1)

        pages_label = Gtk.Label(label=_("Total pages:"))
        pages_label.set_halign(Gtk.Align.START)
        pages_label.set_margin_start(8)
        left_grid.attach(pages_label, 1, 1, 1, 1)

        self.result_page_count = Gtk.Label(label="0")
        self.result_page_count.set_halign(Gtk.Align.END)
        self.result_page_count.add_css_class("heading")
        left_grid.attach(self.result_page_count, 2, 1, 1, 1)

        # Right column - Time and Size
        right_grid = Gtk.Grid()
        right_grid.set_column_spacing(16)
        right_grid.set_row_spacing(16)
        right_grid.set_hexpand(True)
        right_grid.set_halign(Gtk.Align.START)

        # Processing time
        time_icon = Gtk.Image.new_from_icon_name("clock-symbolic")
        time_icon.set_pixel_size(16)
        right_grid.attach(time_icon, 0, 0, 1, 1)

        time_label = Gtk.Label(label=_("Processing time:"))
        time_label.set_halign(Gtk.Align.START)
        time_label.set_margin_start(8)
        right_grid.attach(time_label, 1, 0, 1, 1)

        self.result_time = Gtk.Label(label="00:00")
        self.result_time.set_halign(Gtk.Align.END)
        self.result_time.add_css_class("heading")
        self.result_time.set_hexpand(True)
        right_grid.attach(self.result_time, 2, 0, 1, 1)

        # Output file size
        size_icon = Gtk.Image.new_from_icon_name("drive-harddisk-symbolic")
        size_icon.set_pixel_size(16)
        right_grid.attach(size_icon, 0, 1, 1, 1)

        size_label = Gtk.Label(label=_("File size:"))
        size_label.set_halign(Gtk.Align.START)
        size_label.set_margin_start(8)
        right_grid.attach(size_label, 1, 1, 1, 1)

        self.result_file_size = Gtk.Label(label="0 KB")
        self.result_file_size.set_halign(Gtk.Align.END)
        self.result_file_size.add_css_class("heading")
        right_grid.attach(self.result_file_size, 2, 1, 1, 1)

        # Add the two grids to the columns box
        columns_box.append(left_grid)
        columns_box.append(right_grid)

        # Add the columns box to the summary card
        summary_card.append(columns_box)
        main_box.append(summary_card)

        # Output files section
        files_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        files_card.add_css_class("card")
        files_card.set_margin_top(16)
        files_card.set_margin_bottom(16)

        # Card header
        files_header = Gtk.Label(label=_("Generated Files"))
        files_header.add_css_class("heading")
        files_header.set_halign(Gtk.Align.START)
        files_header.set_margin_top(16)
        files_header.set_margin_start(16)
        files_card.append(files_header)

        # Create scrolled window for output files
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(100)
        scrolled.set_max_content_height(200)
        scrolled.set_margin_start(16)
        scrolled.set_margin_end(16)
        scrolled.set_margin_bottom(16)

        # Create list box for output files
        self.output_list_box = Gtk.ListBox()
        self.output_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.output_list_box.add_css_class("boxed-list")
        self.output_list_box.add_css_class(
            "output-files-list"
        )  # Add custom class for styling hover effect
        self.output_list_box.set_vexpand(True)

        scrolled.set_child(self.output_list_box)
        files_card.append(scrolled)

        main_box.append(files_card)

        return main_box

    def _populate_file_list(self) -> None:
        """Populate the file list box with the selected files"""
        # Remove existing items
        while True:
            child = self.file_list_box.get_first_child()
            if child:
                self.file_list_box.remove(child)
            else:
                break

        # Add each file as a row
        for idx, file_path in enumerate(self.window.settings.selected_files):
            row = Adw.ActionRow()

            # Set file name as title
            file_name = os.path.basename(file_path)
            row.set_title(file_name)

            # Add directory as subtitle
            dir_name = os.path.dirname(file_path)
            row.set_subtitle(dir_name)

            # Add page count if available
            try:
                result = subprocess.run(
                    ["pdfinfo", file_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for line in result.stdout.split("\n"):
                    if line.startswith("Pages:"):
                        pages = int(line.split(":")[1].strip())
                        page_label = Gtk.Label()
                        page_label.set_markup(f"<small>{pages} pg.</small>")
                        row.add_suffix(page_label)
            except Exception:
                pass

            # Add file icon
            file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
            row.add_prefix(file_icon)

            # Store the file path using Python attributes instead of set_data
            row.file_path = file_path
            row.file_index = idx

            # Add trash icon button for individual file removal
            trash_button = Gtk.Button()
            trash_button.set_icon_name("user-trash-symbolic")
            trash_button.set_tooltip_text("Remove from queue")
            trash_button.add_css_class("flat")
            trash_button.add_css_class("circular")
            trash_button.set_valign(Gtk.Align.CENTER)
            trash_button.connect(
                "clicked", lambda _b, idx=idx: self._remove_single_file(idx)
            )
            row.add_suffix(trash_button)

            # Add the row to the list box
            self.file_list_box.append(row)

        # Update remove button sensitivity
        if hasattr(self, "remove_button"):
            self.remove_button.set_sensitive(
                len(self.window.settings.selected_files) > 0
            )

    def update_conclusion_page(self) -> None:
        """Update the conclusion page with results from OCR processing"""
        if not hasattr(self, "result_file_count") or not self.result_file_count:
            logger.warning("Conclusion page components not initialized")
            return

        # Make sure extracted_text dictionary is initialized
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Get processed file count from OCR processor
        file_count = self.window.ocr_processor.get_processed_count()
        self.result_file_count.set_text(str(file_count))

        # Calculate total page count
        page_count = 0
        total_file_size = 0

        # Clear output file list
        while True:
            child = self.output_list_box.get_first_child()
            if child:
                self.output_list_box.remove(child)
            else:
                break

        # Process output files and gather stats
        for output_file in self.window.settings.processed_files:
            # Skip if file doesn't exist or wasn't created
            if not os.path.exists(output_file):
                continue

            # Get filename for display
            filename = os.path.basename(output_file)

            # Verify the file was created during this OCR run (within the last 5 minutes)
            try:
                file_creation_time = os.path.getctime(output_file)
                if time.time() - file_creation_time > 300:  # 300 seconds = 5 minutes
                    logger.warning(
                        f"Skipping {filename} as it was not created in this OCR run"
                    )
                    continue
            except:
                # If we can't get creation time, we'll trust the filename
                pass

            # Get file info
            try:
                # Get file size
                file_size = os.path.getsize(output_file)
                total_file_size += file_size

                # Get page count
                result = subprocess.run(
                    ["pdfinfo", output_file],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                for line in result.stdout.split("\n"):
                    if line.startswith("Pages:"):
                        pages = int(line.split(":")[1].strip())
                        page_count += pages
                        break

                # Add file to list
                row = Adw.ActionRow()
                row.set_title(os.path.basename(output_file))
                row.set_subtitle(os.path.dirname(output_file))

                # Add page count
                page_label = Gtk.Label()
                page_label.set_markup(f"<small>{pages} pg.</small>")
                row.add_suffix(page_label)

                # Add size
                size_label = Gtk.Label()
                size_label.set_markup(f"<small>{self._format_size(file_size)}</small>")
                row.add_suffix(size_label)

                # Add file icon
                file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
                row.add_prefix(file_icon)

                # Create a button container for better spacing and organization
                button_container = Gtk.Box(
                    orientation=Gtk.Orientation.HORIZONTAL, spacing=4
                )
                button_container.set_halign(Gtk.Align.END)

                # Add open button
                open_button = Gtk.Button()
                open_button.set_icon_name("document-open-symbolic")
                open_button.set_tooltip_text(_("Open file"))
                open_button.add_css_class("circular")
                open_button.set_valign(Gtk.Align.CENTER)
                open_button.set_margin_end(12)
                open_button.set_margin_start(12)

                open_button.connect(
                    "clicked", lambda _b, f=output_file: self._open_file(f)
                )
                button_container.append(open_button)

                # Add view text button if we have extracted text for this file
                logger.info(
                    f"Checking if {output_file} is in extracted_text keys: {list(self.window.settings.extracted_text.keys())}"
                )

                # Ensure the extracted_text dictionary exists
                if not hasattr(self.window.settings, "extracted_text"):
                    self.window.settings.extracted_text = {}

                # Try to load any extracted text if not already in memory
                if not (
                    output_file in self.window.settings.extracted_text
                    and self.window.settings.extracted_text[output_file]
                ):
                    # Check for sidecar text file (regular location)
                    sidecar_file = os.path.splitext(output_file)[0] + ".txt"
                    if os.path.exists(sidecar_file):
                        try:
                            with open(sidecar_file, "r", encoding="utf-8") as f:
                                extracted_text = f.read()
                                self.window.settings.extracted_text[output_file] = (
                                    extracted_text
                                )
                                logger.info(
                                    f"Loaded {len(extracted_text)} chars from sidecar file for {os.path.basename(output_file)}"
                                )
                        except Exception as e:
                            logger.error(f"Error reading sidecar file: {e}")

                # Store text length for logging
                if output_file in self.window.settings.extracted_text:
                    text_length = len(self.window.settings.extracted_text[output_file])
                    if text_length > 0:
                        logger.info(
                            f"Text available for {os.path.basename(output_file)} ({text_length} chars)"
                        )

                # Always show the button regardless of whether text was loaded
                # per user's request to maintain previous behavior

                # Always add the text button regardless of whether we found extracted text
                # This ensures backward compatibility with previous versions
                logger.info(
                    f"Adding View Text button for {os.path.basename(output_file)}"
                )
                # Create button with more prominent styling
                text_button = Gtk.Button()
                # Try a different icon that's more universally available
                text_button.set_icon_name(
                    "format-text-uppercase-symbolic"
                )  # Text editor icon
                text_button.set_tooltip_text(_("View extracted text"))
                text_button.add_css_class("circular")
                text_button.set_valign(Gtk.Align.CENTER)
                text_button.connect(
                    "clicked",
                    lambda _b, f=output_file: self._show_extracted_text(f),
                )
                # Add to button container instead of directly to row
                button_container.append(text_button)

                # Add the button container to the row
                row.add_suffix(button_container)

                # Add the row to the list box
                self.output_list_box.append(row)
            except Exception as e:
                logger.error(f"Error processing output file {output_file}: {e}")

        # Update page count
        self.result_page_count.set_text(str(page_count))

        # Update processing time
        if hasattr(self.window, "process_start_time"):
            elapsed_time = time.time() - self.window.process_start_time
            minutes = int(elapsed_time / 60)
            seconds = int(elapsed_time % 60)
            self.result_time.set_text(f"{minutes:02d}:{seconds:02d}")
        else:
            self.result_time.set_text("--:--")

        # Update file size
        self.result_file_size.set_text(self._format_size(total_file_size))

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        # Handle edge cases
        if size_bytes < 0:
            return "0 B"

        # Define size units
        units = ["B", "KB", "MB", "GB", "TB"]

        # Determine unit to use
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024
            i += 1

        # Format with appropriate precision
        if i == 0:  # Bytes
            return f"{int(size_bytes)} {units[i]}"
        else:
            return f"{size_bytes:.2f} {units[i]}"

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application

        Args:
            file_path: Path to the file to open
        """
        try:
            subprocess.Popen(["xdg-open", file_path])
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")

    def _create_language_dropdown(self) -> Gtk.DropDown:
        """Create the language dropdown with available OCR languages

        Returns:
            A Gtk.DropDown widget configured with available languages
        """
        dropdown = Gtk.DropDown()
        string_list = Gtk.StringList()

        # Get available languages
        languages = self.window.ocr_processor.get_available_ocr_languages()
        default_index = 0

        for i, (lang_code, lang_name) in enumerate(languages):
            string_list.append(lang_name)
            # Set default based on current setting
            if lang_code == self.window.settings.lang:
                default_index = i

        dropdown.set_model(string_list)
        dropdown.set_valign(Gtk.Align.CENTER)
        dropdown.set_selected(default_index)
        return dropdown

    def _create_quality_dropdown(self) -> Gtk.DropDown:
        """Create the quality dropdown

        Returns:
            A Gtk.DropDown widget configured with quality options
        """
        dropdown = Gtk.DropDown()
        string_list = Gtk.StringList()

        qualities = [
            ("normal", "Normal"),
            ("economic", "Economic"),
            ("economicplus", "More economic"),
        ]

        default_index = 0
        for i, (quality_code, quality_name) in enumerate(qualities):
            string_list.append(quality_name)
            if quality_code == self.window.settings.quality:
                default_index = i

        dropdown.set_model(string_list)
        dropdown.set_valign(Gtk.Align.CENTER)
        dropdown.set_selected(default_index)
        return dropdown

    def _create_alignment_dropdown(self) -> Gtk.DropDown:
        """Create the alignment dropdown with tooltips

        Returns:
            A Gtk.DropDown widget configured with alignment options and tooltips
        """
        dropdown = Gtk.DropDown()
        string_list = Gtk.StringList()

        alignments = [
            ("none", _("Don't change")),
            ("align", _("Align")),
            ("rotate", _("Auto rotate")),
            ("alignrotate", _("Align + rotate")),  # Texto encurtado
        ]

        # Default to "Align + rotate" (index 3) unless another setting is found
        default_index = 3  # Index of "Align + rotate"
        for i, (align_code, align_name) in enumerate(alignments):
            string_list.append(align_name)
            if align_code == self.window.settings.align:
                default_index = i

        dropdown.set_model(string_list)
        dropdown.set_valign(Gtk.Align.CENTER)
        dropdown.set_selected(default_index)
        
        # Set initial tooltip
        dropdown.set_tooltip_text(
            _("Choose how to handle page alignment and rotation during OCR processing")
        )
        
        # Connect signal to update tooltip based on selection
        def on_alignment_selection_changed(dropdown_widget, param):
            selected_index = dropdown_widget.get_selected()
            if 0 <= selected_index < len(self.window.ALIGNMENT_TOOLTIPS):
                tooltip_text = self.window.ALIGNMENT_TOOLTIPS[selected_index]
                dropdown_widget.set_tooltip_text(tooltip_text)
        
        dropdown.connect("notify::selected", on_alignment_selection_changed)
        
        # Set initial detailed tooltip
        if len(self.window.ALIGNMENT_TOOLTIPS) > default_index:
            dropdown.set_tooltip_text(self.window.ALIGNMENT_TOOLTIPS[default_index])
        
        return dropdown

    def _update_queue_status(self) -> None:
        """Update the status label with current file and page information"""
        if not hasattr(self, "status_label"):
            return

        file_count = len(self.window.settings.selected_files)
        if file_count > 0:
            status_info = (
                f"{file_count} {'file' if file_count == 1 else 'files'} selected"
            )
            status_info += f" • {self.window.settings.pages_count} {'page' if self.window.settings.pages_count == 1 else 'pages'} in total"
        else:
            status_info = _("No files selected")

        self.status_label.set_text(status_info)

    def _on_drop(
        self, drop_target: Gtk.DropTarget, value, _x: float, _y: float
    ) -> bool:
        """Handle file drop events for both single and multiple files

        Args:
            drop_target: The drop target widget
            value: The dropped file(s) - could be Gio.File or Gdk.FileList
            _x: X coordinate of the drop
            _y: Y coordinate of the drop

        Returns:
            True if the drop was handled successfully, False otherwise
        """
        try:
            file_paths = []

            # Handle different types of drop values
            if isinstance(value, Gio.File):
                # Single file case
                file_path = value.get_path()
                if file_path:
                    file_paths.append(file_path)
            elif isinstance(value, list) or hasattr(value, "__iter__"):
                # Multiple files case
                for file in value:
                    if isinstance(file, Gio.File):
                        file_path = file.get_path()
                        if file_path:
                            file_paths.append(file_path)
            else:
                logger.warning(f"Unsupported drop value type: {type(value)}")
                return False

            if not file_paths:
                logger.warning("No valid files in drop data")
                return False

            # Filter for PDF files only
            pdf_file_paths = []
            for file_path in file_paths:
                if not file_path.lower().endswith(".pdf"):
                    logger.warning(f"Ignoring non-PDF file: {file_path}")
                    continue

                if not os.path.exists(file_path):
                    logger.warning(f"Ignoring nonexistent file: {file_path}")
                    continue

                pdf_file_paths.append(file_path)

            if not pdf_file_paths:
                logger.warning("No valid PDF files in drop data")
                return False

            logger.info(f"{len(pdf_file_paths)} PDF files dropped")

            # Add all valid PDF files to the queue
            added = self.window.settings.add_files(pdf_file_paths)

            if added > 0:
                # Update the UI
                self._populate_file_list()
                self._update_queue_status()

                # Hide the drop indicator
                if self.drop_label:
                    self.drop_label.set_visible(False)

                return True

            return False
        except Exception as e:
            logger.error(f"Error handling dropped file(s): {e}")
            return False

    def _on_drop_enter(
        self, drop_target: Gtk.DropTarget, _x: float, _y: float
    ) -> Gdk.DragAction:
        """Handle when files are dragged over the drop area

        Args:
            drop_target: The drop target widget
            _x: X coordinate
            _y: Y coordinate

        Returns:
            The drag action to perform (COPY)
        """
        # Show the drop indicator
        if self.drop_label:
            self.drop_label.set_visible(True)

        return Gdk.DragAction.COPY

    def _on_drop_leave(self, drop_target: Gtk.DropTarget) -> None:
        """Handle when files are dragged away from the drop area

        Args:
            drop_target: The drop target widget
        """
        # Hide the drop indicator
        if self.drop_label:
            self.drop_label.set_visible(False)

    def _remove_single_file(self, idx: int) -> None:
        """Remove a single file from the list

        Args:
            idx: Index of the file to remove
        """
        # Check if the index is valid
        if idx < 0 or idx >= len(self.window.settings.selected_files):
            return

        # Remove the file from the settings
        file_path = self.window.settings.selected_files.pop(idx)
        logger.info(f"Removed file: {file_path}")

        # Update pages count (if available)
        try:
            result = subprocess.run(
                ["pdfinfo", file_path],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    pages = int(line.split(":")[1].strip())
                    self.window.settings.pages_count -= pages
                    break
        except Exception:
            pass

        # Refresh the list
        self._populate_file_list()

        # Update the status information with the new file and page counts
        self._update_queue_status()

        # Update the remove all button sensitivity
        self.remove_button.set_sensitive(len(self.window.settings.selected_files) > 0)

    def _remove_all_files(self) -> None:
        """Remove all files from the queue"""
        # Check if there are any files
        if not self.window.settings.selected_files:
            return

        # Log the action
        logger.info(
            f"Removing all {len(self.window.settings.selected_files)} files from queue"
        )

        # Clear all files
        self.window.settings.selected_files.clear()

        # Reset page count
        self.window.settings.pages_count = 0

        # Refresh the list
        self._populate_file_list()

        # Update the status information with the new file and page counts
        self._update_queue_status()

        # Update the remove all button sensitivity
        self.remove_button.set_sensitive(False)

    def refresh_queue_status(self) -> None:
        """Update the queue status without rebuilding the entire settings page"""
        if hasattr(self, "status_label") and self.status_label:
            self._update_queue_status()

        if hasattr(self, "remove_button") and self.remove_button:
            self.remove_button.set_sensitive(
                len(self.window.settings.selected_files) > 0
            )

        # Update destination container visibility based on switch row state
        if hasattr(self, "same_folder_switch_row") and hasattr(self, "dest_container"):
            switch_state = self.same_folder_switch_row.get_active()
            self.dest_container.set_visible(not switch_state)

        # Also update the file list if needed
        if hasattr(self, "file_list_box") and self.file_list_box:
            self._populate_file_list()

    def _on_same_folder_toggled(self, switch_row: Adw.SwitchRow, param_spec) -> None:
        """Handle toggling of the 'save in same folder' switch row

        Args:
            switch_row: The switch row that was toggled
            param_spec: The parameter specification
        """
        # Get the active state from the switch row
        is_active = switch_row.get_active()

        # Show/hide the destination entry container based on switch state
        if hasattr(self, "dest_container") and self.dest_container:
            # If save in same folder is enabled, hide the destination entry
            self.dest_container.set_visible(not is_active)

        # Store the setting in the window's settings object
        self.window.settings.save_in_same_folder = is_active

    def _on_suffix_changed(self, entry: Gtk.Entry) -> None:
        """Handle changes to the PDF suffix entry

        Args:
            entry: The suffix entry widget
        """
        # Get the text from the entry
        suffix = entry.get_text().strip()

        # Update the settings object
        self.window.settings.pdf_suffix = suffix or "ocr"  # Default to "ocr" if empty

    def _on_overwrite_toggled(self, switch_row: Adw.SwitchRow, param_spec) -> None:
        """Handle toggling of the 'overwrite existing files' switch row

        Args:
            switch_row: The switch row that was toggled
            param_spec: The parameter specification
        """
        # Get the active state from the switch row
        is_active = switch_row.get_active()

        # Update the settings object
        self.window.settings.overwrite_existing = is_active

    def _on_date_toggled(self, switch_row: Adw.SwitchRow, param_spec) -> None:
        """Handle toggling of the 'include date in filename' switch row

        Args:
            switch_row: The switch row that was toggled
            param_spec: The parameter specification
        """
        # Get the active state from the switch row
        is_active = switch_row.get_active()

        # Update the settings object
        self.window.settings.include_date = is_active

        # Enable/disable date element checkboxes
        if hasattr(self, "date_elements_checks"):
            for check in self.date_elements_checks.values():
                check.set_sensitive(is_active)

    def _on_year_toggled(self, check_button: Gtk.CheckButton) -> None:
        """Handle toggling of the year checkbox

        Args:
            check_button: The year checkbox
        """
        self.window.settings.include_year = check_button.get_active()

    def _on_month_toggled(self, check_button: Gtk.CheckButton) -> None:
        """Handle toggling of the month checkbox

        Args:
            check_button: The month checkbox
        """
        self.window.settings.include_month = check_button.get_active()

    def _on_day_toggled(self, check_button: Gtk.CheckButton) -> None:
        """Handle toggling of the day checkbox

        Args:
            check_button: The day checkbox
        """
        self.window.settings.include_day = check_button.get_active()

    def _on_time_toggled(self, check_button: Gtk.CheckButton) -> None:
        """Handle toggling of the time checkbox

        Args:
            check_button: The time checkbox
        """
        self.window.settings.include_time = check_button.get_active()

    def _show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a dialog

        Args:
            file_path: Path to the PDF file
        """
        # Make sure extracted_text dictionary exists
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Make sure we have text for this file
        if file_path not in self.window.settings.extracted_text:
            logger.warning(
                _("No extracted text available for {0}").format(
                    os.path.basename(file_path)
                )
            )
            # Try to read from the sidecar text file if it exists
            sidecar_file = os.path.splitext(file_path)[0] + ".txt"
            if os.path.exists(sidecar_file):
                try:
                    with open(sidecar_file, "r", encoding="utf-8") as f:
                        self.window.settings.extracted_text[file_path] = f.read()
                    logger.info(
                        _("Read {0} characters from sidecar file").format(
                            len(self.window.settings.extracted_text[file_path])
                        )
                    )
                except Exception as e:
                    logger.error(f"Error reading sidecar file: {e}")
                    # Show error dialog
                    dialog = Adw.MessageDialog(transient_for=self.window)
                    dialog.set_heading(_("Text Not Available"))
                    dialog.set_body(
                        _(
                            "Could not read extracted text for this file. Error: {0}"
                        ).format(str(e))
                    )
                    dialog.add_response("ok", _("OK"))
                    dialog.present()
                    return
            else:
                # Try looking for a temporary file in the .temp directory first
                temp_dir = os.path.join(os.path.dirname(file_path), ".temp")
                if os.path.exists(temp_dir):
                    temp_filename = (
                        f"temp_{os.path.basename(os.path.splitext(file_path)[0])}.txt"
                    )
                    temp_sidecar = os.path.join(temp_dir, temp_filename)
                    if os.path.exists(temp_sidecar):
                        try:
                            with open(temp_sidecar, "r", encoding="utf-8") as f:
                                self.window.settings.extracted_text[file_path] = (
                                    f.read()
                                )
                            logger.info(f"Found text in temporary file: {temp_sidecar}")
                            return
                        except Exception as e:
                            logger.error(f"Error reading temp sidecar file: {e}")

                # If no temporary file exists, provide a generic message
                # This ensures we always display something when the user clicks "View Text"
                self.window.settings.extracted_text[file_path] = (
                    "OCR processing was completed for this file, but the extracted text could not be found."
                )

        extracted_text = self.window.settings.extracted_text[file_path]
        if not extracted_text or not extracted_text.strip():
            # Even if no text was extracted, provide a generic message
            extracted_text = "No text content was detected in this file during OCR processing.\n\nThis could happen if the PDF contains only images without recognizable text or if OCR processing encountered issues."

        # Create a dialog to display the text
        dialog = Gtk.Window()
        dialog.set_title(f"Extracted Text - {os.path.basename(file_path)}")
        dialog.set_default_size(700, 500)
        dialog.set_modal(True)
        dialog.set_transient_for(self.window)

        # Create a box for the content
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content_box.set_margin_top(16)
        content_box.set_margin_bottom(16)
        content_box.set_margin_start(16)
        content_box.set_margin_end(16)

        # Add a search box for large texts
        search_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_box.set_margin_bottom(8)

        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text(_("Search in text..."))
        search_entry.set_hexpand(True)
        search_box.append(search_entry)

        # Add previous/next search buttons
        prev_button = Gtk.Button()
        prev_button.set_icon_name("go-up-symbolic")
        prev_button.set_tooltip_text(_("Find previous match"))
        prev_button.add_css_class("flat")
        prev_button.set_sensitive(False)  # Initially disabled
        search_box.append(prev_button)

        next_button = Gtk.Button()
        next_button.set_icon_name("go-down-symbolic")
        next_button.set_tooltip_text(_("Find next match"))
        next_button.add_css_class("flat")
        next_button.set_sensitive(False)  # Initially disabled
        search_box.append(next_button)

        # Add the search box to the content
        content_box.append(search_box)

        # Create a scrolled window to contain the text
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        # Create a text view to display the text
        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        text_view.set_monospace(True)
        text_view.set_left_margin(12)  # Add some margin for better readability
        text_view.set_right_margin(12)
        text_view.set_top_margin(12)
        text_view.set_bottom_margin(12)

        # Add the text to the text view
        buffer = text_view.get_buffer()
        buffer.set_text(extracted_text)

        # Add text tags for search highlighting
        tag_table = buffer.get_tag_table()
        highlight_tag = Gtk.TextTag.new("search_highlight")
        highlight_tag.set_property("background", "#ffff00")  # Yellow highlight
        highlight_tag.set_property("foreground", "#000000")  # Black text
        tag_table.add(highlight_tag)

        # Add a status label for search results
        status_label = Gtk.Label()
        status_label.set_halign(Gtk.Align.START)
        status_label.add_css_class("caption")
        status_label.add_css_class("dim-label")

        # Set up search functionality
        search_positions = []
        current_match = -1

        def on_search_changed(entry):
            nonlocal search_positions, current_match
            search_text = entry.get_text().lower()

            # Clear previous highlights
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            buffer.remove_tag_by_name("search_highlight", start_iter, end_iter)

            # Reset search positions
            search_positions = []
            current_match = -1

            if not search_text:
                prev_button.set_sensitive(False)
                next_button.set_sensitive(False)
                status_label.set_text("")
                return

            # Find all occurrences of the search text
            text = extracted_text.lower()
            pos = 0
            while True:
                pos = text.find(search_text, pos)
                if pos == -1:
                    break
                search_positions.append(pos)
                pos += 1

            # Highlight all matches
            for pos in search_positions:
                start_iter = buffer.get_iter_at_offset(pos)
                end_iter = buffer.get_iter_at_offset(pos + len(search_text))
                buffer.apply_tag_by_name("search_highlight", start_iter, end_iter)

            # Update status and buttons
            match_count = len(search_positions)
            status_label.set_text(_("{0} matches found").format(match_count))

            prev_button.set_sensitive(match_count > 0)
            next_button.set_sensitive(match_count > 0)

            # Go to first match
            if match_count > 0:
                goto_match(0)

        def goto_match(index):
            nonlocal current_match
            if not search_positions or index < 0 or index >= len(search_positions):
                return

            current_match = index
            pos = search_positions[index]

            # Scroll to the match
            match_iter = buffer.get_iter_at_offset(pos)
            text_view.scroll_to_iter(match_iter, 0.2, False, 0.0, 0.0)

            # Update status
            status_label.set_text(
                _("Match {0} of {1}").format(index + 1, len(search_positions))
            )

        def on_prev_clicked(_button):
            nonlocal current_match
            if current_match > 0:
                goto_match(current_match - 1)

        def on_next_clicked(_button):
            nonlocal current_match
            if current_match < len(search_positions) - 1:
                goto_match(current_match + 1)

        # Connect search signals
        search_entry.connect("search-changed", on_search_changed)
        prev_button.connect("clicked", on_prev_clicked)
        next_button.connect("clicked", on_next_clicked)

        # Add the text view to the scrolled window
        scrolled.set_child(text_view)

        # Add a status box to contain both the status label and buttons
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        status_box.set_margin_top(16)

        # Add the status label to the left side
        status_box.append(status_label)
        status_label.set_hexpand(True)

        # Add buttons to the right side
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        button_box.set_halign(Gtk.Align.END)

        # Add copy to clipboard button
        copy_button = Gtk.Button(label=_("Copy to Clipboard"))
        copy_button.set_tooltip_text(_("Copy the entire text to clipboard"))
        copy_button.connect(
            "clicked", lambda b: self._copy_text_to_clipboard(extracted_text)
        )
        button_box.append(copy_button)

        # Add save to file button
        save_button = Gtk.Button(label=_("Save as TXT"))
        save_button.set_tooltip_text(_("Save the extracted text to a .txt file"))
        save_button.connect(
            "clicked", lambda b: self._save_text_to_file(extracted_text, file_path)
        )
        button_box.append(save_button)

        # Add close button
        close_button = Gtk.Button(label=_("Close"))
        close_button.add_css_class("suggested-action")
        close_button.connect("clicked", lambda b: dialog.destroy())
        button_box.append(close_button)

        # Add components to the content box
        content_box.append(scrolled)
        content_box.append(
            status_box
        )  # Add the status box that contains both status and buttons

        # Add the button box to the status box
        status_box.append(button_box)

        # Add the content box to the dialog
        dialog.set_child(content_box)

        # Add keyboard shortcut controller for Ctrl+F to focus search
        key_controller = Gtk.EventControllerKey()
        dialog.add_controller(key_controller)

        def on_key_pressed(controller, keyval, keycode, state):
            # Check for Ctrl+F
            if keyval == Gdk.KEY_f and state & Gdk.ModifierType.CONTROL_MASK:
                search_entry.grab_focus()
                return True
            return False

        key_controller.connect("key-pressed", on_key_pressed)

        # Show the dialog
        dialog.present()

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy the text to the clipboard

        Args:
            text: The text to copy to clipboard
        """
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        logger.info("Text copied to clipboard")

    def _save_text_to_file(self, text: str, source_pdf_path: str) -> None:
        """Save the text to a file directly

        Args:
            text: The text to save
            source_pdf_path: Path to the source PDF file (used to derive filename)
        """
        # Generate a simple suggested filename
        base_name = os.path.splitext(os.path.basename(source_pdf_path))[0]
        suggested_filename = f"{base_name}-text.txt"

        # Show the save dialog directly
        self._show_file_save_dialog(text, source_pdf_path, suggested_filename)

    # We've removed the _show_suffix_options_dialog method to simplify the interface
    # Now we go directly to the file save dialog

    def _show_file_save_dialog(
        self, text: str, source_pdf_path: str, suggested_filename: str
    ) -> None:
        """Show the file save dialog with the suggested filename

        Args:
            text: The text to save
            source_pdf_path: Path to the source PDF file
            suggested_filename: The suggested filename with applied formatting
        """
        # Create a file dialog for saving
        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Save Extracted Text"))
        save_dialog.set_modal(True)

        # Set suggested filename
        name_model = Gio.FileInfo()
        name_model.set_attribute_string("standard::display-name", suggested_filename)
        save_dialog.set_initial_name(suggested_filename)

        # Use the same folder as the output PDF by default
        initial_folder = os.path.dirname(source_pdf_path)
        if os.path.exists(initial_folder):
            save_dialog.set_initial_folder(Gio.File.new_for_path(initial_folder))

        # Show the save dialog
        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_save_dialog_response(d, r, text),
        )

    def _on_save_dialog_response(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, text: str
    ) -> None:
        """Handle the response from the save file dialog

        Args:
            dialog: The file dialog
            result: The async result
            text: The text to save
        """
        try:
            # Get the file from the dialog
            file = dialog.save_finish(result)
            file_path = file.get_path()

            # Check if file already exists
            if os.path.exists(file_path):
                # Show confirmation dialog
                self._show_file_exists_dialog(file_path, text)
                return

            # File doesn't exist, save directly
            self._write_text_to_file(file_path, text)

        except Exception as e:
            logger.error(f"Error saving text to file: {e}")
            # If it's a user cancellation, don't show error
            if "Dismissed" not in str(e):
                # Show an error dialog
                error_dialog = Gtk.AlertDialog()
                error_dialog.set_modal(True)
                error_dialog.set_message(_("Save Failed"))
                error_dialog.set_detail(
                    _("Failed to save the text file: {0}").format(e)
                )
                error_dialog.show(self.window)

    def _show_file_exists_dialog(self, file_path: str, text: str) -> None:
        """Show a dialog to handle existing files

        Args:
            file_path: Path to the file that already exists
            text: Text content to save
        """
        # Create dialog
        dialog = Adw.MessageDialog(
            transient_for=self.window,
            heading=_("File Already Exists"),
            body=_("The file '{0}' already exists. What would you like to do?").format(
                os.path.basename(file_path)
            ),
        )

        # Add buttons for different actions
        dialog.add_response("overwrite", _("Overwrite"))
        dialog.add_response("rename", _("Auto-Rename"))
        dialog.add_response("cancel", _("Cancel"))

        # Set appearance of buttons
        dialog.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_response_appearance("cancel", Adw.ResponseAppearance.DEFAULT)

        # Connect response signal
        dialog.connect("response", self._on_file_exists_response, file_path, text)

        # Show the dialog
        dialog.present()

    def _on_file_exists_response(
        self, dialog: Adw.MessageDialog, response: str, file_path: str, text: str
    ) -> None:
        """Handle response from file exists dialog

        Args:
            dialog: The dialog that triggered the response
            response: Response ID ("overwrite", "rename", or "cancel")
            file_path: Path to the file that already exists
            text: Text content to save
        """
        if response == "overwrite":
            # Overwrite the existing file
            self._write_text_to_file(file_path, text)
        elif response == "rename":
            # Auto-generate unique filename
            new_path = self._generate_unique_filename(file_path)
            self._write_text_to_file(new_path, text)
        # Cancel - do nothing

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate a unique filename by appending a number

        Args:
            file_path: Original file path

        Returns:
            A unique file path that doesn't exist on the filesystem
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        # Keep incrementing counter until we find a filename that doesn't exist
        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{name}-{counter}{ext}")
            counter += 1

        return new_path

    def _write_text_to_file(self, file_path: str, text: str) -> None:
        """Write text content to a file

        Args:
            file_path: Path where to save the file
            text: Text content to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Save the text to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"Text saved to {file_path}")

            # Show success toast
            toast = Adw.Toast.new(f"Text saved to {os.path.basename(file_path)}")
            toast.set_timeout(3)

            # Find a toast overlay to show the toast
            if hasattr(self.window, "toast_overlay") and self.window.toast_overlay:
                self.window.toast_overlay.add_toast(toast)

        except Exception as e:
            logger.error(f"Error writing text to file: {e}")
            error_dialog = Gtk.AlertDialog()
            error_dialog.set_modal(True)
            error_dialog.set_message(_("Save Failed"))
            error_dialog.set_detail(_("Failed to save the text file: {0}").format(e))
            error_dialog.show(self.window)

    def show_pdf_options_dialog(self, callback) -> None:
        """Show dialog with PDF output options before starting OCR

        Args:
            callback: Function to call with options when dialog is confirmed
        """
        # Create the options dialog
        dialog = Adw.Window()
        dialog.set_title(_("PDF Output Options"))
        dialog.set_default_size(400, 380)  # Reduced height for smaller screens
        dialog.set_modal(True)
        dialog.set_transient_for(self.window)

        # Set up the Adwaita toolbar view structure (main container)
        toolbar_view = Adw.ToolbarView()

        # Create a header bar
        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        header_bar.set_title_widget(Gtk.Label(label=_("PDF Output Settings")))

        # Add cancel button to header
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda b: dialog.destroy())
        header_bar.pack_start(cancel_button)

        # Add save button to header
        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        header_bar.pack_end(save_button)

        # Add the header to the toolbar view
        toolbar_view.add_top_bar(header_bar)

        # Create scrolled window for content
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)  # Make sure it expands to fill the space

        # Create a preferences page
        prefs_page = Adw.PreferencesPage()  # File Settings group
        main_group = Adw.PreferencesGroup()
        main_group.set_title(_("File Settings"))

        # Use original filename switch row
        use_original_name_row = Adw.SwitchRow()
        use_original_name_row.set_title(_("Use Original Filename"))
        use_original_name_row.set_subtitle(_("Use the same name as the original file"))

        # Check if the setting exists, if not, default to False
        use_orig_name = getattr(self.window.settings, "use_original_filename", False)
        use_original_name_row.set_active(use_orig_name)

        # Add a warning for users about potential file replacement
        warning_row = Adw.ActionRow()
        warning_row.set_title(_("Warning"))
        warning_row.set_subtitle(
            _(
                "If saving to the same folder as original, this will replace the original files"
            )
        )
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_icon.set_pixel_size(16)
        warning_row.add_prefix(warning_icon)
        warning_row.add_css_class("warning-row")
        warning_row.set_visible(use_orig_name)  # Only show when option is active

        main_group.add(use_original_name_row)
        main_group.add(warning_row)

        # Suffix entry row - compact with proper Adwaita style
        suffix_row = Adw.EntryRow()
        suffix_row.set_title(_("Filename Suffix"))
        # EntryRow doesn't have set_subtitle method, so use tooltip instead
        suffix_row.set_tooltip_text(
            _("Text added to the filename (e.g. document-ocr.pdf)")
        )
        suffix_row.set_text(self.window.settings.pdf_suffix or "ocr")
        suffix_row.set_show_apply_button(False)
        suffix_row.set_sensitive(not use_orig_name)  # Disable if using original name
        main_group.add(suffix_row)

        # Overwrite files switch row
        overwrite_row = Adw.SwitchRow()
        overwrite_row.set_title(_("Overwrite Existing Files"))
        overwrite_row.set_subtitle(_("Replace files with the same name"))
        overwrite_row.set_active(self.window.settings.overwrite_existing)
        main_group.add(overwrite_row)

        # Text extraction group
        text_group = Adw.PreferencesGroup()
        text_group.set_title(_("Text Extraction"))

        # Save TXT files switch row
        save_txt_row = Adw.SwitchRow()
        save_txt_row.set_title(_("Save Text Files"))
        save_txt_row.set_subtitle(_("Automatically save extracted text as .txt files"))
        # Use the new setting, default to False if it doesn't exist
        save_txt = getattr(self.window.settings, "save_txt", False)
        save_txt_row.set_active(save_txt)
        text_group.add(save_txt_row)

        # Separate folder for TXT files switch row
        separate_folder_row = Adw.SwitchRow()
        separate_folder_row.set_title(_("Use Separate Folder for Text Files"))
        separate_folder_row.set_subtitle(_("Save text files to a different folder"))
        # Use the new setting, default to False if it doesn't exist
        use_separate_folder = getattr(
            self.window.settings, "separate_txt_folder", False
        )
        separate_folder_row.set_active(use_separate_folder)
        separate_folder_row.set_sensitive(save_txt)  # Only enable if saving text files
        text_group.add(separate_folder_row)

        # Text folder selection button row
        text_folder_row = Adw.ActionRow()
        text_folder_row.set_title(_("Text Files Folder"))
        text_folder_row.set_subtitle(_("Select where to save text files"))
        text_folder_row.set_sensitive(save_txt and use_separate_folder)

        # Get current text folder or empty string if not set
        txt_folder = getattr(self.window.settings, "txt_folder", "")
        folder_label = Gtk.Label(label=txt_folder or _("Not set"))
        folder_label.set_ellipsize(Pango.EllipsizeMode.START)
        folder_label.set_halign(Gtk.Align.END)
        folder_label.set_margin_end(8)  # Add some spacing for better appearance

        # Add button to select folder
        folder_button = Gtk.Button()
        folder_button.set_icon_name("folder-symbolic")
        folder_button.set_valign(Gtk.Align.CENTER)
        folder_button.add_css_class("flat")
        folder_button.set_tooltip_text(_("Select folder"))
        folder_button.set_sensitive(
            save_txt and use_separate_folder
        )  # Only enable if both switches are on

        text_folder_row.add_suffix(folder_label)
        text_folder_row.add_suffix(folder_button)
        text_group.add(text_folder_row)

        # Function to update text extraction UI sensitivity
        def update_text_options_sensitivity(*args):
            is_save_txt = save_txt_row.get_active()
            is_separate_folder = separate_folder_row.get_active()

            # Update UI elements based on switch states
            separate_folder_row.set_sensitive(is_save_txt)
            text_folder_row.set_sensitive(is_save_txt and is_separate_folder)
            folder_button.set_sensitive(is_save_txt and is_separate_folder)

        # Connect signals for text extraction options
        save_txt_row.connect("notify::active", update_text_options_sensitivity)
        separate_folder_row.connect("notify::active", update_text_options_sensitivity)

        # Function to handle folder selection
        def on_folder_selected(dialog, result):
            try:
                file = dialog.select_folder_finish(result)
                if file:
                    path = file.get_path()
                    folder_label.set_label(path)
            except Exception as e:
                logger.error(f"Error selecting folder: {e}")

        # Setup folder selection
        def select_text_folder(*args):
            dialog = Gtk.FileDialog.new()
            dialog.set_title(_("Select Folder for Text Files"))
            if txt_folder and os.path.exists(txt_folder):
                dialog.set_initial_folder(Gio.File.new_for_path(txt_folder))
            dialog.select_folder(
                parent=self.window, cancellable=None, callback=on_folder_selected
            )

        folder_button.connect("clicked", select_text_folder)

        # Add the text group to preferences page
        prefs_page.add(text_group)
        prefs_page.add(main_group)

        # Store references for later access
        suffix_entry = suffix_row

        # Store the switch for later access
        overwrite_check = overwrite_row

        # Date format options
        date_group = Adw.PreferencesGroup()
        date_group.set_title(_("Date and Time"))

        # Date switch row
        include_date_row = Adw.SwitchRow()
        include_date_row.set_title(_("Add Date to Filename"))
        include_date_row.set_subtitle(_("Include date elements in the filename"))
        include_date_row.set_active(self.window.settings.include_date)
        date_group.add(include_date_row)

        # Date format row with dropdown
        format_row = Adw.ComboRow()
        format_row.set_title(_("Date Format"))
        format_model = Gtk.StringList()
        format_model.append(_("YYYY-MM-DD (ISO)"))
        format_model.append(_("DD-MM-YYYY (Europe)"))
        format_model.append(_("MM-DD-YYYY (US)"))
        format_row.set_model(format_model)
        format_row.set_sensitive(include_date_row.get_active())

        # Set default format (try to use saved format if available)
        order = getattr(
            self.window.settings, "date_format_order", {"year": 1, "month": 2, "day": 3}
        )
        if order["day"] < order["month"] and order["month"] < order["year"]:
            # DD-MM-YYYY format
            format_row.set_selected(1)
        elif order["month"] < order["day"] and order["day"] < order["year"]:
            # MM-DD-YYYY format
            format_row.set_selected(2)
        else:
            # Default to ISO format YYYY-MM-DD
            format_row.set_selected(0)

        date_group.add(format_row)

        # Store the dropdown for later access
        date_format_dropdown = format_row

        # Year switch row
        year_row = Adw.SwitchRow()
        year_row.set_title(_("Include Year"))
        year_row.set_subtitle(_("Add YYYY to the date"))
        year_row.set_active(self.window.settings.include_year)
        year_row.set_sensitive(include_date_row.get_active())
        date_group.add(year_row)

        # Store switch for later access
        year_check = year_row

        # Month switch row
        month_row = Adw.SwitchRow()
        month_row.set_title(_("Include Month"))
        month_row.set_subtitle(_("Add MM to the date"))
        month_row.set_active(self.window.settings.include_month)
        month_row.set_sensitive(include_date_row.get_active())
        date_group.add(month_row)

        # Store switch for later access
        month_check = month_row

        # Day switch row
        day_row = Adw.SwitchRow()
        day_row.set_title(_("Include Day"))
        day_row.set_subtitle(_("Add DD to the date"))
        day_row.set_active(self.window.settings.include_day)
        day_row.set_sensitive(include_date_row.get_active())
        date_group.add(day_row)

        # Store switch for later access
        day_check = day_row

        # Time switch row
        time_row = Adw.SwitchRow()
        time_row.set_title(_("Include Time"))
        time_row.set_subtitle(_("Add HHMM to the filename"))
        time_row.set_active(self.window.settings.include_time)
        time_row.set_sensitive(include_date_row.get_active())
        date_group.add(time_row)

        # Store switch for later access
        time_check = time_row

        # Preview group
        preview_group = Adw.PreferencesGroup()
        preview_group.set_title(_("Preview"))

        # Create a preview row with custom widget
        preview_row = Adw.ActionRow()
        preview_row.set_title(_("Filename Example:"))

        # Sample file preview
        preview_value = Gtk.Label()
        preview_value.set_halign(Gtk.Align.END)
        preview_value.add_css_class("monospace")
        preview_value.add_css_class("caption")
        preview_value.add_css_class("dim-label")
        preview_row.add_suffix(preview_value)

        preview_group.add(preview_row)

        # Add groups to preferences page
        prefs_page.add(main_group)
        prefs_page.add(date_group)
        prefs_page.add(preview_group)

        # Add preferences page to scrolled window
        scrolled.set_child(prefs_page)

        # Add scrolled window to toolbar view content
        toolbar_view.set_content(scrolled)

        # Set the toolbar view as the dialog content
        dialog.set_content(toolbar_view)  # Function to update the preview

        def update_preview(*args):
            now = time.localtime()
            suffix = suffix_entry.get_text() or "ocr"
            use_original = use_original_name_row.get_active()

            # Format date parts if enabled
            if include_date_row.get_active():
                date_parts = []
                selected_format = date_format_dropdown.get_selected()

                # Get date components to include
                need_year = year_check.get_active()
                need_month = month_check.get_active()
                need_day = day_check.get_active()

                # Get format based on dropdown selection
                if selected_format == 0:  # YYYY-MM-DD
                    if need_year:
                        date_parts.append(f"{now.tm_year}")
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                elif selected_format == 1:  # DD-MM-YYYY
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_year:
                        date_parts.append(f"{now.tm_year}")
                else:  # MM-DD-YYYY
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                    if need_year:
                        date_parts.append(f"{now.tm_year}")

                # Always use hyphen separator for filename compatibility
                date_str = "-".join(date_parts)

                # Add time as a separate component if enabled
                if time_check.get_active():
                    time_str = f"{now.tm_hour:02d}{now.tm_min:02d}"
                    if date_parts:
                        date_str = f"{date_str}-{time_str}"
                    else:
                        date_str = time_str
            else:
                date_str = ""

            # Create sample filename based on settings
            if use_original:
                # Use example name to show it's using original name
                sample_name = "original_document"
                preview_text = f"{sample_name}.pdf"
            else:
                sample_name = "document"
                if date_str:
                    preview_text = f"{sample_name}-{suffix}-{date_str}.pdf"
                else:
                    preview_text = f"{sample_name}-{suffix}.pdf"

            preview_value.set_text(preview_text)

            # Update UI sensitivity
            suffix_entry.set_sensitive(not use_original)
            # Show/hide warning based on original filename setting
            warning_row.set_visible(use_original)

        # Function to enable/disable date options based on switch
        def update_date_options_sensitivity(*args):
            is_date_enabled = include_date_row.get_active()
            year_check.set_sensitive(is_date_enabled)
            month_check.set_sensitive(is_date_enabled)
            day_check.set_sensitive(is_date_enabled)
            time_check.set_sensitive(is_date_enabled)
            date_format_dropdown.set_sensitive(is_date_enabled)

            update_preview()

        # Connect signals for preview updates
        suffix_entry.connect("changed", update_preview)
        use_original_name_row.connect("notify::active", update_preview)
        include_date_row.connect("notify::active", update_date_options_sensitivity)
        year_check.connect("notify::active", update_preview)
        month_check.connect("notify::active", update_preview)
        day_check.connect("notify::active", update_preview)
        time_check.connect("notify::active", update_preview)
        date_format_dropdown.connect("notify::selected", update_preview)

        # Handle save button
        def on_save_button_clicked(button):
            # Save settings to the application
            self.window.settings.use_original_filename = (
                use_original_name_row.get_active()
            )
            self.window.settings.pdf_suffix = suffix_entry.get_text() or "ocr"
            self.window.settings.overwrite_existing = overwrite_check.get_active()
            self.window.settings.include_date = include_date_row.get_active()
            self.window.settings.include_year = year_check.get_active()
            self.window.settings.include_month = month_check.get_active()
            self.window.settings.include_day = day_check.get_active()
            self.window.settings.include_time = time_check.get_active()

            # Save text extraction settings
            self.window.settings.save_txt = save_txt_row.get_active()
            self.window.settings.separate_txt_folder = separate_folder_row.get_active()
            self.window.settings.txt_folder = (
                folder_label.get_label()
                if folder_label.get_label() != "Not set"
                else ""
            )

            # Save date format order preferences based on selected dropdown value
            selected_format = date_format_dropdown.get_selected()

            if not hasattr(self.window.settings, "date_format_order"):
                self.window.settings.date_format_order = {}

            # Set order based on dropdown selection
            if selected_format == 1:  # DD/MM/YYYY (Europe)
                self.window.settings.date_format_order = {
                    "day": 1,
                    "month": 2,
                    "year": 3,
                }
            elif selected_format == 2:  # MM/DD/YYYY (US)
                self.window.settings.date_format_order = {
                    "month": 1,
                    "day": 2,
                    "year": 3,
                }
            else:  # YYYY-MM-DD (ISO)
                self.window.settings.date_format_order = {
                    "year": 1,
                    "month": 2,
                    "day": 3,
                }

            # Close dialog and call callback
            dialog.destroy()
            callback(True)

        save_button.connect("clicked", on_save_button_clicked)

        # Initial preview update
        update_preview()

        # Show the dialog (toolbar_view is already set as content)
        dialog.present()
