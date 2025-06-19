"""
BigOcrPdf - First Window (Settings Page)

This module contains the settings page implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, Gdk

import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _


class FirstWindow:
    """Settings page for BigOcrPdf application"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize settings page

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI components references
        self.lang_dropdown = None
        self.quality_dropdown = None
        self.alignment_dropdown = None
        self.dest_entry = None
        self.file_list_box = None
        self.remove_button = None
        self.drop_label = None
        self.same_folder_switch_row = None
        self.dest_container = None
        self.status_label = None

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
        settings_card.add_css_class("card")

        # Add a header for destination
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
            "clicked",
            lambda b: self.window.settings_dialog.show_pdf_options_dialog(
                lambda _: None
            ),
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
        align_row.add_css_class("action-row-config")
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
        # add css class for button important
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
            ("alignrotate", _("Align + rotate")),
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
