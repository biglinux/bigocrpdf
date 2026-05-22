"""Conclusion Page UI Builder Mixin."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _


class ConclusionPageBuilderMixin:
    """Mixin providing conclusion page UI structure creation."""

    def create_conclusion_page(self) -> Gtk.Box:
        """Create the conclusion page showing OCR processing results

        Returns:
            A Gtk.Box containing the conclusion page UI
        """
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        # Set individual margins
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Add summary card
        summary_card = self._create_summary_card()
        main_box.append(summary_card)

        # Add output files section
        files_card = self._create_files_card()
        main_box.append(files_card)

        return main_box

    def _create_summary_card(self) -> Gtk.Box:
        """Create the processing summary card

        Returns:
            A Gtk.Box containing the summary card
        """
        summary_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        summary_card.add_css_class("card")

        # Card header
        card_header = Gtk.Label(label=_("Processing Summary"))
        card_header.add_css_class("heading")
        card_header.set_halign(Gtk.Align.START)
        card_header.set_margin_top(16)
        card_header.set_margin_start(16)
        summary_card.append(card_header)

        # Create statistics grid
        stats_grid = self._create_statistics_grid()
        summary_card.append(stats_grid)

        return summary_card

    def _create_statistics_grid(self) -> Gtk.Box:
        """Create the statistics grid with two columns

        Returns:
            A Gtk.Box containing the statistics
        """
        # Create a horizontal box to contain two columns
        columns_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=32)
        columns_box.set_margin_start(16)
        columns_box.set_margin_end(16)
        columns_box.set_margin_bottom(16)
        columns_box.set_margin_top(8)

        # Create left and right columns
        left_grid = self._create_left_statistics_column()
        right_grid = self._create_right_statistics_column()

        # Add the two grids to the columns box
        columns_box.append(left_grid)
        columns_box.append(right_grid)

        return columns_box

    def _create_left_statistics_column(self) -> Gtk.Grid:
        """Create the left statistics column (Files and Pages)

        Returns:
            A Gtk.Grid containing the left column statistics
        """
        left_grid = Gtk.Grid()
        left_grid.set_column_spacing(16)
        left_grid.set_row_spacing(16)
        left_grid.set_hexpand(True)
        left_grid.set_halign(Gtk.Align.START)

        # Files processed
        self._add_statistic_row(
            left_grid,
            0,
            "document-multiple-symbolic",
            _("Files processed:"),
            self._create_file_count_label(),
        )

        # Total pages
        self._add_statistic_row(
            left_grid,
            1,
            "view-paged-symbolic",
            _("Total pages:"),
            self._create_page_count_label(),
        )

        return left_grid

    def _create_right_statistics_column(self) -> Gtk.Grid:
        """Create the right statistics column (Time and Size)

        Returns:
            A Gtk.Grid containing the right column statistics
        """
        right_grid = Gtk.Grid()
        right_grid.set_column_spacing(16)
        right_grid.set_row_spacing(16)
        right_grid.set_hexpand(True)
        right_grid.set_halign(Gtk.Align.START)

        # Processing time
        self._add_statistic_row(
            right_grid,
            0,
            "clock-symbolic",
            _("Processing time:"),
            self._create_time_label(),
        )

        # Output file size
        self._add_statistic_row(
            right_grid,
            1,
            "drive-harddisk-symbolic",
            _("Output size:"),
            self._create_file_size_label(),
        )

        # Size change comparison (before -> after)
        self._add_statistic_row(
            right_grid,
            2,
            "emblem-synchronizing-symbolic",
            _("Size change:"),
            self._create_size_change_label(),
        )

        return right_grid

    def _add_statistic_row(
        self,
        grid: Gtk.Grid,
        row: int,
        icon_name: str,
        label_text: str,
        value_label: Gtk.Label,
    ) -> None:
        """Add a statistic row to a grid

        Args:
            grid: Grid to add the row to
            row: Row number
            icon_name: Name of the icon to display
            label_text: Text for the label
            value_label: Label widget for the value
        """
        # Icon
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        grid.attach(icon, 0, row, 1, 1)

        # Label
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_margin_start(8)
        grid.attach(label, 1, row, 1, 1)

        # Value
        value_label.set_halign(Gtk.Align.END)
        value_label.add_css_class("heading")
        value_label.set_hexpand(True)
        grid.attach(value_label, 2, row, 1, 1)

    def _create_file_count_label(self) -> Gtk.Label:
        """Create the file count label

        Returns:
            A Gtk.Label for displaying file count
        """
        self.result_file_count = Gtk.Label(label="0")
        set_a11y_label(self.result_file_count, _("Files processed"))
        return self.result_file_count

    def _create_page_count_label(self) -> Gtk.Label:
        """Create the page count label

        Returns:
            A Gtk.Label for displaying page count
        """
        self.result_page_count = Gtk.Label(label="0")
        set_a11y_label(self.result_page_count, _("Pages processed"))
        return self.result_page_count

    def _create_time_label(self) -> Gtk.Label:
        """Create the processing time label

        Returns:
            A Gtk.Label for displaying processing time
        """
        self.result_time = Gtk.Label(label="00:00")
        set_a11y_label(self.result_time, _("Processing time"))
        return self.result_time

    def _create_file_size_label(self) -> Gtk.Label:
        """Create the file size label

        Returns:
            A Gtk.Label for displaying file size
        """
        self.result_file_size = Gtk.Label(label="0 KB")
        set_a11y_label(self.result_file_size, _("Output file size"))
        return self.result_file_size

    def _create_size_change_label(self) -> Gtk.Label:
        """Create the size change label for before/after comparison.

        Returns:
            A Gtk.Label for displaying size change
        """
        self.result_size_change = Gtk.Label(label="--")
        set_a11y_label(self.result_size_change, _("File size change"))
        return self.result_size_change

    def _create_files_card(self) -> Gtk.Box:
        """Create the output files card

        Returns:
            A Gtk.Box containing the files card
        """
        files_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        files_card.add_css_class("card")
        files_card.set_margin_top(16)
        files_card.set_margin_bottom(16)

        # Header row: title on the left, selection toggle on the right.
        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header_row.set_margin_top(16)
        header_row.set_margin_start(16)
        header_row.set_margin_end(16)

        files_header = Gtk.Label(label=_("Generated Files"))
        files_header.add_css_class("heading")
        files_header.set_halign(Gtk.Align.START)
        files_header.set_hexpand(True)
        header_row.append(files_header)

        self._selection_toggle_btn = Gtk.ToggleButton()
        self._selection_toggle_btn.set_icon_name("object-select-symbolic")
        self._selection_toggle_btn.set_tooltip_text(_("Select files for bulk actions"))
        self._selection_toggle_btn.add_css_class("flat")
        self._selection_toggle_btn.connect("toggled", self._on_selection_toggle_clicked)
        header_row.append(self._selection_toggle_btn)

        files_card.append(header_row)

        # Create scrollable file list
        scrolled_list = self._create_scrollable_file_list()
        files_card.append(scrolled_list)

        # Bulk-action bar (hidden until selection mode is on).
        self._selection_action_bar = self._create_selection_action_bar()
        self._selection_action_bar.set_visible(False)
        files_card.append(self._selection_action_bar)

        return files_card

    def _create_selection_action_bar(self) -> Gtk.Box:
        """Build the bulk-action bar shown at the bottom of the files card."""
        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        bar.set_margin_start(16)
        bar.set_margin_end(16)
        bar.set_margin_bottom(16)

        self._selection_count_label = Gtk.Label(label=_("Selected: 0"))
        self._selection_count_label.set_halign(Gtk.Align.START)
        self._selection_count_label.set_hexpand(True)
        self._selection_count_label.add_css_class("dim-label")
        bar.append(self._selection_count_label)

        select_all_btn = Gtk.Button(label=_("Select all"))
        select_all_btn.add_css_class("flat")
        select_all_btn.connect("clicked", lambda _b: self._on_select_all_clicked())
        bar.append(select_all_btn)

        clear_btn = Gtk.Button(label=_("Clear"))
        clear_btn.add_css_class("flat")
        clear_btn.connect("clicked", lambda _b: self._on_clear_selection_clicked())
        bar.append(clear_btn)

        # The export menu builds itself when there's a selection to act on.
        self._bulk_export_button = self._create_bulk_export_menu_button()
        bar.append(self._bulk_export_button)

        return bar

    def _create_scrollable_file_list(self) -> Gtk.ScrolledWindow:
        """Create the scrollable file list

        Returns:
            A Gtk.ScrolledWindow containing the file list
        """
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
        self.output_list_box.add_css_class("output-files-list")
        self.output_list_box.set_vexpand(True)
        set_a11y_label(self.output_list_box, _("Output files"))

        scrolled.set_child(self.output_list_box)
        return scrolled
