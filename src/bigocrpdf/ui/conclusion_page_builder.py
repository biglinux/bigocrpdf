"""Conclusion page layout."""
# Host attributes are supplied by the conclusion page's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.adw_compat import create_wrap_box
from bigocrpdf.utils.i18n import _


class ConclusionPageBuilderMixin:
    """Build the conclusion page widgets."""

    def create_conclusion_page(self) -> Gtk.Box:
        """Create the page that presents processing results and output files."""
        main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=16,
            margin_start=16,
            margin_end=16,
        )

        summary_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        summary_card.add_css_class("card")
        summary_card.append(
            Gtk.Label(
                label=_("Processing Summary"),
                css_classes=["heading"],
                halign=Gtk.Align.START,
                margin_top=16,
                margin_start=16,
            )
        )

        columns_box = create_wrap_box(
            child_spacing=32,
            line_spacing=16,
            margin_start=16,
            margin_end=16,
            margin_top=8,
            margin_bottom=16,
        )
        left_grid = Gtk.Grid(
            column_spacing=16,
            row_spacing=16,
            hexpand=True,
            halign=Gtk.Align.START,
        )
        right_grid = Gtk.Grid(
            column_spacing=16,
            row_spacing=16,
            hexpand=True,
            halign=Gtk.Align.START,
        )

        self.result_file_count = Gtk.Label(label="0")
        set_a11y_label(self.result_file_count, _("Files processed"))
        self.result_page_count = Gtk.Label(label="0")
        set_a11y_label(self.result_page_count, _("Pages processed"))
        self.result_time = Gtk.Label(label="00:00")
        set_a11y_label(self.result_time, _("Processing time"))
        self.result_file_size = Gtk.Label(label="0 KB")
        set_a11y_label(self.result_file_size, _("Output file size"))
        self.result_size_change = Gtk.Label(label="--")
        set_a11y_label(self.result_size_change, _("File size change"))

        for grid, row, icon, title, value in (
            (
                left_grid,
                0,
                "document-multiple-symbolic",
                _("Files processed:"),
                self.result_file_count,
            ),
            (left_grid, 1, "view-paged-symbolic", _("Total pages:"), self.result_page_count),
            (right_grid, 0, "clock-symbolic", _("Processing time:"), self.result_time),
            (
                right_grid,
                1,
                "drive-harddisk-symbolic",
                _("Output size:"),
                self.result_file_size,
            ),
            (
                right_grid,
                2,
                "emblem-synchronizing-symbolic",
                _("Size change:"),
                self.result_size_change,
            ),
        ):
            self._add_statistic_row(grid, row, icon, title, value)

        columns_box.append(left_grid)
        columns_box.append(right_grid)
        summary_card.append(columns_box)
        main_box.append(summary_card)

        files_card = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            margin_top=16,
            margin_bottom=16,
        )
        files_card.add_css_class("card")

        header_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=8,
            margin_top=16,
            margin_start=16,
            margin_end=16,
        )
        header_row.append(
            Gtk.Label(
                label=_("Generated Files"),
                css_classes=["heading"],
                halign=Gtk.Align.START,
                hexpand=True,
            )
        )
        self._selection_toggle_btn = Gtk.ToggleButton(
            icon_name="object-select-symbolic",
            tooltip_text=_("Select files for bulk actions"),
            css_classes=["flat"],
        )
        self._selection_toggle_btn.connect("toggled", self._on_selection_toggle_clicked)
        header_row.append(self._selection_toggle_btn)
        files_card.append(header_row)

        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=100,
            max_content_height=200,
            margin_start=16,
            margin_end=16,
            margin_bottom=16,
        )
        self.output_list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list", "output-files-list"],
            vexpand=True,
        )
        set_a11y_label(self.output_list_box, _("Output files"))
        scrolled.set_child(self.output_list_box)
        files_card.append(scrolled)

        self._selection_action_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=8,
            margin_start=16,
            margin_end=16,
            margin_bottom=16,
            visible=False,
        )
        self._selection_count_label = Gtk.Label(
            label=_("Selected: 0"),
            halign=Gtk.Align.START,
            hexpand=True,
            css_classes=["dim-label"],
        )
        self._selection_action_bar.append(self._selection_count_label)

        select_all_button = Gtk.Button(label=_("Select all"), css_classes=["flat"])
        select_all_button.connect("clicked", lambda _button: self._on_select_all_clicked())
        self._selection_action_bar.append(select_all_button)

        clear_button = Gtk.Button(label=_("Clear"), css_classes=["flat"])
        clear_button.connect("clicked", lambda _button: self._on_clear_selection_clicked())
        self._selection_action_bar.append(clear_button)

        self._bulk_export_button = self._create_bulk_export_menu_button()
        self._selection_action_bar.append(self._bulk_export_button)
        files_card.append(self._selection_action_bar)
        main_box.append(files_card)
        return main_box

    @staticmethod
    def _add_statistic_row(
        grid: Gtk.Grid,
        row: int,
        icon_name: str,
        label_text: str,
        value_label: Gtk.Label,
    ) -> None:
        """Add one icon, description, and value to a statistics grid."""
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        grid.attach(icon, 0, row, 1, 1)

        grid.attach(
            Gtk.Label(label=label_text, halign=Gtk.Align.START, margin_start=8),
            1,
            row,
            1,
            1,
        )

        value_label.set_halign(Gtk.Align.END)
        value_label.set_hexpand(True)
        value_label.add_css_class("heading")
        grid.attach(value_label, 2, row, 1, 1)
