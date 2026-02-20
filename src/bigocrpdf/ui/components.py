"""
BigOcrPdf - UI Components Factory

Factory pattern implementation for creating consistent UI widgets.
This module provides standardized widget creation to ensure UI consistency
and reduce code duplication across the application.

Based on ashyterm FormWidgetBuilder pattern and BigLinux ARCHITECTURAL RULES.
"""

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.a11y import set_a11y_label


class UIComponents:
    """Factory class for creating consistent UI widgets.

    This class implements the Factory Pattern for UI widget creation,
    ensuring consistency across the application and reducing code duplication.

    Usage:
        from bigocrpdf.ui.components import create_action_button, create_switch_row

        button = create_action_button("Start", on_click=my_handler)
        switch = create_switch_row("Enable feature", active=True)
    """

    @staticmethod
    def create_action_button(
        label: str,
        icon_name: str | None = None,
        tooltip: str | None = None,
        on_click: Callable | None = None,
        css_classes: list[str] | None = None,
        halign: Gtk.Align = Gtk.Align.CENTER,
    ) -> Gtk.Button:
        """Create a styled action button.

        Args:
            label: Button text
            icon_name: Optional icon name (e.g., "document-open-symbolic")
            tooltip: Optional tooltip text
            on_click: Optional click handler callback
            css_classes: Optional list of CSS classes to apply
            halign: Horizontal alignment (default: CENTER)

        Returns:
            Configured Gtk.Button instance
        """
        button = Gtk.Button()
        button.set_label(label)
        button.set_halign(halign)

        if icon_name:
            button.set_icon_name(icon_name)

        if tooltip:
            button.set_tooltip_text(tooltip)

        if on_click:
            button.connect("clicked", lambda _: on_click())

        if css_classes:
            for css_class in css_classes:
                button.add_css_class(css_class)

        set_a11y_label(button, tooltip or label)

        return button

    @staticmethod
    def create_navigation_button(
        label: str,
        direction: str = "next",
        on_click: Callable | None = None,
        sensitive: bool = True,
        visible: bool = True,
    ) -> Gtk.Button:
        """Create a navigation button (next/back/start).

        Args:
            label: Button text
            direction: "next", "back", or "start" - determines styling
            on_click: Click handler callback
            sensitive: Whether button is clickable
            visible: Whether button is visible

        Returns:
            Configured navigation button
        """
        button = Gtk.Button()
        button.set_label(label)
        button.set_sensitive(sensitive)
        button.set_visible(visible)

        if direction == "next" or direction == "start":
            button.add_css_class("suggested-action")
        elif direction == "back":
            button.set_icon_name("go-previous-symbolic")
        elif direction == "destructive":
            button.add_css_class("destructive-action")

        if on_click:
            button.connect("clicked", lambda _: on_click())

        set_a11y_label(button, label)

        return button

    @staticmethod
    def create_icon_button(
        icon_name: str,
        tooltip: str | None = None,
        on_click: Callable | None = None,
        css_classes: list[str] | None = None,
        circular: bool = True,
        flat: bool = True,
    ) -> Gtk.Button:
        """Create an icon-only button.

        Args:
            icon_name: Icon name (e.g., "user-trash-symbolic")
            tooltip: Tooltip text
            on_click: Click handler callback
            css_classes: Additional CSS classes
            circular: Whether to use circular style
            flat: Whether to use flat style

        Returns:
            Configured icon button
        """
        button = Gtk.Button()
        button.set_icon_name(icon_name)
        button.set_valign(Gtk.Align.CENTER)

        if tooltip:
            button.set_tooltip_text(tooltip)
            set_a11y_label(button, tooltip)

        if circular:
            button.add_css_class("circular")

        if flat:
            button.add_css_class("flat")

        if css_classes:
            for css_class in css_classes:
                button.add_css_class(css_class)

        if on_click:
            button.connect("clicked", lambda _: on_click())

        return button

    @staticmethod
    def create_action_row(
        title: str,
        subtitle: str | None = None,
        icon_name: str | None = None,
        suffix_widget: Gtk.Widget | None = None,
        prefix_widget: Gtk.Widget | None = None,
        css_class: str = "action-row-config",
        activatable: bool = True,
    ) -> Adw.ActionRow:
        """Create a styled Adw.ActionRow.

        Args:
            title: Row title
            subtitle: Optional subtitle text
            icon_name: Optional prefix icon name
            suffix_widget: Optional widget to add at the end
            prefix_widget: Optional widget to add at the start (overrides icon)
            css_class: CSS class to apply
            activatable: Whether row responds to clicks

        Returns:
            Configured Adw.ActionRow instance
        """
        row = Adw.ActionRow(title=title)
        row.add_css_class(css_class)
        row.set_activatable(activatable)

        if subtitle:
            row.set_subtitle(subtitle)

        if prefix_widget:
            row.add_prefix(prefix_widget)
        elif icon_name:
            icon = Gtk.Image.new_from_icon_name(icon_name)
            row.add_prefix(icon)

        if suffix_widget:
            row.add_suffix(suffix_widget)
            if activatable:
                row.set_activatable_widget(suffix_widget)

        return row

    @staticmethod
    def create_switch_row(
        title: str,
        subtitle: str | None = None,
        active: bool = False,
        icon_name: str | None = None,
        on_toggle: Callable[[bool], None] | None = None,
        css_class: str = "action-row-config",
    ) -> Adw.SwitchRow:
        """Create a styled Adw.SwitchRow.

        Args:
            title: Row title
            subtitle: Optional subtitle text
            active: Initial switch state
            icon_name: Optional prefix icon name
            on_toggle: Callback when toggled (receives new state as bool)
            css_class: CSS class to apply

        Returns:
            Configured Adw.SwitchRow instance
        """
        row = Adw.SwitchRow(title=title)
        row.set_active(active)
        row.add_css_class(css_class)

        if subtitle:
            row.set_subtitle(subtitle)

        if icon_name:
            icon = Gtk.Image.new_from_icon_name(icon_name)
            row.add_prefix(icon)

        if on_toggle:
            row.connect("notify::active", lambda r, _: on_toggle(r.get_active()))

        return row

    @staticmethod
    def create_dropdown(
        items: list[tuple[str, str]],
        selected_value: str,
        on_change: Callable[[str], None] | None = None,
        tooltip: str | None = None,
    ) -> Gtk.DropDown:
        """Create a configured dropdown.

        Args:
            items: List of (value, display_text) tuples
            selected_value: Value to select initially
            on_change: Callback when selection changes (receives selected value)
            tooltip: Optional tooltip text

        Returns:
            Configured Gtk.DropDown instance
        """
        dropdown = Gtk.DropDown()
        dropdown.set_can_focus(False)  # Prevent focus during construction
        string_list = Gtk.StringList()

        selected_index = 0
        for i, (value, display_text) in enumerate(items):
            string_list.append(display_text)
            if value == selected_value:
                selected_index = i

        dropdown.set_model(string_list)
        dropdown.set_valign(Gtk.Align.CENTER)

        if tooltip:
            dropdown.set_tooltip_text(tooltip)
            dropdown.update_property([Gtk.AccessibleProperty.LABEL], [tooltip])

        if on_change:

            def handle_change(dd, _param):
                index = dd.get_selected()
                if 0 <= index < len(items):
                    on_change(items[index][0])

            dropdown.connect("notify::selected", handle_change)

        # Defer set_selected and re-enable focus after widget is mapped
        def on_map(_widget):
            dropdown.set_can_focus(True)
            if selected_index != 0:
                dropdown.set_selected(selected_index)

        dropdown.connect("map", on_map)

        return dropdown

    @staticmethod
    def create_card(
        orientation: Gtk.Orientation = Gtk.Orientation.VERTICAL,
        spacing: int = 8,
        margins: tuple[int, int, int, int] | None = None,
        vexpand: bool = False,
        hexpand: bool = False,
    ) -> Gtk.Box:
        """Create a styled card container.

        Args:
            orientation: Box orientation (VERTICAL or HORIZONTAL)
            spacing: Spacing between children
            margins: (top, end, bottom, start) margins tuple
            vexpand: Whether to expand vertically
            hexpand: Whether to expand horizontally

        Returns:
            Styled Gtk.Box with card class
        """
        card = Gtk.Box(orientation=orientation, spacing=spacing)
        card.add_css_class("card")
        card.set_vexpand(vexpand)
        card.set_hexpand(hexpand)

        if margins:
            top, end, bottom, start = margins
            card.set_margin_top(top)
            card.set_margin_end(end)
            card.set_margin_bottom(bottom)
            card.set_margin_start(start)

        return card

    @staticmethod
    def create_preferences_group(
        title: str,
        description: str | None = None,
        margin_top: int = 0,
        margin_bottom: int = 0,
    ) -> Adw.PreferencesGroup:
        """Create a preferences group.

        Args:
            title: Group title
            description: Optional description text
            margin_top: Top margin
            margin_bottom: Bottom margin

        Returns:
            Configured Adw.PreferencesGroup instance
        """
        group = Adw.PreferencesGroup(title=title)

        if description:
            group.set_description(description)

        if margin_top > 0:
            group.set_margin_top(margin_top)

        if margin_bottom > 0:
            group.set_margin_bottom(margin_bottom)

        return group

    @staticmethod
    def create_entry_with_icon(
        placeholder: str | None = None,
        text: str = "",
        icon_name: str | None = None,
        editable: bool = True,
        hexpand: bool = True,
    ) -> Gtk.Entry:
        """Create an entry field with optional icon.

        Args:
            placeholder: Placeholder text
            text: Initial text value
            icon_name: Icon to show at start of entry
            editable: Whether entry is editable
            hexpand: Whether to expand horizontally

        Returns:
            Configured Gtk.Entry instance
        """
        entry = Gtk.Entry()
        entry.set_text(text)
        entry.set_hexpand(hexpand)
        entry.set_editable(editable)

        if placeholder:
            entry.set_placeholder_text(placeholder)

        if icon_name:
            icon = Gio.ThemedIcon.new(icon_name)
            entry.set_icon_from_gicon(Gtk.EntryIconPosition.PRIMARY, icon)

        return entry

    @staticmethod
    def create_labeled_widget(
        label_text: str,
        widget: Gtk.Widget,
        orientation: Gtk.Orientation = Gtk.Orientation.HORIZONTAL,
        spacing: int = 8,
        label_halign: Gtk.Align = Gtk.Align.START,
    ) -> Gtk.Box:
        """Create a box with a label and widget.

        Args:
            label_text: Text for the label
            widget: Widget to place next to label
            orientation: Box orientation
            spacing: Spacing between label and widget
            label_halign: Horizontal alignment of label

        Returns:
            Gtk.Box containing label and widget
        """
        box = Gtk.Box(orientation=orientation, spacing=spacing)

        label = Gtk.Label(label=label_text)
        label.set_halign(label_halign)

        box.append(label)
        box.append(widget)

        return box

    @staticmethod
    def create_header_label(
        text: str,
        css_class: str = "heading",
        halign: Gtk.Align = Gtk.Align.START,
        margin_top: int = 12,
        margin_start: int = 12,
    ) -> Gtk.Label:
        """Create a header/title label.

        Args:
            text: Label text
            css_class: CSS class (default: "heading")
            halign: Horizontal alignment
            margin_top: Top margin
            margin_start: Start margin

        Returns:
            Configured header Gtk.Label
        """
        label = Gtk.Label(label=text)
        label.add_css_class(css_class)
        label.set_halign(halign)
        label.set_margin_top(margin_top)
        label.set_margin_start(margin_start)

        return label

    @staticmethod
    def create_scrolled_window(
        child: Gtk.Widget | None = None,
        min_height: int = 100,
        max_height: int = 400,
        hscroll_policy: Gtk.PolicyType = Gtk.PolicyType.NEVER,
        vscroll_policy: Gtk.PolicyType = Gtk.PolicyType.AUTOMATIC,
        vexpand: bool = True,
    ) -> Gtk.ScrolledWindow:
        """Create a configured scrolled window.

        Args:
            child: Child widget to contain
            min_height: Minimum content height
            max_height: Maximum content height
            hscroll_policy: Horizontal scroll policy
            vscroll_policy: Vertical scroll policy
            vexpand: Whether to expand vertically

        Returns:
            Configured Gtk.ScrolledWindow
        """
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(hscroll_policy, vscroll_policy)
        scrolled.set_min_content_height(min_height)
        scrolled.set_max_content_height(max_height)
        scrolled.set_vexpand(vexpand)

        if child:
            scrolled.set_child(child)

        return scrolled

    @staticmethod
    def create_progress_bar(
        show_text: bool = True,
        text: str = "0%",
        fraction: float = 0.0,
        margin_bottom: int = 8,
    ) -> Gtk.ProgressBar:
        """Create a configured progress bar.

        Args:
            show_text: Whether to show percentage text
            text: Initial text
            fraction: Initial fraction (0.0-1.0)
            margin_bottom: Bottom margin

        Returns:
            Configured Gtk.ProgressBar
        """
        progress = Gtk.ProgressBar()
        progress.set_show_text(show_text)
        progress.set_text(text)
        progress.set_fraction(fraction)
        progress.set_margin_bottom(margin_bottom)

        return progress

    @staticmethod
    def create_step_indicator(
        steps: list[str],
        current_step: int = 0,
        completed_steps: list[int] | None = None,
    ) -> Gtk.Box:
        """Create a visual step indicator (breadcrumbs) showing process steps.

        Args:
            steps: List of step names
            current_step: Index of the current active step (0-based)
            completed_steps: List of completed step indices

        Returns:
            Gtk.Box containing the step indicator
        """
        if completed_steps is None:
            completed_steps = list(range(current_step))

        container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        container.set_halign(Gtk.Align.CENTER)

        for i, step_name in enumerate(steps):
            # Create step circle/indicator
            step_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            step_box.set_valign(Gtk.Align.CENTER)

            # Step number/check indicator
            indicator = Gtk.Label()

            if i in completed_steps:
                # Completed - show checkmark
                indicator.set_markup("✓")
                indicator.add_css_class("step-completed")
            else:
                # Show step number
                indicator.set_label(str(i + 1))

            indicator.add_css_class("step-indicator")
            if i == current_step:
                indicator.add_css_class("step-current")
            elif i in completed_steps:
                indicator.add_css_class("step-done")
            else:
                indicator.add_css_class("step-pending")

            step_box.append(indicator)

            # Step label
            label = Gtk.Label(label=step_name)
            label.add_css_class("step-label")
            if i == current_step:
                label.add_css_class("step-label-current")
            elif i not in completed_steps:
                label.add_css_class("dim-label")

            step_box.append(label)
            container.append(step_box)

            # Add separator (arrow) between steps
            if i < len(steps) - 1:
                separator = Gtk.Label(label="›")
                separator.add_css_class("step-separator")
                separator.add_css_class("dim-label")
                separator.set_margin_start(8)
                separator.set_margin_end(8)
                container.append(separator)

        return container


# =============================================================================
# Convenience aliases for common use cases
# These allow direct import: from bigocrpdf.ui.components import create_action_button
# =============================================================================

create_action_button = UIComponents.create_action_button
create_navigation_button = UIComponents.create_navigation_button
create_icon_button = UIComponents.create_icon_button
create_action_row = UIComponents.create_action_row
create_switch_row = UIComponents.create_switch_row
create_dropdown = UIComponents.create_dropdown
create_card = UIComponents.create_card
create_preferences_group = UIComponents.create_preferences_group
create_entry_with_icon = UIComponents.create_entry_with_icon
create_labeled_widget = UIComponents.create_labeled_widget
create_header_label = UIComponents.create_header_label
create_scrolled_window = UIComponents.create_scrolled_window
create_progress_bar = UIComponents.create_progress_bar
create_step_indicator = UIComponents.create_step_indicator
