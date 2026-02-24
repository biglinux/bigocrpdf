"""Window UI Setup Mixin."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.config import APP_ICON_NAME
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class WindowUISetupMixin:
    """Mixin providing UI setup and layout creation for the main window."""

    # Type stubs for attributes initialized by the concrete window class
    custom_header_bar: HeaderBar

    def setup_ui(self) -> None:
        """Set up the main user interface with responsive split view layout."""
        self.toast_overlay = Adw.ToastOverlay()

        # Hidden live-region label for screen reader announcements.
        # AccessibleRole.STATUS means Orca reads updates automatically.
        self._a11y_status = Gtk.Label(accessible_role=Gtk.AccessibleRole.STATUS)
        self._a11y_status.set_visible(False)

        # Create responsive split view layout (sidebar + content)
        self.split_view = Adw.OverlaySplitView()
        self.split_view.set_sidebar_position(Gtk.PackType.START)
        self.split_view.set_min_sidebar_width(300)
        self.split_view.set_max_sidebar_width(420)
        self.split_view.set_sidebar_width_fraction(0.35)
        self.split_view.set_enable_hide_gesture(True)
        self.split_view.set_enable_show_gesture(True)

        # Create left sidebar pane
        self._create_left_sidebar()

        # Create right content pane with header bar
        self._create_right_content_area()

        # Create master ViewStack for main view and other pages
        self.main_stack = Adw.ViewStack()

        # Add split_view as primary view
        self.main_stack.add_titled(self.split_view, "main_view", _("Main"))

        # Set up window-level actions for keyboard shortcuts
        self._setup_window_actions()

        # Set up global drag and drop
        self._setup_global_drag_drop()

        # Set up responsive breakpoint — collapse sidebar when window is narrow
        breakpoint = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 700sp"))
        breakpoint.add_setter(self.split_view, "collapsed", True)
        self.add_breakpoint(breakpoint)

        self.main_stack.connect("notify::visible-child", self._on_main_stack_changed)

        # Wrap main_stack and the hidden a11y status label together
        content_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_wrapper.append(self._a11y_status)
        content_wrapper.append(self.main_stack)
        self.main_stack.set_vexpand(True)

        self.toast_overlay.set_child(content_wrapper)
        self.set_content(self.toast_overlay)

        # Set up pages (must be after UI structure is created)
        self.setup_pages()

        # Initialize header bar view for settings page
        self.custom_header_bar.set_view("queue")

        # Bind sidebar toggle to split view for responsive collapse
        self.custom_header_bar.bind_split_view(self.split_view)

    def _create_left_sidebar(self) -> None:
        """Create the left sidebar with header bar (video-converter style)."""
        # Create ToolbarView for left pane with sidebar styling
        left_toolbar_view = Adw.ToolbarView()
        left_toolbar_view.add_css_class("sidebar")

        # Detect window button layout
        window_buttons_left = self._window_buttons_on_left()

        # Create HeaderBar for the sidebar
        left_header = Adw.HeaderBar()
        left_header.add_css_class("sidebar")
        left_header.set_show_title(True)
        # Configure decoration layout
        left_header.set_decoration_layout(
            "close,maximize,minimize:menu" if window_buttons_left else ""
        )

        # Create title with app icon
        if not window_buttons_left:
            center_box = Gtk.CenterBox()
            center_box.set_hexpand(True)
            app_icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
            app_icon.set_pixel_size(20)
            app_icon.set_halign(Gtk.Align.START)
            app_icon.set_valign(Gtk.Align.CENTER)
            app_icon.set_hexpand(False)
            center_box.set_start_widget(app_icon)
            title_label = Gtk.Label(label="Big OCR PDF")
            title_label.set_halign(Gtk.Align.CENTER)
            title_label.set_valign(Gtk.Align.CENTER)
            title_label.set_hexpand(True)
            center_box.set_center_widget(title_label)
            left_header.set_title_widget(center_box)
        else:
            title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            title_label = Gtk.Label(label="Big OCR PDF")
            title_box.append(title_label)
            expander = Gtk.Box()
            expander.set_hexpand(True)
            title_box.append(expander)
            left_header.set_title_widget(title_box)

        left_toolbar_view.add_top_bar(left_header)

        # Create sidebar content (OCR settings)
        sidebar_content = self._create_sidebar_settings()
        left_toolbar_view.set_content(sidebar_content)

        self.split_view.set_sidebar(left_toolbar_view)
        self.left_toolbar_view = left_toolbar_view  # Store reference for later settings

    def _create_sidebar_settings(self) -> Gtk.Widget:
        """Create a placeholder for sidebar settings.

        The actual content is set by _populate_sidebar_settings().

        Returns:
            Widget placeholder for OCR settings
        """
        # Just a placeholder box - the real content is set later
        self.sidebar_content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.sidebar_content_box.set_vexpand(True)
        return self.sidebar_content_box

    def _create_right_content_area(self) -> None:
        """Create the right content area with header bar."""
        # Create ToolbarView for right pane
        right_toolbar_view = Adw.ToolbarView()

        # Detect window button layout
        window_buttons_left = self._window_buttons_on_left()

        # Create custom header bar (Add Files, Start OCR, etc)
        self.custom_header_bar = HeaderBar(self, window_buttons_left)
        right_toolbar_view.add_top_bar(self.custom_header_bar)

        # Set up help action
        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", lambda *_: self.show_welcome_dialog())
        self.add_action(help_action)

        # Set up reset settings action
        reset_action = Gio.SimpleAction.new("reset_settings", None)
        reset_action.connect("activate", lambda *_: self._confirm_reset_settings())
        self.add_action(reset_action)

        # Create stack for right content (file queue only for now)
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        self.stack.set_transition_duration(300)

        # Content scroll for the file queue
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(450)
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        right_toolbar_view.set_content(content_scroll)

        self.split_view.set_content(right_toolbar_view)

    def setup_pages(self) -> None:
        """Set up the application pages."""
        # File queue goes into the right stack
        settings_page = self.ui.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        # Terminal page goes into main_stack (full-width) - with Cancel button
        terminal_toolbar = self._create_full_width_page_with_header("terminal", is_terminal=True)
        terminal_content = self.ui.create_terminal_page()
        terminal_toolbar.set_content(terminal_content)
        self.main_stack.add_titled(terminal_toolbar, "terminal", _("Processing"))

        # Conclusion page goes into main_stack (full-width) - with Back button
        conclusion_toolbar = self._create_full_width_page_with_header("complete")
        conclusion_content = self.ui.create_conclusion_page()
        conclusion_clamp = Adw.Clamp()
        conclusion_clamp.set_maximum_size(800)
        conclusion_clamp.set_tightening_threshold(600)
        conclusion_clamp.set_child(conclusion_content)
        conclusion_toolbar.set_content(conclusion_clamp)
        self.main_stack.add_titled(conclusion_toolbar, "conclusion", _("Results"))

        # Populate sidebar settings
        self._populate_sidebar_settings()

    def _create_full_width_page_with_header(
        self, _view_mode: str, is_terminal: bool = False
    ) -> Adw.ToolbarView:
        """Create a full-width page with its own header bar.

        Args:
            view_mode: The view mode for the header bar
            is_terminal: If True, no navigation button is added (Cancel is in the content area)

        Returns:
            ToolbarView with header
        """
        toolbar_view = Adw.ToolbarView()

        # Create a simpler header bar for non-settings pages
        header = Adw.HeaderBar()
        header.set_decoration_layout("menu:minimize,maximize,close")

        if not is_terminal:
            # Back button for conclusion page
            back_button = Gtk.Button()
            back_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            back_icon = Gtk.Image.new_from_icon_name("go-previous-symbolic")
            back_label = Gtk.Label(label=_("Back"))
            back_box.append(back_icon)
            back_box.append(back_label)
            back_button.set_child(back_box)
            back_button.add_css_class("suggested-action")
            set_a11y_label(back_button, _("Back"))
            back_button.connect("clicked", lambda _: self._return_to_main_view())
            header.pack_start(back_button)

        # Title
        header.set_title_widget(Gtk.Label(label="Big OCR PDF"))

        # Menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_tooltip_text(_("Menu"))
        set_a11y_label(menu_button, _("Menu"))
        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Help"), "win.help")
        menu.append(_("About"), "app.about")
        menu.append(_("Quit"), "app.quit")
        menu_button.set_menu_model(menu)
        header.pack_end(menu_button)

        toolbar_view.add_top_bar(header)

        return toolbar_view

    def _return_to_main_view(self) -> None:
        """Return to the main view (settings page) and clear processed files."""
        # Reset conclusion page labels to avoid stale data on next run
        if hasattr(self, "ui") and hasattr(self.ui, "conclusion_page_manager"):
            self.ui.conclusion_page_manager.reset_page()

        # Clear processing results via shared method
        self.settings.reset_processing_state()

        # Clear the file queue (files have been processed)
        self.clear_file_queue()

        # Navigate back to main view
        self.main_stack.set_visible_child_name("main_view")
        self.custom_header_bar.set_view("queue")

    def _populate_sidebar_settings(self) -> None:
        """Populate the sidebar with OCR settings content."""
        if hasattr(self.ui, "settings_page_manager") and self.ui.settings_page_manager:
            settings_content = self.ui.settings_page_manager.create_sidebar_content()
            # Set content directly on the toolbar view
            self.left_toolbar_view.set_content(settings_content)

    def _on_main_stack_changed(self, stack: Adw.ViewStack, _param) -> None:
        """Handle main stack page changes."""
        current_page = stack.get_visible_child_name()
        logger.debug(f"Main stack changed to: {current_page}")

    def announce_status(self, message: str) -> None:
        """Announce a status message via the accessible live region.

        Screen readers (Orca) will read the text automatically because
        ``_a11y_status`` has ``Gtk.AccessibleRole.STATUS``.

        Args:
            message: Text to announce
        """
        self._a11y_status.set_text(message)
