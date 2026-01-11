"""
BigOcrPdf - Navigation Manager Module

Handles page navigation and step label management.
"""

import gi

gi.require_version("Gtk", "4.0")
from collections.abc import Callable
from typing import TYPE_CHECKING

from gi.repository import Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from window import BigOcrPdfWindow


class NavigationState:
    """Represents the state of a navigation page."""

    def __init__(
        self,
        step_text: str,
        back_enabled: bool = False,
        back_visible: bool = False,
        next_enabled: bool = True,
        next_visible: bool = True,
        next_label: str = "Next",
    ):
        """
        Initialize navigation state.

        Args:
            step_text: Text to display in the step label
            back_enabled: Whether the back button is enabled
            back_visible: Whether the back button is visible
            next_enabled: Whether the next button is enabled
            next_visible: Whether the next button is visible
            next_label: Label for the next button
        """
        self.step_text = step_text
        self.back_enabled = back_enabled
        self.back_visible = back_visible
        self.next_enabled = next_enabled
        self.next_visible = next_visible
        self.next_label = next_label


class NavigationManager:
    """
    Manages page navigation and step labels.

    This class handles:
    - Page transitions
    - Step label updates
    - Back/Next button states
    - Navigation history
    """

    # Page names
    PAGE_SETTINGS = "settings"
    PAGE_TERMINAL = "terminal"
    PAGE_CONCLUSION = "conclusion"

    # Step labels
    STEP_SETTINGS = "Step 1/3: Settings"
    STEP_TERMINAL = "Step 2/3: Processing"
    STEP_CONCLUSION = "Step 3/3: Results"

    def __init__(self, window: "BigOcrPdfWindow"):
        """
        Initialize the navigation manager.

        Args:
            window: Reference to the main application window
        """
        self.window = window
        self._on_apply_callback: Callable | None = None
        self._on_reset_callback: Callable | None = None

        # Define navigation states for each page
        self._page_states: dict[str, NavigationState] = {
            self.PAGE_SETTINGS: NavigationState(
                step_text=_(self.STEP_SETTINGS),
                back_enabled=False,
                back_visible=False,
                next_enabled=True,
                next_visible=True,
                next_label=_("Start"),
            ),
            self.PAGE_TERMINAL: NavigationState(
                step_text=_(self.STEP_TERMINAL),
                back_enabled=True,
                back_visible=True,
                next_enabled=False,
                next_visible=False,
                next_label=_("Cancel"),
            ),
            self.PAGE_CONCLUSION: NavigationState(
                step_text=_(self.STEP_CONCLUSION),
                back_enabled=True,
                back_visible=True,
                next_enabled=True,
                next_visible=True,
                next_label=_("Process New Files"),
            ),
        }

    def set_on_apply_callback(self, callback: Callable) -> None:
        """
        Set callback for when apply/start is triggered.

        Args:
            callback: Function to call when starting processing
        """
        self._on_apply_callback = callback

    def set_on_reset_callback(self, callback: Callable) -> None:
        """
        Set callback for when reset is triggered.

        Args:
            callback: Function to call when resetting state
        """
        self._on_reset_callback = callback

    @property
    def stack(self) -> Gtk.Stack:
        """Get the page stack from the window."""
        return self.window.stack

    @property
    def back_button(self) -> Gtk.Button:
        """Get the back button widget."""
        return self.window.back_button

    @property
    def next_button(self) -> Gtk.Button:
        """Get the next button widget."""
        return self.window.next_button

    def get_current_page(self) -> str:
        """
        Get the name of the current visible page.

        Returns:
            Name of the current page
        """
        return self.stack.get_visible_child_name()

    def navigate_to(self, page_name: str) -> None:
        """
        Navigate to a specific page.

        Args:
            page_name: Name of the page to navigate to
        """
        if page_name not in self._page_states:
            logger.warning(f"Unknown page: {page_name}")
            return

        self.stack.set_visible_child_name(page_name)
        self._apply_page_state(page_name)
        self._update_step_indicator(page_name)

        logger.debug(f"Navigated to page: {page_name}")

    def _apply_page_state(self, page_name: str) -> None:
        """
        Apply the navigation state for a page.

        Args:
            page_name: Name of the page
        """
        state = self._page_states.get(page_name)
        if not state:
            return

        # Update back button
        self.back_button.set_sensitive(state.back_enabled)
        self.back_button.set_visible(state.back_visible)

        # Update next button
        self.next_button.set_sensitive(state.next_enabled)
        self.next_button.set_visible(state.next_visible)
        self.next_button.set_label(state.next_label)

    def _update_step_indicator(self, page_name: str) -> None:
        """
        Update the step indicator based on the current page.

        Args:
            page_name: Name of the current page
        """
        # Map page names to step indices
        step_mapping = {
            self.PAGE_SETTINGS: 0,
            self.PAGE_TERMINAL: 1,
            self.PAGE_CONCLUSION: 2,
        }

        current_step = step_mapping.get(page_name, 0)
        completed_steps = list(range(current_step))

        # Call window's set_current_step method
        self.window.set_current_step(current_step, completed_steps)

    def handle_back_clicked(self, _button: Gtk.Button = None) -> None:
        """
        Handle back button click.

        Args:
            _button: The button that was clicked (unused)
        """
        current_page = self.get_current_page()

        if current_page == self.PAGE_TERMINAL:
            self._navigate_from_terminal_to_settings()
        elif current_page == self.PAGE_CONCLUSION:
            self._reset_to_settings()

    def handle_next_clicked(self, button: Gtk.Button = None) -> None:
        """
        Handle next button click.

        Args:
            button: The button that was clicked
        """
        current_page = self.get_current_page()

        if current_page == self.PAGE_SETTINGS:
            self._start_processing(button)
        elif current_page == self.PAGE_CONCLUSION:
            self._reset_to_settings()

    def _navigate_from_terminal_to_settings(self) -> None:
        """Navigate from terminal page back to settings."""
        self.navigate_to(self.PAGE_SETTINGS)
        self.window.update_file_info()

    def _start_processing(self, button: Gtk.Button = None) -> None:
        """
        Start the OCR processing.

        Args:
            button: The button that triggered the action
        """
        if self._on_apply_callback:
            self._on_apply_callback(button)

    def _reset_to_settings(self) -> None:
        """Reset state and go back to settings page."""
        if self._on_reset_callback:
            self._on_reset_callback()

    def navigate_to_terminal(self) -> None:
        """Navigate to the terminal/processing page."""
        self.navigate_to(self.PAGE_TERMINAL)

    def navigate_to_conclusion(self) -> None:
        """Navigate to the conclusion/results page."""
        self.navigate_to(self.PAGE_CONCLUSION)

    def navigate_to_settings(self) -> None:
        """Navigate to the settings page."""
        self.navigate_to(self.PAGE_SETTINGS)

    def update_processing_buttons(self, is_processing: bool) -> None:
        """
        Update button states during processing.

        Args:
            is_processing: Whether processing is active
        """
        if is_processing:
            self.back_button.set_sensitive(False)
            self.next_button.set_sensitive(False)
            self.next_button.set_visible(False)
        else:
            state = self._page_states.get(self.get_current_page())
            if state:
                self.back_button.set_sensitive(state.back_enabled)
                self.next_button.set_sensitive(state.next_enabled)
                self.next_button.set_visible(state.next_visible)

    def show_cancel_button(self, on_cancel_callback: Callable) -> None:
        """
        Show and configure the cancel button during processing.

        Args:
            on_cancel_callback: Callback for when cancel is clicked
        """
        self.window.disconnect_signal(self.next_button, "clicked")
        self.window.connect_signal(self.next_button, "clicked", on_cancel_callback)

        self.next_button.set_label(_("Cancel"))
        self.next_button.set_sensitive(True)
        self.next_button.set_visible(True)

    def restore_next_button(self) -> None:
        """Restore the next button to its default state for the current page."""
        self.window.disconnect_signal(self.next_button, "clicked")
        self.window.connect_signal(self.next_button, "clicked", self.handle_next_clicked)

        state = self._page_states.get(self.get_current_page())
        if state:
            self.next_button.set_label(state.next_label)
            self.next_button.set_sensitive(state.next_enabled)
            self.next_button.set_visible(state.next_visible)

    def on_page_changed(self, stack: Gtk.Stack, _param) -> None:
        """
        Handle page change events.

        Args:
            stack: The stack widget
            _param: The parameter that changed (unused)
        """
        current_page = stack.get_visible_child_name()

        if current_page == self.PAGE_SETTINGS:
            self.window._apply_headerbar_sidebar_style()
        else:
            self.window._remove_headerbar_sidebar_style()
