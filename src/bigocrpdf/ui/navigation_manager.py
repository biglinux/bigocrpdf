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
    def main_stack(self) -> Gtk.Stack:
        """Get the main stack from the window (for terminal/conclusion pages)."""
        return self.window.main_stack

    def get_current_page(self) -> str:
        """
        Get the name of the current visible page.

        Returns:
            Name of the current page
        """
        # Check main_stack first for terminal/conclusion
        main_page = self.main_stack.get_visible_child_name()
        if main_page in (self.PAGE_TERMINAL, self.PAGE_CONCLUSION):
            return main_page
        return self.PAGE_SETTINGS  # Default to settings when on main_view

    def navigate_to(self, page_name: str) -> None:
        """
        Navigate to a specific page.

        Args:
            page_name: Name of the page to navigate to
        """
        if page_name not in self._page_states:
            logger.warning(f"Unknown page: {page_name}")
            return

        if page_name == self.PAGE_SETTINGS:
            # Navigate to main view (which shows settings)
            self.main_stack.set_visible_child_name("main_view")
            self.stack.set_visible_child_name("settings")
        elif page_name in (self.PAGE_TERMINAL, self.PAGE_CONCLUSION):
            # Navigate to full-width pages in main_stack
            self.main_stack.set_visible_child_name(page_name)

        self._update_header_bar_for_page(page_name)
        logger.debug(f"Navigated to page: {page_name}")

    def _update_header_bar_for_page(self, page_name: str) -> None:
        """
        Update header bar buttons for the current page.

        Args:
            page_name: Name of the current page
        """
        if not hasattr(self.window, "custom_header_bar") or not self.window.custom_header_bar:
            return

        if page_name == self.PAGE_SETTINGS:
            self.window.custom_header_bar.set_view("queue")
        elif page_name == self.PAGE_TERMINAL:
            self.window.custom_header_bar.set_view("processing")
        elif page_name == self.PAGE_CONCLUSION:
            self.window.custom_header_bar.set_view("complete")

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

    def restore_next_button(self) -> None:
        """Restore the header bar to its default state for the current page."""
        self._update_header_bar_for_page(self.get_current_page())
