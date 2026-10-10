"""Page navigation and step labels of the main window."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from bigocrpdf.utils.i18n import N_, _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.ui.window_ui import BigOcrPdfUI


class NavigationManager:
    """Manage page transitions and accessible step announcements."""

    # Page names
    PAGE_SETTINGS = "settings"
    PAGE_TERMINAL = "terminal"
    PAGE_CONCLUSION = "conclusion"

    # Step labels
    STEP_SETTINGS = N_("Step 1/3: Settings")
    STEP_TERMINAL = N_("Step 2/3: Processing")
    STEP_CONCLUSION = N_("Step 3/3: Results")

    def __init__(self, ui: "BigOcrPdfUI", announce_status: Callable[[str], None]) -> None:
        self.ui = ui
        self._announce_status = announce_status

        self._page_steps = {
            self.PAGE_SETTINGS: self.STEP_SETTINGS,
            self.PAGE_TERMINAL: self.STEP_TERMINAL,
            self.PAGE_CONCLUSION: self.STEP_CONCLUSION,
        }

    def navigate_to(self, page_name: str) -> None:
        """
        Navigate to a specific page.

        Args:
            page_name: Name of the page to navigate to
        """
        if page_name not in self._page_steps:
            logger.warning(f"Unknown page: {page_name}")
            return

        if page_name == self.PAGE_SETTINGS:
            # Navigate to main view (which shows settings)
            self.ui.main_stack.set_visible_child_name("main_view")
            self.ui.stack.set_visible_child_name("settings")
        elif page_name in (self.PAGE_TERMINAL, self.PAGE_CONCLUSION):
            # Navigate to full-width pages in main_stack
            self.ui.main_stack.set_visible_child_name(page_name)

        # Announce step change for screen readers (Orca)
        self._announce_status(_(self._page_steps[page_name]))

        logger.debug(f"Navigated to page: {page_name}")

    def navigate_to_terminal(self) -> None:
        """Navigate to the terminal/processing page."""
        self.navigate_to(self.PAGE_TERMINAL)

    def navigate_to_conclusion(self) -> None:
        """Navigate to the conclusion/results page."""
        self.navigate_to(self.PAGE_CONCLUSION)

    def navigate_to_settings(self) -> None:
        """Navigate to the settings page."""
        self.navigate_to(self.PAGE_SETTINGS)
