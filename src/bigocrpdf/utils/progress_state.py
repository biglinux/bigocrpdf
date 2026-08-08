"""
BigOcrPdf - Progress State Module

This module provides a dataclass for tracking UI progress display state.
"""

from dataclasses import dataclass, field


@dataclass
class ProgressState:
    """Track the state of progress display to avoid redundant updates.

    This class helps minimize UI updates by tracking what was last displayed,
    allowing the code to only update when there are meaningful changes.

    Attributes:
        fraction: Current progress fraction (0.0-1.0)
        text: Progress bar text (e.g., "50%")
        status: Status bar text
    """

    fraction: float = 0.0
    text: str = ""
    status: str = ""
    # Threshold for progress update (1%)
    _threshold: float = field(default=0.01, repr=False)

    def update_fraction(self, new_fraction: float) -> bool:
        """Update fraction if changed significantly.

        Args:
            new_fraction: New progress value

        Returns:
            True if updated, False if unchanged
        """
        changed = new_fraction != self.fraction
        is_endpoint = new_fraction in (0.0, 1.0)
        if changed and (is_endpoint or abs(new_fraction - self.fraction) >= self._threshold):
            self.fraction = new_fraction
            return True
        return False

    def update_text(self, new_text: str) -> bool:
        """Update text if changed.

        Args:
            new_text: New text value

        Returns:
            True if updated, False if unchanged
        """
        if new_text != self.text:
            self.text = new_text
            return True
        return False

    def update_status(self, new_status: str) -> bool:
        """Update status if changed.

        Args:
            new_status: New status value

        Returns:
            True if updated, False if unchanged
        """
        if new_status != self.status:
            self.status = new_status
            return True
        return False

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.fraction = 0.0
        self.text = ""
        self.status = ""

    def get_percentage(self) -> int:
        """Get current progress as integer percentage.

        Returns:
            Progress as integer 0-100
        """
        return int(self.fraction * 100)
