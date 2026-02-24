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
        start_time: Processing start timestamp
    """

    fraction: float = 0.0
    text: str = ""
    status: str = ""
    start_time: float = 0.0

    # Threshold for progress update (1%)
    _threshold: float = field(default=0.01, repr=False)

    def should_update_fraction(self, new_fraction: float) -> bool:
        """Check if progress fraction has changed enough to warrant a UI update.

        Args:
            new_fraction: New progress value to compare

        Returns:
            True if the change is significant (>= 1%)
        """
        return abs(new_fraction - self.fraction) >= self._threshold

    def should_update_text(self, new_text: str) -> bool:
        """Check if progress text has changed.

        Args:
            new_text: New text to compare

        Returns:
            True if text is different
        """
        return new_text != self.text and bool(new_text)

    def should_update_status(self, new_status: str) -> bool:
        """Check if status text has changed.

        Args:
            new_status: New status to compare

        Returns:
            True if status is different
        """
        return new_status != self.status

    def update_fraction(self, new_fraction: float) -> bool:
        """Update fraction if changed significantly.

        Args:
            new_fraction: New progress value

        Returns:
            True if updated, False if unchanged
        """
        if self.should_update_fraction(new_fraction):
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
        if self.should_update_text(new_text):
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
        if self.should_update_status(new_status):
            self.status = new_status
            return True
        return False

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.fraction = 0.0
        self.text = ""
        self.status = ""
        self.start_time = 0.0

    def get_percentage(self) -> int:
        """Get current progress as integer percentage.

        Returns:
            Progress as integer 0-100
        """
        return int(self.fraction * 100)

    def is_complete(self) -> bool:
        """Check if progress indicates completion.

        Returns:
            True if progress is at or above 100%
        """
        return self.fraction >= 1.0

    def __str__(self) -> str:
        """String representation of progress state."""
        return f"Progress: {self.get_percentage()}% | {self.text} | {self.status}"
