"""
BigOcrPdf - PDF Editor Page Model

Data models for PDF document and page state management.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gi.repository import GdkPixbuf


@dataclass
class PageState:
    """State of a single PDF page.

    Attributes:
        page_number: Original page number (1-indexed)
        rotation: Rotation angle in degrees (0, 90, 180, 270)
        included_for_ocr: Whether this page should be processed by OCR
        deleted: Whether page is marked for deletion (soft delete)
        position: Current position after reordering (0-indexed)
        source_file: Source PDF file path (for merged documents)
        thumbnail_pixbuf: Cached thumbnail pixbuf (None until rendered)
    """

    page_number: int
    rotation: int = 0
    included_for_ocr: bool = True
    deleted: bool = False
    position: int = 0
    source_file: str = ""
    thumbnail_pixbuf: "GdkPixbuf.Pixbuf | None" = None

    def __post_init__(self) -> None:
        """Validate and normalize rotation angle."""
        self.rotation = self.rotation % 360
        if self.rotation not in (0, 90, 180, 270):
            # Round to nearest valid rotation
            self.rotation = round(self.rotation / 90) * 90 % 360

    def rotate(self, degrees: int) -> None:
        """Rotate page by specified degrees."""
        self.rotation = (self.rotation + degrees) % 360
        if self.rotation not in (0, 90, 180, 270):
            # Round to nearest valid rotation
            self.rotation = round(self.rotation / 90) * 90 % 360

    def rotate_left(self) -> None:
        """Rotate page 90 degrees counter-clockwise."""
        self.rotation = (self.rotation - 90) % 360

    def rotate_right(self) -> None:
        """Rotate page 90 degrees clockwise."""
        self.rotation = (self.rotation + 90) % 360

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the page state
        """
        return {
            "page_number": self.page_number,
            "rotation": self.rotation,
            "included_for_ocr": self.included_for_ocr,
            "deleted": self.deleted,
            "position": self.position,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PageState":
        """Create PageState from dictionary.

        Args:
            data: Dictionary with page state data

        Returns:
            New PageState instance
        """
        return cls(
            page_number=data.get("page_number", 1),
            rotation=data.get("rotation", 0),
            included_for_ocr=data.get("included_for_ocr", True),
            deleted=data.get("deleted", False),
            position=data.get("position", 0),
            source_file=data.get("source_file", ""),
        )


@dataclass
class PDFDocument:
    """Represents a PDF document being edited.

    Attributes:
        path: Path to the source PDF file
        pages: List of PageState objects for each page
        split_points: Page indices where splits should occur
        modified: Whether document has unsaved changes
        total_pages: Total number of pages in the document
    """

    path: str
    pages: list[PageState] = field(default_factory=list)
    split_points: list[int] = field(default_factory=list)
    modified: bool = False
    total_pages: int = 0

    def __post_init__(self) -> None:
        """Initialize pages list if not provided."""
        if not self.pages and self.total_pages > 0:
            self.pages = [
                PageState(
                    page_number=i + 1,
                    position=i,
                    source_file=self.path,
                )
                for i in range(self.total_pages)
            ]

    def get_active_pages(self) -> list[PageState]:
        """Get pages that are not marked as deleted.

        Returns:
            List of non-deleted pages sorted by position
        """
        return sorted(
            [p for p in self.pages if not p.deleted],
            key=lambda p: p.position,
        )

    def get_page_by_position(self, position: int) -> PageState | None:
        """Get page at the specified position.

        Args:
            position: Position index (0-based)

        Returns:
            PageState at the position, or None if not found
        """
        for page in self.pages:
            if page.position == position and not page.deleted:
                return page
        return None

    def mark_modified(self) -> None:
        """Mark the document as modified."""
        self.modified = True

    def clear_modifications(self) -> None:
        """Clear the modified flag."""
        self.modified = False

    def update_positions(self) -> None:
        """Update positions to be sequential after changes."""
        active_pages = self.get_active_pages()
        for i, page in enumerate(active_pages):
            page.position = i

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the document
        """
        return {
            "path": self.path,
            "pages": [p.to_dict() for p in self.pages],
            "split_points": self.split_points,
            "total_pages": self.total_pages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PDFDocument":
        """Create PDFDocument from dictionary.

        Args:
            data: Dictionary with document data

        Returns:
            New PDFDocument instance
        """
        doc = cls(
            path=data.get("path", ""),
            total_pages=data.get("total_pages", 0),
        )
        doc.pages = [PageState.from_dict(p) for p in data.get("pages", [])]
        doc.split_points = data.get("split_points", [])
        return doc
