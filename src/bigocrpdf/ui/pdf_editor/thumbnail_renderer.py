"""
BigOcrPdf - PDF Thumbnail Renderer

Renders PDF page thumbnails using pdftoppm (CLI) for performance and thread safety.
Maintains an LRU cache of rendered thumbnails.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Poppler", "0.18")
from gi.repository import GdkPixbuf, GLib, Poppler

from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    pass


class ThumbnailRenderer:
    """Renders PDF page thumbnails with caching and lazy loading.

    Uses `pdftoppm` (CLI) in background threads for rendering to avoid
    blocking the UI and to bypass GObject Introspection thread-safety issues.
    """

    def __init__(self, cache_size: int = 100, default_size: int = 100) -> None:
        """Initialize the thumbnail renderer.

        Args:
            cache_size: Maximum number of thumbnails to cache
            default_size: Default thumbnail width in pixels
        """
        self._cache: OrderedDict[str, GdkPixbuf.Pixbuf] = OrderedDict()
        self._cache_size = cache_size
        self._default_size = default_size
        self._documents: dict[str, Poppler.Document] = {}
        self._lock = threading.Lock()

    def _get_cache_key(self, pdf_path: str, page_num: int, size: int, rotation: int) -> str:
        return f"{pdf_path}:{page_num}:{size}:{rotation}"

    def _get_document(self, pdf_path: str) -> Poppler.Document | None:
        """Get or load a Poppler document (for metadata only)."""
        if pdf_path in self._documents:
            return self._documents[pdf_path]

        try:
            uri = GLib.filename_to_uri(pdf_path, None)
            doc = Poppler.Document.new_from_file(uri, None)
            self._documents[pdf_path] = doc
            return doc
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            return None

    def _evict_cache(self) -> None:
        """Evict oldest items from cache."""
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def render_page_thumbnail_async(
        self,
        pdf_path: str,
        page_num: int,
        callback: Callable[[GdkPixbuf.Pixbuf | None], None],
        size: int | None = None,
        rotation: int = 0,
    ) -> None:
        """Render a thumbnail asynchronously using pdftoppm in a thread."""
        if size is None:
            size = self._default_size

        cache_key = self._get_cache_key(pdf_path, page_num, size, rotation)

        # Check cache
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                pixbuf = self._cache[cache_key]
                GLib.idle_add(lambda: callback(pixbuf))
                return

        # Start background thread
        thread = threading.Thread(
            target=self._render_worker,
            args=(pdf_path, page_num, size, rotation, callback, cache_key),
        )
        thread.daemon = True
        thread.start()

    def _render_worker(
        self,
        pdf_path: str,
        page_num: int,
        size: int,
        rotation: int,
        callback: Callable[[GdkPixbuf.Pixbuf | None], None],
        cache_key: str,
    ):
        """Worker thread logic."""
        try:
            lower_path = pdf_path.lower()
            is_image = lower_path.endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
            )

            if is_image:
                # Process image in main thread to avoid Gdk/Gtk threading issues
                # Or just load file bytes here?
                # GdkPixbuf.new_from_file is NOT thread safe if it touches Gdk.
                # But widely used in threads... actually it is NOT safe.
                # Safe way: Signal main thread to load.
                GLib.idle_add(
                    lambda: self._finish_render_image(pdf_path, size, rotation, callback, cache_key)
                )
                return

            # PDF Processing using pdftoppm via utility function
            from bigocrpdf.utils.pdf_utils import render_pdf_page_to_png

            data = render_pdf_page_to_png(pdf_path, page_num, size)

            if data:
                # Pass data to main thread completion
                GLib.idle_add(
                    lambda: self._finish_render_pdf(data, size, rotation, callback, cache_key)
                )
            else:
                logger.error(f"pdftoppm produced no output for page {page_num}")
                GLib.idle_add(lambda: callback(self._create_error_pixbuf(size)))

        except Exception as e:
            logger.error(f"Render worker error: {e}")
            GLib.idle_add(lambda: callback(self._create_error_pixbuf(size)))

    def _finish_render_image(self, path, size, rotation, callback, cache_key):
        """Finish image rendering on main thread."""
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            pixbuf = self._apply_transform(pixbuf, size, rotation)

            with self._lock:
                self._cache[cache_key] = pixbuf
                self._evict_cache()

            callback(pixbuf)
        except Exception as e:
            logger.error(f"Image load error: {e}")
            callback(self._create_error_pixbuf(size))

    def _finish_render_pdf(self, data, size, rotation, callback, cache_key):
        """Finish PDF rendering on main thread with raw PNG data."""
        try:
            loader = GdkPixbuf.PixbufLoader.new_with_type("png")
            loader.write(data)
            loader.close()
            pixbuf = loader.get_pixbuf()

            # pdftoppm handles scaling (-scale-to), but we need to handle rotation
            # Wait, pdftoppm respects PDF active rotation.
            # But the 'rotation' arg is the EDITOR visual rotation (overrides/adds).
            # So we must apply it.

            if rotation != 0:
                pixbuf = self._apply_rotation(pixbuf, rotation)

            with self._lock:
                self._cache[cache_key] = pixbuf
                self._evict_cache()

            callback(pixbuf)
        except Exception as e:
            logger.error(f"PDF finish error: {e}")
            callback(self._create_error_pixbuf(size))

    def _apply_transform(self, pixbuf, size, rotation):
        """Resize and rotate a pixbuf."""
        # Scale
        w, h = pixbuf.get_width(), pixbuf.get_height()
        scale = size / w
        new_w, new_h = int(w * scale), int(h * scale)
        pixbuf = pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)

        # Rotate
        if rotation != 0:
            pixbuf = self._apply_rotation(pixbuf, rotation)
        return pixbuf

    def _apply_rotation(self, pixbuf, rotation):
        """Apply rotation to pixbuf."""
        rot = rotation % 360
        if rot == 90:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.CLOCKWISE)
        elif rot == 180:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.UPSIDEDOWN)
        elif rot == 270:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.COUNTERCLOCKWISE)
        return pixbuf

    def _create_error_pixbuf(self, size: int) -> GdkPixbuf.Pixbuf:
        """Create a placeholder pixbuf for error cases."""
        import cairo

        height = int(size * 1.414)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.paint()
        ctx.set_source_rgb(0.8, 0.2, 0.2)
        ctx.set_line_width(max(2, size / 30))
        m = size * 0.2
        ctx.move_to(m, m)
        ctx.line_to(size - m, height - m)
        ctx.stroke()
        ctx.move_to(size - m, m)
        ctx.line_to(m, height - m)
        ctx.stroke()

        # Convert surface to pixbuf (inline logic to avoid dependency on missing method)
        return GdkPixbuf.Pixbuf.get_from_surface(surface, 0, 0, size, height)

    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        lower_path = pdf_path.lower()
        if lower_path.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")):
            return 1
        doc = self._get_document(pdf_path)
        return doc.get_n_pages() if doc else 0

    def clear_document_cache(self, pdf_path: str) -> None:
        with self._lock:
            if pdf_path in self._documents:
                del self._documents[pdf_path]
            # Inefficient but safe removal
            to_del = [k for k in self._cache if k.startswith(f"{pdf_path}:")]
            for k in to_del:
                del self._cache[k]


# Global instance
_renderer: ThumbnailRenderer | None = None


def get_thumbnail_renderer() -> ThumbnailRenderer:
    global _renderer
    if _renderer is None:
        _renderer = ThumbnailRenderer()
    return _renderer
