"""
BigOcrPdf - PDF Thumbnail Renderer

Renders PDF page thumbnails using pdftoppm (poppler-utils) with
thread-pooled background rendering and LRU caching for performance.
"""

import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Poppler", "0.18")
from gi.repository import Gdk, GdkPixbuf, GLib, Poppler

from bigocrpdf.utils.logger import logger


class ThumbnailRenderer:
    """Renders PDF page thumbnails with caching and lazy loading.

    Uses pdftoppm (poppler-utils C binary) for fast rendering via a
    bounded thread pool. Rendered pixbufs are cached in an LRU cache.
    """

    def __init__(self, cache_size: int = 200, default_size: int = 200) -> None:
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
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._pending: set[str] = set()  # Track in-flight cache keys
        self._tls = threading.local()  # Thread-local document cache
        self._doc_version = 0  # Bumped on cache clear to invalidate stale docs

    def _get_cache_key(self, pdf_path: str, page_num: int, size: int, rotation: int) -> str:
        return f"{pdf_path}:{page_num}:{size}:{rotation}"

    def _get_document(self, pdf_path: str) -> Poppler.Document | None:
        """Get or load a Poppler document (main thread only, for metadata)."""
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
        """Render a thumbnail asynchronously using a thread pool."""
        if size is None:
            size = self._default_size

        cache_key = self._get_cache_key(pdf_path, page_num, size, rotation)

        # Check cache
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                pixbuf = self._cache[cache_key]
                GLib.idle_add(lambda pb=pixbuf: callback(pb))
                return
            # Skip if already being rendered
            if cache_key in self._pending:
                return
            self._pending.add(cache_key)

        # Submit to thread pool
        self._pool.submit(
            self._render_worker, pdf_path, page_num, size, rotation, callback, cache_key
        )

    def batch_preload(
        self,
        pdf_path: str,
        page_count: int,
        callback: Callable[[], None] | None = None,
        size: int | None = None,
    ) -> None:
        """Pre-render all pages of a PDF using pdftoppm (30x faster than Poppler GI).

        Renders all pages in a single pdftoppm invocation and caches the
        resulting pixbufs. Subsequent per-page requests will hit the cache.

        Args:
            pdf_path: Path to the PDF file
            page_count: Total number of pages
            callback: Optional callback invoked on main thread when done
            size: Thumbnail width in pixels
        """
        if size is None:
            size = self._default_size
        self._pool.submit(self._batch_worker, pdf_path, page_count, size, callback)

    def _batch_worker(
        self,
        pdf_path: str,
        page_count: int,
        size: int,
        callback: Callable[[], None] | None,
    ) -> None:
        """Worker: render all pages via pdftoppm and populate cache."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                prefix = os.path.join(tmpdir, "p")
                result = subprocess.run(
                    [
                        "pdftoppm",
                        "-jpeg",
                        "-r",
                        "150",
                        "-scale-to-x",
                        str(size),
                        "-scale-to-y",
                        "-1",
                        pdf_path,
                        prefix,
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.warning(
                        f"pdftoppm failed ({result.returncode}), falling back to Poppler"
                    )
                    if callback:
                        GLib.idle_add(callback)
                    return

                files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jpg"))
                for idx, fname in enumerate(files):
                    if idx >= page_count:
                        break
                    fpath = os.path.join(tmpdir, fname)
                    try:
                        pixbuf = GdkPixbuf.Pixbuf.new_from_file(fpath)
                    except Exception:
                        continue
                    # Cache under rotation=0 (base render)
                    cache_key = self._get_cache_key(pdf_path, idx, size, 0)
                    with self._lock:
                        self._cache[cache_key] = pixbuf
                        self._pending.discard(cache_key)

                with self._lock:
                    self._evict_cache()

                logger.info(
                    f"Batch preload: {len(files)} pages of {os.path.basename(pdf_path)} "
                    f"rendered via pdftoppm"
                )
        except Exception as e:
            logger.warning(f"Batch preload error: {e}")

        if callback:
            GLib.idle_add(callback)

    def _render_worker(
        self,
        pdf_path: str,
        page_num: int,
        size: int,
        rotation: int,
        callback: Callable[[GdkPixbuf.Pixbuf | None], None],
        cache_key: str,
    ):
        """Worker thread: render PDF page to pixbuf via cairo surface."""
        try:
            lower_path = pdf_path.lower()
            is_image = lower_path.endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".tif",
                    ".tiff",
                    ".bmp",
                )
            )

            if is_image:
                # Image loading needs GdkPixbuf on main thread
                GLib.idle_add(
                    self._finish_render_image, pdf_path, size, rotation, callback, cache_key
                )
                return

            # Use pdftoppm for fast single-page render (25x faster than Poppler GI)
            pixbuf = self._render_pdf_page_pdftoppm(pdf_path, page_num, size)

            if pixbuf:
                if rotation != 0:
                    pixbuf = self._apply_rotation(pixbuf, rotation)

                with self._lock:
                    self._cache[cache_key] = pixbuf
                    self._pending.discard(cache_key)
                    self._evict_cache()

                GLib.idle_add(lambda pb=pixbuf: callback(pb))
            else:
                logger.error(f"Failed to render page {page_num} of {pdf_path}")
                with self._lock:
                    self._pending.discard(cache_key)
                GLib.idle_add(lambda: callback(self._create_error_pixbuf(size)))

        except Exception as e:
            logger.error(f"Render worker error: {e}")
            with self._lock:
                self._pending.discard(cache_key)
            GLib.idle_add(lambda: callback(self._create_error_pixbuf(size)))

    def _render_pdf_page_pdftoppm(
        self, pdf_path: str, page_num: int, size: int
    ) -> GdkPixbuf.Pixbuf | None:
        """Render a single PDF page via pdftoppm (native C, ~25x faster)."""
        try:
            page_1based = page_num + 1
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-jpeg",
                    "-r",
                    "150",
                    "-scale-to-x",
                    str(size),
                    "-scale-to-y",
                    "-1",
                    "-f",
                    str(page_1based),
                    "-l",
                    str(page_1based),
                    "-singlefile",
                    pdf_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
                loader.write(result.stdout)
                loader.close()
                return loader.get_pixbuf()
        except Exception as e:
            logger.warning(f"pdftoppm single-page failed: {e}")

        # Fallback to Poppler GI
        return self._render_pdf_to_pixbuf(pdf_path, page_num, size)

    def _get_thread_document(self, pdf_path: str) -> Poppler.Document | None:
        """Get a Poppler document cached per worker thread.

        Poppler documents are not thread-safe so each thread keeps its own
        instance. The document is opened once per thread and reused for all
        pages, reducing PDF opens from N (one per page) to W (one per worker).
        """
        ver = self._doc_version
        if not hasattr(self._tls, "docs") or getattr(self._tls, "ver", -1) != ver:
            self._tls.docs = {}
            self._tls.ver = ver
        if pdf_path not in self._tls.docs:
            try:
                uri = GLib.filename_to_uri(pdf_path, None)
                self._tls.docs[pdf_path] = Poppler.Document.new_from_file(uri, None)
            except Exception as e:
                logger.error(f"Failed to load PDF in worker: {e}")
                return None
        return self._tls.docs[pdf_path]

    def _render_pdf_to_pixbuf(
        self, pdf_path: str, page_num: int, size: int
    ) -> GdkPixbuf.Pixbuf | None:
        """Render a PDF page to a GdkPixbuf using Poppler GI + Cairo.

        Converts the Cairo surface directly to a GdkPixbuf via
        Gdk.MemoryTexture, avoiding the slow PNG encode/decode round-trip.
        Uses a thread-local Poppler document cache for speed.
        """
        import cairo

        try:
            doc = self._get_thread_document(pdf_path)
            if doc is None:
                return None
            if page_num >= doc.get_n_pages():
                return None
            page = doc.get_page(page_num)
            pw, ph = page.get_size()
            if pw <= 0 or ph <= 0:
                return None

            scale = size / pw
            w = int(pw * scale)
            h = int(ph * scale)

            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
            ctx = cairo.Context(surface)
            ctx.set_source_rgb(1.0, 1.0, 1.0)
            ctx.paint()
            ctx.scale(scale, scale)
            page.render(ctx)
            surface.flush()

            # Direct pixel copy — Cairo ARGB32 matches B8G8R8A8_PREMULTIPLIED
            data = bytes(surface.get_data())
            stride = surface.get_stride()
            texture = Gdk.MemoryTexture.new(
                w,
                h,
                Gdk.MemoryFormat.B8G8R8A8_PREMULTIPLIED,
                GLib.Bytes.new(data),
                stride,
            )
            return Gdk.pixbuf_get_from_texture(texture)
        except Exception as e:
            logger.error(f"Error rendering PDF page with Poppler: {e}")
            return None

    def _finish_render_image(self, path, size, rotation, callback, cache_key):
        """Finish image rendering on main thread."""
        try:
            # Load at target size directly to avoid full-res decode
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, size, -1, True)
            if rotation != 0:
                pixbuf = self._apply_rotation(pixbuf, rotation)

            with self._lock:
                self._cache[cache_key] = pixbuf
                self._pending.discard(cache_key)
                self._evict_cache()

            callback(pixbuf)
        except Exception as e:
            logger.error(f"Image load error: {e}")
            with self._lock:
                self._pending.discard(cache_key)
            callback(self._create_error_pixbuf(size))

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
        surface.flush()

        data = bytes(surface.get_data())
        stride = surface.get_stride()
        texture = Gdk.MemoryTexture.new(
            size, height, Gdk.MemoryFormat.B8G8R8A8_PREMULTIPLIED, GLib.Bytes.new(data), stride
        )
        return Gdk.pixbuf_get_from_texture(texture)

    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        lower_path = pdf_path.lower()
        if lower_path.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")):
            return 1
        doc = self._get_document(pdf_path)
        return doc.get_n_pages() if doc else 0

    def clear_document_cache(self, pdf_path: str) -> None:
        """Clear cached thumbnails for a document and invalidate thread-local docs."""
        self._doc_version += 1
        with self._lock:
            to_del = [k for k in self._cache if k.startswith(f"{pdf_path}:")]
            for k in to_del:
                del self._cache[k]
            # Also clear main-thread Poppler doc
            self._documents.pop(pdf_path, None)

    def clear_all(self) -> None:
        """Clear all caches and shut down the thread pool."""
        self._doc_version += 1
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._cache.clear()
            self._documents.clear()
            self._pending.clear()
        # Re-create pool for potential reuse
        self._pool = ThreadPoolExecutor(max_workers=4)


# Global instance
_renderer: ThumbnailRenderer | None = None


def get_thumbnail_renderer() -> ThumbnailRenderer:
    global _renderer
    if _renderer is None:
        _renderer = ThumbnailRenderer()
    return _renderer
