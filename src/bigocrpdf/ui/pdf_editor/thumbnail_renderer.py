"""
BigOcrPdf - PDF Thumbnail Renderer

Renders PDF page thumbnails using pdftoppm (poppler-utils) with
thread-pooled background rendering and LRU caching for performance.
"""

import os
import subprocess
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import count

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Poppler", "0.18")
from gi.repository import Gdk, GdkPixbuf, GLib, Poppler

from bigocrpdf.utils.logger import logger

DocumentIdentity = tuple[str, int | None, int | None, int | None, int | None]
ThumbnailCallback = Callable[[GdkPixbuf.Pixbuf | None], None]
ThumbnailCacheKey = tuple[DocumentIdentity, int, int, int, int]


@dataclass
class _ThumbnailJob:
    """One shared render and its independently cancelable subscribers."""

    cache_key: ThumbnailCacheKey
    callbacks: dict[int, tuple["ThumbnailRequest", ThumbnailCallback]]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = None
    process: subprocess.Popen[bytes] | None = None


class ThumbnailRequest:
    """Cancelable subscription to one thumbnail result."""

    def __init__(
        self,
        renderer: "ThumbnailRenderer",
        cache_key: ThumbnailCacheKey,
        request_id: int,
    ) -> None:
        self._renderer_ref = weakref.ref(renderer)
        self._cache_key = cache_key
        self._request_id = request_id
        self._lock = threading.Lock()
        self._cancelled = False
        self._finished = False

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled or self._finished:
                return
            self._cancelled = True
        renderer = self._renderer_ref()
        if renderer is not None:
            renderer._cancel_request(self._cache_key, self._request_id)

    def _claim_delivery(self) -> bool:
        with self._lock:
            if self._cancelled or self._finished:
                return False
            self._finished = True
            return True

    def _finish_without_delivery(self) -> None:
        with self._lock:
            self._finished = True


class ThumbnailRenderer:
    """Renders PDF page thumbnails with caching and lazy loading.

    Uses pdftoppm (poppler-utils C binary) for fast rendering via a
    bounded thread pool. Rendered pixbufs are cached in an LRU cache.
    """

    def __init__(
        self,
        cache_size: int = 200,
        default_size: int = 200,
        max_cache_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        """Initialize the thumbnail renderer.

        Args:
            cache_size: Maximum number of thumbnails to cache
            default_size: Default thumbnail width in pixels
            max_cache_bytes: Approximate maximum pixel memory retained by the LRU cache
        """
        self._cache: OrderedDict[ThumbnailCacheKey, GdkPixbuf.Pixbuf] = OrderedDict()
        self._cache_costs: dict[ThumbnailCacheKey, int] = {}
        self._cache_bytes = 0
        self._cache_size = cache_size
        self._max_cache_bytes = max_cache_bytes
        self._default_size = default_size
        self._documents: dict[DocumentIdentity, Poppler.Document] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._jobs: dict[ThumbnailCacheKey, _ThumbnailJob] = {}
        self._request_ids = count(1)
        self._document_generations: dict[str, int] = {}
        self._tls = threading.local()  # Thread-local document cache
        self._doc_version = 0  # Bumped on cache clear to invalidate stale docs
        self._shutdown = False

    @staticmethod
    def _canonical_path(path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    def _get_document_identity(self, path: str) -> DocumentIdentity:
        canonical_path = self._canonical_path(path)
        try:
            file_stat = os.stat(canonical_path, follow_symlinks=False)
        except OSError:
            return canonical_path, None, None, None, None
        return (
            canonical_path,
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )

    def _get_cache_key(
        self, pdf_path: str, page_num: int, size: int, rotation: int
    ) -> ThumbnailCacheKey:
        identity = self._get_document_identity(pdf_path)
        with self._lock:
            generation = self._document_generations.get(identity[0], 0)
        return identity, generation, page_num, size, rotation

    def _get_document(self, pdf_path: str) -> Poppler.Document | None:
        """Get or load a Poppler document (main thread only, for metadata)."""
        identity = self._get_document_identity(pdf_path)
        if identity in self._documents:
            return self._documents[identity]

        try:
            uri = GLib.filename_to_uri(identity[0], None)
            doc = Poppler.Document.new_from_file(uri, None)
            self._documents = {
                key: value for key, value in self._documents.items() if key[0] != identity[0]
            }
            self._documents[identity] = doc
            return doc
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            return None

    def _evict_cache(self) -> None:
        """Evict oldest items from cache."""
        while len(self._cache) > self._cache_size or self._cache_bytes > self._max_cache_bytes:
            cache_key, _pixbuf = self._cache.popitem(last=False)
            self._cache_bytes -= self._cache_costs.pop(cache_key, 0)

    @staticmethod
    def _pixbuf_cost(pixbuf: GdkPixbuf.Pixbuf) -> int:
        try:
            return max(0, int(pixbuf.get_rowstride()) * int(pixbuf.get_height()))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _store_cache_locked(
        self,
        cache_key: ThumbnailCacheKey,
        pixbuf: GdkPixbuf.Pixbuf,
    ) -> None:
        if cache_key in self._cache:
            self._cache_bytes -= self._cache_costs.pop(cache_key, 0)
        self._cache[cache_key] = pixbuf
        cost = self._pixbuf_cost(pixbuf)
        self._cache_costs[cache_key] = cost
        self._cache_bytes += cost
        self._evict_cache()

    def render_page_thumbnail_async(
        self,
        pdf_path: str,
        page_num: int,
        callback: ThumbnailCallback,
        size: int | None = None,
        rotation: int = 0,
    ) -> ThumbnailRequest:
        """Render a thumbnail asynchronously using a thread pool."""
        if size is None:
            size = self._default_size

        cache_key = self._get_cache_key(pdf_path, page_num, size, rotation)
        request_id = next(self._request_ids)
        request = ThumbnailRequest(self, cache_key, request_id)
        cached_pixbuf: GdkPixbuf.Pixbuf | None = None

        # Check cache
        with self._lock:
            if self._shutdown:
                raise RuntimeError("thumbnail renderer is shut down")
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                cached_pixbuf = self._cache[cache_key]
            else:
                job = self._jobs.get(cache_key)
                if job is not None and not job.cancel_event.is_set():
                    job.callbacks[request_id] = (request, callback)
                    return request
                job = _ThumbnailJob(
                    cache_key=cache_key,
                    callbacks={request_id: (request, callback)},
                )
                self._jobs[cache_key] = job
                try:
                    job.future = self._pool.submit(
                        self._render_worker,
                        pdf_path,
                        page_num,
                        size,
                        rotation,
                        job,
                    )
                except RuntimeError:
                    if self._jobs.get(cache_key) is job:
                        self._jobs.pop(cache_key, None)
                    GLib.idle_add(self._deliver_callbacks, [(request, callback)], None)
                return request

        GLib.idle_add(self._deliver_callbacks, [(request, callback)], cached_pixbuf)
        return request

    def _cancel_request(self, cache_key: ThumbnailCacheKey, request_id: int) -> None:
        process: subprocess.Popen[bytes] | None = None
        future: Future[None] | None = None
        job: _ThumbnailJob | None = None
        with self._lock:
            job = self._jobs.get(cache_key)
            if job is None:
                return
            job.callbacks.pop(request_id, None)
            if job.callbacks:
                return
            job.cancel_event.set()
            future = job.future
            process = job.process

        cancelled_before_start = future is not None and future.cancel()
        if process is not None:
            self._request_process_termination(process)
        if cancelled_before_start:
            with self._lock:
                if self._jobs.get(cache_key) is job:
                    self._jobs.pop(cache_key, None)

    def _render_worker(
        self,
        pdf_path: str,
        page_num: int,
        size: int,
        rotation: int,
        job: _ThumbnailJob,
    ) -> None:
        """Worker thread: render PDF page to pixbuf via cairo surface."""
        try:
            if job.cancel_event.is_set():
                self._finish_cancelled_job(job)
                return
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
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(pdf_path, size, -1, True)
            else:
                # Use pdftoppm for fast single-page render (25x faster than Poppler GI)
                pixbuf = self._render_pdf_page_pdftoppm(
                    pdf_path,
                    page_num,
                    size,
                    job.cancel_event,
                    job,
                )

            if job.cancel_event.is_set():
                self._finish_cancelled_job(job)
            elif pixbuf:
                if rotation != 0:
                    pixbuf = self._apply_rotation(pixbuf, rotation)
                self._complete_render(job, pixbuf)
            else:
                logger.error(f"Failed to render page {page_num} of {pdf_path}")
                GLib.idle_add(self._complete_render_error, job, size)

        except Exception as e:
            if job.cancel_event.is_set():
                self._finish_cancelled_job(job)
            else:
                logger.error(f"Render worker error: {e}")
                GLib.idle_add(self._complete_render_error, job, size)

    def _complete_render_error(self, job: _ThumbnailJob, size: int) -> bool:
        if job.cancel_event.is_set():
            self._finish_cancelled_job(job)
            return False
        self._complete_render(job, self._create_error_pixbuf(size), cache_result=False)
        return False

    def _complete_render(
        self,
        job: _ThumbnailJob,
        pixbuf: GdkPixbuf.Pixbuf,
        *,
        cache_result: bool = True,
    ) -> None:
        cache_key = job.cache_key
        identity, generation, _page_num, _size, _rotation = cache_key
        current_identity = self._get_document_identity(identity[0])
        with self._lock:
            if self._jobs.get(cache_key) is not job:
                return
            self._jobs.pop(cache_key, None)
            job.process = None
            callbacks = list(job.callbacks.values())
            job.callbacks.clear()
            current_generation = self._document_generations.get(identity[0], 0)
            is_current = identity == current_identity and generation == current_generation
            if is_current and cache_result and not job.cancel_event.is_set():
                self._store_cache_locked(cache_key, pixbuf)

        if not callbacks:
            return
        if not is_current or job.cancel_event.is_set():
            GLib.idle_add(self._deliver_callbacks, callbacks, None)
            return
        GLib.idle_add(self._deliver_callbacks, callbacks, pixbuf)

    def _finish_cancelled_job(self, job: _ThumbnailJob) -> None:
        """Forget a canceled job without publishing or invoking its subscribers."""
        with self._lock:
            if self._jobs.get(job.cache_key) is job:
                self._jobs.pop(job.cache_key, None)
            job.process = None
            callbacks = list(job.callbacks.values())
            job.callbacks.clear()
        for request, _callback in callbacks:
            request._finish_without_delivery()

    @staticmethod
    def _deliver_callbacks(
        callbacks: list[tuple[ThumbnailRequest, ThumbnailCallback]],
        pixbuf: GdkPixbuf.Pixbuf | None,
    ) -> bool:
        for request, callback in callbacks:
            if not request._claim_delivery():
                continue
            try:
                callback(pixbuf)
            except Exception as error:
                logger.error(f"Thumbnail callback failed: {error}")
        return False

    def _render_pdf_page_pdftoppm(
        self,
        pdf_path: str,
        page_num: int,
        size: int,
        cancel_event: threading.Event,
        job: _ThumbnailJob | None,
    ) -> GdkPixbuf.Pixbuf | None:
        """Render a single PDF page via pdftoppm (native C, ~25x faster)."""
        process: subprocess.Popen[bytes] | None = None
        try:
            page_1based = page_num + 1
            process = subprocess.Popen(
                [
                    "pdftoppm",
                    "-cropbox",
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if job is not None:
                with self._lock:
                    if self._jobs.get(job.cache_key) is job:
                        job.process = process
            deadline = time.monotonic() + 30.0
            while True:
                if cancel_event.is_set():
                    self._terminate_and_reap(process)
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, 30)
                try:
                    stdout, _stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode == 0 and stdout:
                loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
                loader.write(stdout)
                loader.close()
                return loader.get_pixbuf()
        except subprocess.TimeoutExpired as error:
            if process is not None:
                self._terminate_and_reap(process)
            logger.warning(f"pdftoppm single-page timed out: {error}")
        except Exception as e:
            logger.warning(f"pdftoppm single-page failed: {e}")
        finally:
            if job is not None:
                with self._lock:
                    if job.process is process:
                        job.process = None

        # Fallback to Poppler GI
        if cancel_event.is_set():
            return None
        return self._render_pdf_to_pixbuf(pdf_path, page_num, size)

    @staticmethod
    def _request_process_termination(process: subprocess.Popen[bytes]) -> None:
        """Ask a running renderer to stop; its owner thread performs reaping."""
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            return

    @classmethod
    def _terminate_and_reap(cls, process: subprocess.Popen[bytes]) -> None:
        """Terminate and reap a pdftoppm child without leaving a zombie."""
        cls._request_process_termination(process)
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            process.communicate()
        except OSError:
            pass

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
        identity = self._get_document_identity(pdf_path)
        if identity not in self._tls.docs:
            try:
                uri = GLib.filename_to_uri(identity[0], None)
                self._tls.docs = {
                    key: value for key, value in self._tls.docs.items() if key[0] != identity[0]
                }
                self._tls.docs[identity] = Poppler.Document.new_from_file(uri, None)
            except Exception as e:
                logger.error(f"Failed to load PDF in worker: {e}")
                return None
        return self._tls.docs[identity]

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

    def _apply_rotation(self, pixbuf: GdkPixbuf.Pixbuf, rotation: int) -> GdkPixbuf.Pixbuf:
        """Apply rotation to pixbuf."""
        rot = rotation % 360
        if rot == 90:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.CLOCKWISE) or pixbuf
        elif rot == 180:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.UPSIDEDOWN) or pixbuf
        elif rot == 270:
            return pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.COUNTERCLOCKWISE) or pixbuf
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
        pixbuf = Gdk.pixbuf_get_from_texture(texture)
        if pixbuf is None:
            raise RuntimeError("could not create the error thumbnail")
        return pixbuf

    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF."""
        lower_path = pdf_path.lower()
        if lower_path.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")):
            return 1
        doc = self._get_document(pdf_path)
        return doc.get_n_pages() if doc else 0

    def clear_page_cache(self, pdf_path: str, page_num: int) -> None:
        """Evict cached variants for one page without disrupting other renders."""
        canonical_path = self._canonical_path(pdf_path)
        with self._lock:
            cache_keys = [
                key for key in self._cache if key[0][0] == canonical_path and key[2] == page_num
            ]
            for cache_key in cache_keys:
                del self._cache[cache_key]
                self._cache_bytes -= self._cache_costs.pop(cache_key, 0)

    def clear_document_cache(self, pdf_path: str) -> None:
        """Clear cached thumbnails for a document and invalidate thread-local docs."""
        canonical_path = self._canonical_path(pdf_path)
        jobs_to_cancel: list[_ThumbnailJob] = []
        with self._lock:
            self._doc_version += 1
            self._document_generations[canonical_path] = (
                self._document_generations.get(canonical_path, 0) + 1
            )
            cache_keys = [key for key in self._cache if key[0][0] == canonical_path]
            for cache_key in cache_keys:
                del self._cache[cache_key]
                self._cache_bytes -= self._cache_costs.pop(cache_key, 0)
            pending_keys = [key for key in self._jobs if key[0][0] == canonical_path]
            for cache_key in pending_keys:
                job = self._jobs.pop(cache_key)
                job.cancel_event.set()
                for request, _callback in job.callbacks.values():
                    request._finish_without_delivery()
                job.callbacks.clear()
                jobs_to_cancel.append(job)
            self._documents = {
                identity: document
                for identity, document in self._documents.items()
                if identity[0] != canonical_path
            }
        self._cancel_jobs(jobs_to_cancel)

    def clear_all(self) -> None:
        """Clear all caches without blocking the GTK main loop on active workers."""
        jobs_to_cancel: list[_ThumbnailJob] = []
        with self._lock:
            if self._shutdown:
                return
            self._doc_version += 1
            paths = {
                *(cache_key[0][0] for cache_key in self._cache),
                *(cache_key[0][0] for cache_key in self._jobs),
                *(identity[0] for identity in self._documents),
                *self._document_generations,
            }
            for path in paths:
                self._document_generations[path] = self._document_generations.get(path, 0) + 1
            self._cache.clear()
            self._cache_costs.clear()
            self._cache_bytes = 0
            self._documents.clear()
            for job in self._jobs.values():
                job.cancel_event.set()
                for request, _callback in job.callbacks.values():
                    request._finish_without_delivery()
                job.callbacks.clear()
                jobs_to_cancel.append(job)
            self._jobs.clear()
        self._cancel_jobs(jobs_to_cancel)
        old_pool = self._pool
        self._pool = ThreadPoolExecutor(max_workers=4)
        old_pool.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, *, wait: bool = True) -> None:
        """Cancel every job and terminate the non-daemon executor."""
        jobs_to_cancel: list[_ThumbnailJob] = []
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._doc_version += 1
            self._cache.clear()
            self._cache_costs.clear()
            self._cache_bytes = 0
            self._documents.clear()
            self._document_generations.clear()
            for job in self._jobs.values():
                job.cancel_event.set()
                for request, _callback in job.callbacks.values():
                    request._finish_without_delivery()
                job.callbacks.clear()
                jobs_to_cancel.append(job)
            self._jobs.clear()
            pool = self._pool
        self._cancel_jobs(jobs_to_cancel)
        pool.shutdown(wait=wait, cancel_futures=True)

    def _cancel_jobs(self, jobs: list[_ThumbnailJob]) -> None:
        """Cancel queued jobs and interrupt active pdftoppm children."""
        for job in jobs:
            if job.future is not None:
                job.future.cancel()
            if job.process is not None:
                self._request_process_termination(job.process)


# Global instance
_renderer: ThumbnailRenderer | None = None


def get_thumbnail_renderer() -> ThumbnailRenderer:
    global _renderer
    if _renderer is None:
        _renderer = ThumbnailRenderer()
    return _renderer


def shutdown_thumbnail_renderer(*, wait: bool = True) -> None:
    """Release the process-wide renderer and all executor threads."""
    global _renderer
    renderer = _renderer
    _renderer = None
    if renderer is not None:
        renderer.shutdown(wait=wait)
