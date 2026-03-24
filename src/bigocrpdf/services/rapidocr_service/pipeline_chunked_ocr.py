"""Chunked OCR Pipeline Mixin — image extraction, preprocessing, and OCR."""

import gc
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import refine_ocr_results
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class _OCRPool:
    """Pool of persistent OCR subprocesses for parallel inference.

    Each subprocess holds its own model copy and processes one image at a time.
    A semaphore + lock-per-process ensures thread-safe dispatch.
    """

    def __init__(self, procs: list[subprocess.Popen]):
        self._procs = procs
        self._locks = [threading.Lock() for _ in procs]
        self._sem = threading.Semaphore(len(procs))

    def ocr(self, backend: Any, image_path: str) -> dict | None:
        """Send an image to a free OCR subprocess and return results."""
        self._sem.acquire()
        try:
            for i, lock in enumerate(self._locks):
                if lock.acquire(blocking=False):
                    try:
                        return backend._ocr_image_via_subprocess(self._procs[i], image_path)
                    finally:
                        lock.release()
                        self._sem.release()
            # Should not reach here if semaphore count matches procs count
            self._sem.release()
            return None
        except Exception:
            self._sem.release()
            raise

    def stop_all(self, backend: Any) -> None:
        """Gracefully stop all OCR subprocesses."""
        for proc in self._procs:
            backend._stop_ocr_subprocess(proc)


class ChunkedOCRMixin:
    """Mixin providing the chunked OCR pipeline for image-only PDFs."""

    @staticmethod
    def _render_page_for_ocr(pdf_path: Path, page_num: int, dpi: int = 150) -> str | None:
        """Render a single page via pdftoppm for high-quality OCR input.

        Uses uncompressed PPM output (no PNG deflate overhead) which is
        ~5x faster than colour PNG while producing identical OCR results.
        Returns the path to the rendered image, or None on failure.
        """
        fd, prefix = tempfile.mkstemp(suffix="", prefix=f"ocr_render_{page_num}_")
        os.close(fd)
        os.unlink(prefix)  # pdftoppm adds suffix
        cmd = [
            "pdftoppm",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            prefix,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"pdftoppm OCR render failed for page {page_num}: {e.stderr}")
            return None
        rendered = f"{prefix}.ppm"
        if os.path.exists(rendered):
            return rendered
        return None

    def _ocr_chunk_result(
        self, result, ocr_pool, all_rotation_dicts, c, input_pdf=None, masked_pages=None
    ):
        """Run OCR on a single chunk result.

        Accepts either one ``_OCRPool`` (for multi-worker parallel OCR)
        or a single ``subprocess.Popen`` (legacy single-process path).
        Canvas page-size is NOT set here — the caller sets it at flush time
        so this method can be invoked out-of-order.
        """
        page_num = result["page_num"]
        abs_idx = page_num - 1

        # Store page size for later flush instead of setting on canvas now
        if abs_idx < len(all_rotation_dicts):
            mb = all_rotation_dicts[abs_idx]["mediabox"]
            result["_page_size"] = (mb[2] - mb[0], mb[3] - mb[1])
        else:
            result["_page_size"] = (595, 842)

        # Resolve pool vs single proc
        is_pool = isinstance(ocr_pool, _OCRPool)

        def _do_ocr(path: str) -> dict | None:
            if is_pool:
                return ocr_pool.ocr(self, path)
            return self._ocr_image_via_subprocess(ocr_pool, path)

        if result.get("success") and result.get("temp_out_path"):
            ocr_path = result.get("temp_ocr_path") or result["temp_out_path"]

            # For DjVu-like masked pages in overlay mode (skip_geometric),
            # the extracted BG image is too degraded for reliable OCR.
            # Render the composited page via pdftoppm as uncompressed PPM
            # (~0.22s/page) for high-quality OCR input.
            # When geometry was applied (use_rendered_source path), the
            # worker already processed a high-quality PPM render, so the
            # temp_ocr_path from the worker is fine as-is.
            rendered_ocr = None
            if (
                masked_pages
                and page_num in masked_pages
                and input_pdf
                and not result.get("geometry_applied", False)
            ):
                rendered_ocr = self._render_page_for_ocr(input_pdf, page_num)
                if rendered_ocr:
                    ocr_path = rendered_ocr
                    import cv2

                    rimg = cv2.imread(rendered_ocr)
                    if rimg is not None:
                        result["ocr_img_h"], result["ocr_img_w"] = rimg.shape[:2]
                        del rimg

            ocr_raw = _do_ocr(ocr_path)

            if ocr_raw and ocr_raw.get("boxes"):

                def _ocr_crop(path, _fn=_do_ocr):
                    return _fn(path)

                ocr_raw = refine_ocr_results(ocr_raw, ocr_path, _ocr_crop)

            result["ocr_raw"] = ocr_raw

            if rendered_ocr and os.path.exists(rendered_ocr):
                os.unlink(rendered_ocr)

    def _run_chunked_ocr_pipeline(
        self,
        input_pdf: Path,
        text_layer_pdf: Path,
        images_dir: Path,
        ctx: dict[str, Any],
        pipe_cfg: Any,
        res_profile: Any,
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        """Phase 2: Chunked image extraction, preprocessing, OCR, and text rendering."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from reportlab.pdfgen import canvas

        from bigocrpdf.services.rapidocr_service.page_worker import (
            process_page,
            worker_init,
        )

        total_pages = ctx["total_pages"]
        page_rotations = ctx["page_rotations"]
        all_rotation_dicts = ctx["all_rotation_dicts"]
        native_text_pages = ctx["native_text_pages"]
        page_encodings = ctx["page_encodings"]

        from bigocrpdf.services.rapidocr_service.resource_manager import adjust_chunk_size

        page_dims = []
        for rd in all_rotation_dicts:
            mb = rd.get("mediabox")
            if mb and len(mb) >= 4:
                page_dims.append((float(mb[2]) - float(mb[0]), float(mb[3]) - float(mb[1])))
        CHUNK_SIZE = adjust_chunk_size(pipe_cfg.chunk_size, page_dims, res_profile.available_ram_mb)
        max_workers = pipe_cfg.max_workers
        ocr_workers = getattr(pipe_cfg, "ocr_workers", 1)

        logger.info(
            f"Chunked processing: {total_pages} pages in chunks of "
            f"{CHUNK_SIZE}, parallel preprocessing ({max_workers} workers, nice=19), "
            f"{ocr_workers} persistent OCR subprocess(es) "
            f"({pipe_cfg.ocr_threads} threads each, nice=19)"
        )

        c = canvas.Canvas(str(text_layer_pdf))
        page_standalone_flags: list[bool] = []
        page_result_encodings: dict[int, str] = {}
        total_confidence = 0.0
        num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Build set of pages to skip (deleted or excluded from OCR)
        skip_pages: set[int] = set()
        for rot in page_rotations:
            if rot.deleted or not rot.included_for_ocr:
                skip_pages.add(rot.page_number)
        if skip_pages:
            logger.info(f"Skipping {len(skip_pages)} excluded page(s): {sorted(skip_pages)}")

        if progress_callback:
            progress_callback(5, 100, _("Starting OCR..."))

        # Launch OCR subprocess pool
        ocr_procs = [
            self._launch_ocr_subprocess(ocr_threads=pipe_cfg.ocr_threads)
            for _i in range(ocr_workers)
        ]
        ocr_pool = _OCRPool(ocr_procs)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
        ) as executor:
            try:
                for proc in ocr_procs:
                    self._wait_for_ocr_ready(proc)

                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * CHUNK_SIZE
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)

                    # Check if ALL pages in this chunk are excluded — skip entirely
                    chunk_page_nums = set(range(chunk_start + 1, chunk_end + 1))
                    if chunk_page_nums <= skip_pages:
                        # All pages in chunk are excluded; add blank placeholders
                        for page_num in sorted(chunk_page_nums):
                            abs_idx = page_num - 1
                            if abs_idx < len(all_rotation_dicts):
                                mb = all_rotation_dicts[abs_idx]["mediabox"]
                                c.setPageSize((mb[2] - mb[0], mb[3] - mb[1]))
                            else:
                                c.setPageSize((595, 842))
                            c.showPage()
                            page_standalone_flags.append(False)
                            stats.pages_processed += 1
                        logger.info(
                            f"Chunk {chunk_idx + 1}/{num_chunks} skipped "
                            f"(all {len(chunk_page_nums)} pages excluded)"
                        )
                        continue

                    for f in images_dir.glob("*"):
                        try:
                            f.unlink()
                        except OSError:
                            pass

                    chunk_images = self.extractor.extract(
                        input_pdf,
                        output_dir=images_dir,
                        page_range=(chunk_start + 1, chunk_end),
                        skip_pages=skip_pages,
                    )

                    work_items = self._build_chunk_work_items(
                        chunk_images,
                        chunk_start,
                        page_rotations,
                        native_text_pages,
                        page_encodings,
                        pipe_cfg,
                        input_pdf=input_pdf,
                    )

                    # Pipeline: submit all preprocessing jobs, then OCR
                    # each page as soon as its preprocessing finishes.
                    # Canvas requires pages in order, so we buffer
                    # OCR'd results and flush sequentially.
                    preprocess_futures = {}
                    for idx, wi in enumerate(work_items):
                        fut = executor.submit(process_page, wi)
                        preprocess_futures[fut] = idx

                    # Buffer: idx -> (result, work_item)
                    ocr_done: dict[int, tuple[dict, dict]] = {}
                    masked_pages_set = getattr(self.extractor, "masked_pages", None)

                    if ocr_workers > 1:
                        # Multi-worker: dispatch OCR in parallel threads
                        from concurrent.futures import ThreadPoolExecutor

                        ocr_done_lock = threading.Lock()

                        def _ocr_page(
                            idx: int,
                            result: dict,
                            _masked=masked_pages_set,
                            _lock=ocr_done_lock,
                            _done=ocr_done,
                            _items=work_items,
                        ) -> None:
                            self._ocr_chunk_result(
                                result,
                                ocr_pool,
                                all_rotation_dicts,
                                c,
                                input_pdf=input_pdf,
                                masked_pages=_masked,
                            )
                            with _lock:
                                _done[idx] = (result, _items[idx])

                        with ThreadPoolExecutor(max_workers=ocr_workers) as ocr_tp:
                            ocr_futures = []
                            for pp_fut in as_completed(preprocess_futures):
                                idx = preprocess_futures[pp_fut]
                                result = pp_fut.result()

                                if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                                    raise InterruptedError("Processing cancelled by user")

                                ocr_fut = ocr_tp.submit(_ocr_page, idx, result)
                                ocr_futures.append(ocr_fut)

                            for ocr_fut in ocr_futures:
                                ocr_fut.result()
                    else:
                        # Single worker: OCR inline as pages finish preprocessing
                        for pp_fut in as_completed(preprocess_futures):
                            idx = preprocess_futures[pp_fut]
                            result = pp_fut.result()

                            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                                raise InterruptedError("Processing cancelled by user")

                            self._ocr_chunk_result(
                                result,
                                ocr_pool,
                                all_rotation_dicts,
                                c,
                                input_pdf=input_pdf,
                                masked_pages=masked_pages_set,
                            )
                            ocr_done[idx] = (result, work_items[idx])

                    # Flush all results to canvas in page order
                    for flush_idx in range(len(work_items)):
                        res, wi = ocr_done[flush_idx]
                        page_num = res["page_num"]

                        c.setPageSize(res["_page_size"])

                        if progress_callback:
                            pct = 5 + int((page_num / total_pages) * 75)
                            progress_callback(
                                pct,
                                100,
                                _("Processing page {0}/{1}...").format(page_num, total_pages),
                            )

                        try:
                            masked_rendered = wi.get("use_rendered_source", False)
                            confidence, needs_standalone = self._process_page_result(
                                c,
                                res,
                                wi,
                                all_rotation_dicts,
                                page_num,
                                stats,
                                force_overlay=masked_rendered,
                            )
                            total_confidence += confidence
                            page_standalone_flags.append(needs_standalone)
                            enc = res.get("original_encoding", "")
                            if enc:
                                page_result_encodings[page_num] = enc
                        except Exception as page_err:
                            logger.error(f"Error processing page {page_num}: {page_err}")
                            stats.warnings.append(f"Page {page_num} failed: {page_err}")
                            c.setPageSize((595, 842))
                            c.showPage()
                            page_standalone_flags.append(False)

                        del res

                        if pipe_cfg.gc_after_page:
                            gc.collect()

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} done "
                        f"(pages {chunk_start + 1}-{chunk_end})"
                    )

                    if pipe_cfg.gc_after_chunk:
                        gc.collect()

            finally:
                ocr_pool.stop_all(self)

        c.save()
        stats.average_confidence = total_confidence
        self._page_standalone_flags = page_standalone_flags
        self._page_original_encodings = page_result_encodings

    def _build_chunk_work_items(
        self,
        chunk_images: list,
        chunk_start: int,
        page_rotations: list,
        native_text_pages: set[int],
        page_encodings: dict[int, str],
        pipe_cfg: Any,
        input_pdf: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Build work items for parallel preprocessing of a chunk."""
        work_items = []
        masked_pages = getattr(self.extractor, "masked_pages", set())
        # Determine if user wants image modifications that require
        # high-quality source for masked (DjVu-like) pages.
        format_changed = self.config.image_export_format not in ("original", "")
        geometry_enabled = (
            self.config.enable_deskew
            or self.config.enable_perspective_correction
            or self.config.enable_baseline_dewarp
        )
        for i, img_path in enumerate(chunk_images):
            abs_idx = chunk_start + i
            page_num = abs_idx + 1
            rot = page_rotations[abs_idx]

            if page_num in native_text_pages:
                effective_path = None
            elif rot.deleted or not rot.included_for_ocr:
                effective_path = None
            else:
                effective_path = img_path

            # pdftoppm already renders with /Rotate applied, so images
            # from the fallback renderer are display-oriented; tell the
            # worker to skip its own rotation step.
            rendered = page_num in getattr(self.extractor, "rendered_pages", set())

            masked = page_num in masked_pages

            # DjVu-like pages with FG/BG/mask layer separation:
            # - If user wants image modifications (geometry or format change),
            #   render the composited page via pdftoppm and process normally.
            # - Otherwise, skip geometry to preserve the original composite
            #   via overlay mode.
            use_rendered_source = masked and (format_changed or geometry_enabled)

            work_item: dict[str, Any] = {
                "page_num": page_num,
                "img_path": (str(effective_path) if effective_path else None),
                "config": self.config,
                "pdf_rotation": rot.original_pdf_rotation,
                "skip_rotation": rendered,
                "skip_geometric": masked and not use_rendered_source,
                "run_ocr": False,
                "probmap_max_side": pipe_cfg.downscale_probmap,
                "original_encoding": page_encodings.get(page_num, ""),
            }
            if use_rendered_source and input_pdf:
                work_item["use_rendered_source"] = True
                work_item["input_pdf"] = str(input_pdf)
            work_items.append(work_item)
        return work_items
