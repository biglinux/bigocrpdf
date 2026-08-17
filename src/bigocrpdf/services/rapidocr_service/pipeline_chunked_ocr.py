"""Chunked OCR Pipeline Mixin — image extraction, preprocessing, and OCR."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import (
    apply_ocr_box_offset,
    choose_better_ocr_result,
    refine_ocr_results,
    should_retry_page_ocr,
)
from bigocrpdf.services.rapidocr_service.ocr_runtime_diagnostics import (
    record_ocr_runtime_diagnostics,
)
from bigocrpdf.services.rapidocr_service.pdf_page_geometry import render_pdf_page_to_ppm
from bigocrpdf.services.rapidocr_service.pipeline_chunk_files import (
    chunk_skip_pages,
    clean_chunk_images_dir,
    remove_rendered_chunk_ocr,
    store_rendered_ocr_size,
)
from bigocrpdf.services.rapidocr_service.resource_manager import select_render_dpi_for_page
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


def _chunk_result_page_size(abs_idx: int, all_rotation_dicts: list[dict]) -> tuple[float, float]:
    if 0 <= abs_idx < len(all_rotation_dicts) and (
        mb := all_rotation_dicts[abs_idx].get("mediabox")
    ):
        return abs(mb[2] - mb[0]), abs(mb[3] - mb[1])
    return 595, 842


def _chunk_result_render_size(
    abs_idx: int,
    all_rotation_dicts: list[dict],
) -> tuple[float, float]:
    width_pts, height_pts = _chunk_result_page_size(abs_idx, all_rotation_dicts)
    if not 0 <= abs_idx < len(all_rotation_dicts):
        return width_pts, height_pts
    rotation = all_rotation_dicts[abs_idx].get("page_rotation")
    user_unit = float(getattr(rotation, "user_unit", 1.0))
    return width_pts * user_unit, height_pts * user_unit


def _resource_diagnostics(res_profile: Any, pipe_cfg: Any, chunk_size: int) -> dict[str, Any]:
    """The tier and budget the pipeline chose, for the OCR diagnostics.

    Both arguments are typed ``Any`` by this pipeline, so fields are read
    defensively: a diagnostics gap is acceptable, an AttributeError that aborts
    a page of OCR is not.
    """
    tier = getattr(res_profile, "tier", None)
    return {
        "tier": getattr(tier, "name", None),
        "available_ram_mb": getattr(res_profile, "available_ram_mb", None),
        "total_ram_mb": getattr(res_profile, "total_ram_mb", None),
        "cpu_count": getattr(res_profile, "cpu_count", None),
        "max_workers": getattr(pipe_cfg, "max_workers", None),
        "chunk_size_base": getattr(pipe_cfg, "chunk_size", None),
        "chunk_size_adjusted": chunk_size,
        "gc_after_page": getattr(pipe_cfg, "gc_after_page", None),
        "downscale_probmap": getattr(pipe_cfg, "downscale_probmap", None),
    }


def _needs_rendered_chunk_ocr(
    result: dict,
    page_num: int,
    input_pdf: Path | None,
    masked_pages: set[int] | None,
) -> bool:
    return bool(
        masked_pages
        and page_num in masked_pages
        and input_pdf
        and not result.get("geometry_applied", False)
    )


class ChunkedOCRMixin:
    """Mixin providing the chunked OCR pipeline for image-only PDFs."""

    @staticmethod
    def _render_page_for_ocr(
        pdf_path: Path,
        page_num: int,
        dpi: int = 300,
        scratch_dir: Path | str | None = None,
    ) -> str | None:
        """Render a single page via pdftoppm for high-quality OCR input.

        Uses uncompressed PPM output (no PNG deflate overhead) which is
        ~5x faster than colour PNG while producing identical OCR results.
        Returns the path to the rendered image, or None on failure.
        """
        return render_pdf_page_to_ppm(
            pdf_path,
            page_num,
            dpi,
            output_dir=scratch_dir,
        )

    def _ocr_chunk_result(
        self,
        result,
        ocr_proc,
        all_rotation_dicts,
        c,
        input_pdf=None,
        masked_pages=None,
        scratch_dir: Path | str | None = None,
    ):
        """Run OCR on a single chunk result.

        Canvas page-size is NOT set here — the caller sets it at flush time
        so this method can be invoked out-of-order.
        """
        page_num = result["page_num"]
        abs_idx = page_num - 1

        # Store page size for later flush instead of setting on canvas now
        result["_page_size"] = _chunk_result_page_size(abs_idx, all_rotation_dicts)
        result["_render_page_size"] = _chunk_result_render_size(abs_idx, all_rotation_dicts)

        if result.get("success") and result.get("temp_out_path"):
            ocr_path, rendered_ocr = self._chunk_ocr_path(
                result,
                page_num,
                input_pdf,
                masked_pages,
                scratch_dir,
            )
            try:
                result["ocr_raw"] = self._ocr_subprocess.recognize(ocr_proc, ocr_path)
                if result["ocr_raw"] and result["ocr_raw"].get("boxes"):
                    result["ocr_raw"] = self._refine_chunk_ocr(
                        ocr_proc, result["ocr_raw"], ocr_path
                    )
                self._retry_chunk_ocr_if_needed(
                    result,
                    ocr_proc,
                    page_num,
                    input_pdf,
                    scratch_dir,
                )
                self._map_crop_only_ocr_to_original_image(result)
            finally:
                remove_rendered_chunk_ocr(rendered_ocr)

    def _chunk_ocr_path(
        self,
        result: dict,
        page_num: int,
        input_pdf: Path | None,
        masked_pages: set[int] | None,
        scratch_dir: Path | str | None,
    ) -> tuple[str, str | None]:
        ocr_path = result.get("temp_ocr_path") or result["temp_out_path"]
        if not _needs_rendered_chunk_ocr(result, page_num, input_pdf, masked_pages):
            return ocr_path, None
        if input_pdf is None:
            return ocr_path, None

        preferred_dpi = int(getattr(self.config, "fallback_render_dpi", 300))
        width_pts, height_pts = result.get(
            "_render_page_size", result.get("_page_size", (0.0, 0.0))
        )
        render_dpi = select_render_dpi_for_page(
            width_pts,
            height_pts,
            preferred_dpi,
            float(getattr(self.config, "max_render_megapixels", 45)),
        )
        if render_dpi != preferred_dpi:
            logger.info(
                f"Page {page_num}: reducing pdftoppm OCR render DPI {preferred_dpi} -> {render_dpi}"
            )
        rendered_ocr = self._render_page_for_ocr(
            input_pdf,
            page_num,
            render_dpi,
            scratch_dir,
        )
        if not rendered_ocr:
            return ocr_path, None
        store_rendered_ocr_size(result, rendered_ocr)
        return rendered_ocr, rendered_ocr

    def _refine_chunk_ocr(self, ocr_proc, ocr_raw: dict, ocr_path: str) -> dict:
        return refine_ocr_results(
            ocr_raw, ocr_path, lambda path: self._ocr_subprocess.recognize(ocr_proc, path)
        )

    def _retry_chunk_ocr_if_needed(
        self,
        result: dict,
        ocr_proc,
        page_num: int,
        input_pdf: Path | None,
        scratch_dir: Path | str | None,
    ) -> None:
        decision = should_retry_page_ocr(result.get("ocr_raw"))
        if not decision.should_retry or input_pdf is None:
            result["retry_level"] = 0
            return

        retry_path = self._retry_chunk_ocr_path(
            result,
            page_num,
            input_pdf,
            scratch_dir,
        )
        if retry_path is None:
            result["retry_level"] = 0
            return

        try:
            retry_raw = self._ocr_subprocess.recognize(ocr_proc, retry_path)
            if retry_raw and retry_raw.get("boxes"):
                retry_raw = self._refine_chunk_ocr(ocr_proc, retry_raw, retry_path)
            selected = choose_better_ocr_result(result.get("ocr_raw"), retry_raw)
            if selected is retry_raw:
                result["ocr_raw"] = retry_raw
                result["retry_level"] = 1
                store_rendered_ocr_size(result, retry_path)
                logger.info(f"Page {page_num}: OCR retry accepted ({decision.reason})")
            else:
                result["retry_level"] = 0
                logger.info(f"Page {page_num}: OCR retry kept original ({decision.reason})")
        finally:
            remove_rendered_chunk_ocr(retry_path)

    def _retry_chunk_ocr_path(
        self,
        result: dict,
        page_num: int,
        input_pdf: Path,
        scratch_dir: Path | str | None,
    ) -> str | None:
        preferred_dpi = int(getattr(self.config, "retry_render_dpi", 350))
        width_pts, height_pts = result.get(
            "_render_page_size", result.get("_page_size", (0.0, 0.0))
        )
        render_dpi = select_render_dpi_for_page(
            width_pts,
            height_pts,
            preferred_dpi,
            float(getattr(self.config, "max_render_megapixels", 45)),
        )
        if render_dpi != preferred_dpi:
            logger.info(
                f"Page {page_num}: reducing retry render DPI {preferred_dpi} -> {render_dpi}"
            )
        return self._render_page_for_ocr(
            input_pdf,
            page_num,
            render_dpi,
            scratch_dir,
        )

    @staticmethod
    def _map_crop_only_ocr_to_original_image(result: dict) -> None:
        if not result.get("crop_applied") or result.get("geometry_applied"):
            return
        raw_offset = result.get("crop_offset_px")
        offset = (
            (int(raw_offset[0]), int(raw_offset[1]))
            if isinstance(raw_offset, (list, tuple)) and len(raw_offset) == 2
            else (0, 0)
        )
        result["ocr_raw"] = apply_ocr_box_offset(result.get("ocr_raw"), offset)
        original_size = result.get("crop_original_size_px")
        if original_size:
            result["ocr_img_w"], result["ocr_img_h"] = original_size

    def _run_chunked_ocr_pipeline(
        self,
        input_pdf: Path,
        text_layer_pdf: Path,
        images_dir: Path,
        worker_scratch_dir: Path,
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
                rotation = rd.get("page_rotation")
                user_unit = float(getattr(rotation, "user_unit", 1.0))
                page_dims.append(
                    (
                        abs(float(mb[2]) - float(mb[0])) * user_unit,
                        abs(float(mb[3]) - float(mb[1])) * user_unit,
                    )
                )
        CHUNK_SIZE = adjust_chunk_size(pipe_cfg.chunk_size, page_dims, res_profile.available_ram_mb)
        max_workers = pipe_cfg.max_workers

        logger.info(
            f"Chunked processing: {total_pages} pages in chunks of "
            f"{CHUNK_SIZE}, parallel preprocessing ({max_workers} workers, nice=19), "
            "1 persistent OCR subprocess "
            f"({pipe_cfg.ocr_threads} threads each, nice=19)"
        )
        c = canvas.Canvas(str(text_layer_pdf))
        page_standalone_flags: list[bool] = []
        page_result_encodings: dict[int, str] = {}
        total_confidence = 0.0
        num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE

        skip_pages = chunk_skip_pages(page_rotations)
        if skip_pages:
            logger.info(f"Skipping {len(skip_pages)} excluded page(s): {sorted(skip_pages)}")

        if progress_callback:
            progress_callback(5, 100, _("Starting OCR..."))

        ocr_proc = self._ocr_subprocess.launch(ocr_threads=pipe_cfg.ocr_threads)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            # Release the large preprocessing heap before OCR on constrained systems.
            max_tasks_per_child=1 if max_workers == 1 else None,
        ) as executor:
            try:
                worker_runtime = self._ocr_subprocess.wait_until_ready(ocr_proc)
                record_ocr_runtime_diagnostics(
                    stats,
                    self.config,
                    self._check_openvino_available,
                    pipe_cfg.ocr_threads,
                    CHUNK_SIZE,
                    worker_runtime,
                    ocr_workers=pipe_cfg.max_workers,
                    resource=_resource_diagnostics(res_profile, pipe_cfg, CHUNK_SIZE),
                )

                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * CHUNK_SIZE
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)

                    chunk_page_nums = set(range(chunk_start + 1, chunk_end + 1))
                    if self._skip_excluded_chunk(
                        c,
                        chunk_page_nums,
                        skip_pages,
                        all_rotation_dicts,
                        page_standalone_flags,
                        stats,
                    ):
                        logger.info(
                            f"Chunk {chunk_idx + 1}/{num_chunks} skipped "
                            f"(all {len(chunk_page_nums)} pages excluded)"
                        )
                        continue

                    clean_chunk_images_dir(images_dir)

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
                        scratch_dir=worker_scratch_dir,
                    )

                    ocr_done = self._ocr_chunk_work_items(
                        executor,
                        process_page,
                        as_completed,
                        work_items,
                        ocr_proc,
                        all_rotation_dicts,
                        c,
                        input_pdf,
                    )
                    total_confidence += self._flush_chunk_results(
                        c,
                        ocr_done,
                        work_items,
                        all_rotation_dicts,
                        total_pages,
                        stats,
                        page_standalone_flags,
                        page_result_encodings,
                        pipe_cfg,
                        progress_callback,
                    )

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} done "
                        f"(pages {chunk_start + 1}-{chunk_end})"
                    )

                    gc.collect()

            finally:
                self._ocr_subprocess.stop(ocr_proc)

        c.save()
        stats.average_confidence = total_confidence
        self._page_standalone_flags = page_standalone_flags
        self._page_original_encodings = page_result_encodings

    def _skip_excluded_chunk(
        self,
        c,
        chunk_page_nums: set[int],
        skip_pages: set[int],
        all_rotation_dicts: list[dict],
        page_standalone_flags: list[bool],
        stats: ProcessingStats,
    ) -> bool:
        if not chunk_page_nums <= skip_pages:
            return False
        for page_num in sorted(chunk_page_nums):
            c.setPageSize(_chunk_result_page_size(page_num - 1, all_rotation_dicts))
            c.showPage()
            page_standalone_flags.append(False)
            stats.pages_processed += 1
        return True

    def _ocr_chunk_work_items(
        self,
        executor,
        process_page,
        as_completed,
        work_items: list[dict],
        ocr_proc,
        all_rotation_dicts: list[dict],
        c,
        input_pdf: Path,
    ) -> dict[int, tuple[dict, dict]]:
        preprocess_futures = {
            executor.submit(process_page, work_item): idx
            for idx, work_item in enumerate(work_items)
        }
        try:
            return self._ocr_chunk_results_inline(
                preprocess_futures,
                as_completed,
                work_items,
                ocr_proc,
                all_rotation_dicts,
                c,
                input_pdf,
            )
        except BaseException:
            for future in preprocess_futures:
                future.cancel()
            raise

    def _ocr_chunk_results_inline(
        self,
        preprocess_futures: dict,
        as_completed,
        work_items: list[dict],
        ocr_proc,
        all_rotation_dicts: list[dict],
        c,
        input_pdf: Path,
    ) -> dict[int, tuple[dict, dict]]:
        ocr_done: dict[int, tuple[dict, dict]] = {}
        masked_pages_set = getattr(self.extractor, "masked_pages", None)
        for pp_fut in as_completed(preprocess_futures):
            idx = preprocess_futures[pp_fut]
            result = pp_fut.result()
            self._raise_if_cancelled()
            self._ocr_chunk_result(
                result,
                ocr_proc,
                all_rotation_dicts,
                c,
                input_pdf=input_pdf,
                masked_pages=masked_pages_set,
                scratch_dir=work_items[idx].get("scratch_dir"),
            )
            ocr_done[idx] = (result, work_items[idx])
        return ocr_done

    def _flush_chunk_results(
        self,
        c,
        ocr_done: dict[int, tuple[dict, dict]],
        work_items: list[dict],
        all_rotation_dicts: list[dict],
        total_pages: int,
        stats: ProcessingStats,
        page_standalone_flags: list[bool],
        page_result_encodings: dict[int, str],
        pipe_cfg: Any,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> float:
        total_confidence = 0.0
        for flush_idx in range(len(work_items)):
            res, work_item = ocr_done[flush_idx]
            total_confidence += self._flush_chunk_page(
                c,
                res,
                work_item,
                all_rotation_dicts,
                total_pages,
                stats,
                page_standalone_flags,
                page_result_encodings,
                progress_callback,
            )
            if pipe_cfg.gc_after_page:
                gc.collect()
        return total_confidence

    def _flush_chunk_page(
        self,
        c,
        result: dict,
        work_item: dict,
        all_rotation_dicts: list[dict],
        total_pages: int,
        stats: ProcessingStats,
        page_standalone_flags: list[bool],
        page_result_encodings: dict[int, str],
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> float:
        page_num = result["page_num"]
        c.setPageSize(result["_page_size"])
        if progress_callback:
            progress_callback(
                5 + int((page_num / total_pages) * 75),
                100,
                _("Processing page {0}/{1}...").format(page_num, total_pages),
            )
        try:
            confidence, needs_standalone = self._process_page_result(
                c,
                result,
                work_item,
                all_rotation_dicts,
                page_num,
                stats,
                force_overlay=work_item.get("use_rendered_source", False),
            )
            page_standalone_flags.append(needs_standalone)
            if result.get("original_encoding", ""):
                page_result_encodings[page_num] = result["original_encoding"]
            return confidence
        except Exception as page_err:
            logger.error(f"Error processing page {page_num}: {page_err}")
            stats.warnings.append(f"Page {page_num} failed: {page_err}")
            c.setPageSize((595, 842))
            c.showPage()
            page_standalone_flags.append(False)
            return 0.0
        finally:
            del result

    def _raise_if_cancelled(self) -> None:
        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
            raise InterruptedError("Processing cancelled by user")

    def _build_chunk_work_items(
        self,
        chunk_images: list,
        chunk_start: int,
        page_rotations: list,
        native_text_pages: set[int],
        page_encodings: dict[int, str],
        pipe_cfg: Any,
        input_pdf: Path | None = None,
        scratch_dir: Path | None = None,
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
                "input_pdf": str(input_pdf) if input_pdf else None,
                "pdf_rotation": rot.original_pdf_rotation,
                "skip_rotation": rendered,
                "skip_geometric": masked and not use_rendered_source,
                "run_ocr": False,
                "probmap_max_side": pipe_cfg.downscale_probmap,
                "original_encoding": page_encodings.get(page_num, ""),
                "scratch_dir": str(scratch_dir) if scratch_dir else None,
            }
            if use_rendered_source and input_pdf:
                work_item["use_rendered_source"] = True
                work_item["input_pdf"] = str(input_pdf)
            work_items.append(work_item)
        return work_items
