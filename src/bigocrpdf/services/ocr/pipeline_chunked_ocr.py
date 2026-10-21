"""Chunked OCR pipeline mixin: image extraction, preprocessing, and OCR."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import contextlib
import dataclasses
import gc
import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.pdfgen import canvas

from bigocrpdf.services.ocr.config import ProcessingStats
from bigocrpdf.services.ocr.ocr_postprocess import (
    apply_ocr_box_offset,
    choose_better_ocr_result,
    page_retry_reason,
    read_refined,
)
from bigocrpdf.services.ocr.ocr_runtime_diagnostics import (
    record_ocr_runtime_diagnostics,
)
from bigocrpdf.services.ocr.page_worker import process_page, worker_init
from bigocrpdf.services.ocr.pdf_page_geometry import render_page_for_ocr
from bigocrpdf.services.ocr.ppocr_native import ReadError, Worker
from bigocrpdf.services.ocr.resource_manager import (
    PipelineConfig,
    ResourceProfile,
    adjust_chunk_size,
)
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


def _resource_diagnostics(
    res_profile: ResourceProfile, pipe_cfg: PipelineConfig, chunk_size: int
) -> dict[str, Any]:
    """The tier and budget the pipeline chose, for the OCR diagnostics.

    The recogniser count is left to the top-level ``ocr_threads``, which
    holds the count the worker runs rather than the one asked for.
    """
    resource = {
        **dataclasses.asdict(res_profile),
        **dataclasses.asdict(pipe_cfg),
        "tier": res_profile.tier.name,
        "chunk_size_adjusted": chunk_size,
    }
    del resource["ocr_threads"]
    return resource


def _record_ocr_image_size(result: dict, image_path: str) -> None:
    """Note the size of the image OCR read, from its header alone."""
    with contextlib.suppress(OSError), Image.open(image_path) as image:
        result["ocr_img_w"], result["ocr_img_h"] = image.size


class ChunkedOCRMixin:
    """Mixin providing the chunked OCR pipeline for image-only PDFs."""

    def _ocr_chunk_result(
        self,
        result,
        ocr_worker: Worker,
        all_rotation_dicts,
        input_pdf=None,
        masked_pages=None,
        scratch_dir: Path | str | None = None,
    ):
        """Run OCR on a single chunk result.

        The caller sets the canvas page size at flush time, so chunks can be
        processed out of order.
        """
        page_num = result["page_num"]
        abs_idx = page_num - 1

        # Store page size for later flush instead of setting on canvas now
        result["_page_size"] = _chunk_result_page_size(abs_idx, all_rotation_dicts)
        result["_render_page_size"] = _chunk_result_render_size(abs_idx, all_rotation_dicts)
        page_info = all_rotation_dicts[abs_idx] if 0 <= abs_idx < len(all_rotation_dicts) else {}

        if result.get("success") and result.get("temp_out_path"):
            ocr_path, rendered_ocr = self._chunk_ocr_path(
                result,
                page_num,
                input_pdf,
                masked_pages,
                scratch_dir,
            )
            try:
                try:
                    result["ocr_raw"] = read_refined(ocr_worker.recognize, ocr_path)
                except ReadError as exc:
                    result["ocr_raw"] = None
                    result["ocr_error"] = str(exc)
                self._retry_chunk_ocr_if_needed(
                    result,
                    page_info,
                    ocr_worker,
                    page_num,
                    input_pdf,
                    scratch_dir,
                )
                # A fresh render of the page was never cropped.
                if rendered_ocr is None:
                    self._map_crop_only_ocr_to_original_image(result)
            finally:
                if rendered_ocr:
                    Path(rendered_ocr).unlink(missing_ok=True)

    def _chunk_ocr_path(
        self,
        result: dict,
        page_num: int,
        input_pdf: Path | None,
        masked_pages: set[int] | None,
        scratch_dir: Path | str | None,
    ) -> tuple[str, str | None]:
        ocr_path = result.get("temp_ocr_path") or result["temp_out_path"]
        if not (
            masked_pages
            and page_num in masked_pages
            and input_pdf is not None
            and not result.get("geometry_applied")
        ):
            return ocr_path, None

        rendered_ocr = render_page_for_ocr(
            input_pdf,
            page_num,
            self.config.fallback_render_dpi,
            self.config.max_render_megapixels,
            scratch_dir,
            result["_render_page_size"],
        )
        if not rendered_ocr:
            return ocr_path, None
        _record_ocr_image_size(result, rendered_ocr)
        return rendered_ocr, rendered_ocr

    def _retry_chunk_ocr_if_needed(
        self,
        result: dict,
        page_info: dict,
        ocr_worker: Worker,
        page_num: int,
        input_pdf: Path | None,
        scratch_dir: Path | str | None,
    ) -> None:
        result["retry_level"] = 0
        reason = page_retry_reason(result.get("ocr_raw"))
        if reason is None or input_pdf is None or not self._drawn_in_page_frame(result, page_info):
            return

        retry_path = render_page_for_ocr(
            input_pdf,
            page_num,
            self.config.retry_render_dpi,
            self.config.max_render_megapixels,
            scratch_dir,
            result["_render_page_size"],
        )
        if retry_path is None:
            return

        try:
            try:
                retry_raw = read_refined(ocr_worker.recognize, retry_path)
            except ReadError as exc:
                logger.warning(f"Page {page_num}: OCR retry failed: {exc}")
                retry_raw = None
            first_raw = result.get("ocr_raw")
            if retry_raw is not None and (
                first_raw is None or choose_better_ocr_result(first_raw, retry_raw) is retry_raw
            ):
                result["ocr_raw"] = retry_raw
                result.pop("ocr_error", None)
                result["retry_level"] = 1
                _record_ocr_image_size(result, retry_path)
                logger.info(f"Page {page_num}: OCR retry accepted ({reason})")
            else:
                logger.info(f"Page {page_num}: OCR retry kept original ({reason})")
        finally:
            Path(retry_path).unlink(missing_ok=True)

    def _drawn_in_page_frame(self, result: dict, page_info: dict) -> bool:
        """Whether the page's text lands where a plain render of the whole page puts it.

        A retry reads such a render, so its boxes fit only an overlay page
        whose image fills it unturned and uncropped; a page drawn from the
        processed image, or mapped into a smaller image rectangle, has another
        frame.
        """
        use_processed, _ = self._determine_page_mode(result, result["proc_w"], result["proc_h"])
        if use_processed or result.get("crop_applied") or page_info.get("rotation", 0):
            return False
        image_rect = page_info.get("image_rect")
        if image_rect is None:
            return True
        page_width, page_height = result["_page_size"]
        return abs(image_rect[2] - page_width) <= 2 and abs(image_rect[3] - page_height) <= 2

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
        pipe_cfg: PipelineConfig,
        res_profile: ResourceProfile,
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        """Phase 2: Chunked image extraction, preprocessing, OCR, and text rendering."""
        total_pages = ctx["total_pages"]
        page_rotations = ctx["page_rotations"]
        all_rotation_dicts = ctx["all_rotation_dicts"]
        native_text_pages = ctx["native_text_pages"]
        page_encodings = ctx["page_encodings"]

        page_dims = [
            _chunk_result_render_size(abs_idx, all_rotation_dicts)
            for abs_idx in range(len(all_rotation_dicts))
        ]
        CHUNK_SIZE = adjust_chunk_size(pipe_cfg.chunk_size, page_dims, res_profile.available_ram_mb)
        max_workers = pipe_cfg.max_workers

        logger.info(
            f"Chunked processing: {total_pages} pages in chunks of "
            f"{CHUNK_SIZE}, parallel preprocessing ({max_workers} workers, nice=19), "
            "1 persistent OCR subprocess (nice=19)"
        )
        c = canvas.Canvas(str(text_layer_pdf))
        page_standalone_flags: list[bool] = []
        page_result_encodings: dict[int, str] = {}
        total_confidence = 0.0
        num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE

        skip_pages = {
            rotation.page_number
            for rotation in page_rotations
            if rotation.deleted or not rotation.included_for_ocr
        }
        if skip_pages:
            logger.info(f"Skipping {len(skip_pages)} excluded page(s): {sorted(skip_pages)}")

        if progress_callback:
            progress_callback(5, 100, _("Starting OCR..."))

        ocr_worker = self._start_ocr_worker(pipe_cfg.ocr_threads)

        # Forking would copy the parent's GLib threads and its imported Gtk into
        # every worker; forkserver children start from a clean interpreter.
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("forkserver"),
            initializer=worker_init,
            initargs=(logging.getLogger().getEffectiveLevel(),),
            # Release the large preprocessing heap before OCR on constrained systems.
            max_tasks_per_child=1 if max_workers == 1 else None,
        ) as executor:
            try:
                kernel_tier = ocr_worker.wait_ready()
                record_ocr_runtime_diagnostics(
                    stats,
                    self.config,
                    kernel_tier,
                    ocr_worker.threads,
                    CHUNK_SIZE,
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

                    for file_path in images_dir.glob("*"):
                        with contextlib.suppress(OSError):
                            file_path.unlink()

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
                        work_items,
                        ocr_worker,
                        all_rotation_dicts,
                        input_pdf,
                        progress_callback=progress_callback,
                        pages_before=chunk_start,
                        total_pages=total_pages,
                    )
                    total_confidence += self._flush_chunk_results(
                        c,
                        ocr_done,
                        work_items,
                        all_rotation_dicts,
                        stats,
                        page_standalone_flags,
                        page_result_encodings,
                        pipe_cfg,
                    )

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} done "
                        f"(pages {chunk_start + 1}-{chunk_end})"
                    )

                    gc.collect()

            finally:
                ocr_worker.close()

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
        work_items: list[dict],
        ocr_worker: Worker,
        all_rotation_dicts: list[dict],
        input_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
        pages_before: int = 0,
        total_pages: int = 0,
    ) -> dict[int, tuple[dict, dict]]:
        preprocess_futures = {
            executor.submit(process_page, work_item): idx
            for idx, work_item in enumerate(work_items)
        }
        ocr_done: dict[int, tuple[dict, dict]] = {}
        masked_pages_set = getattr(self.extractor, "masked_pages", None)
        try:
            for pp_fut in as_completed(preprocess_futures):
                idx = preprocess_futures[pp_fut]
                result = pp_fut.result()
                self._raise_if_cancelled()
                self._ocr_chunk_result(
                    result,
                    ocr_worker,
                    all_rotation_dicts,
                    input_pdf=input_pdf,
                    masked_pages=masked_pages_set,
                    scratch_dir=work_items[idx].get("scratch_dir"),
                )
                ocr_done[idx] = (result, work_items[idx])
                # Reported as pages are read: a chunk can be the whole document,
                # and writing its pages out afterwards takes a moment.
                if progress_callback:
                    pages_read = pages_before + len(ocr_done)
                    progress_callback(
                        5 + int((pages_read / total_pages) * 75),
                        100,
                        _("Processing page {0}/{1}...").format(pages_read, total_pages),
                    )
        except BaseException:
            for future in preprocess_futures:
                future.cancel()
            raise
        return ocr_done

    def _flush_chunk_results(
        self,
        c,
        ocr_done: dict[int, tuple[dict, dict]],
        work_items: list[dict],
        all_rotation_dicts: list[dict],
        stats: ProcessingStats,
        page_standalone_flags: list[bool],
        page_result_encodings: dict[int, str],
        pipe_cfg: PipelineConfig,
    ) -> float:
        total_confidence = 0.0
        for flush_idx in range(len(work_items)):
            self._raise_if_cancelled()
            res, work_item = ocr_done[flush_idx]
            total_confidence += self._flush_chunk_page(
                c,
                res,
                work_item,
                all_rotation_dicts,
                stats,
                page_standalone_flags,
                page_result_encodings,
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
        stats: ProcessingStats,
        page_standalone_flags: list[bool],
        page_result_encodings: dict[int, str],
    ) -> float:
        page_num = result["page_num"]
        c.setPageSize(result["_page_size"])
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
            stats.failed_pages[page_num] = str(page_err)
            c.showPage()
            page_standalone_flags.append(False)
            return 0.0

    def _build_chunk_work_items(
        self,
        chunk_images: list,
        chunk_start: int,
        page_rotations: list,
        native_text_pages: set[int],
        page_encodings: dict[int, str],
        pipe_cfg: PipelineConfig,
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
                "image_turn": rot.image_turn,
                "skip_rotation": rendered,
                "skip_geometric": masked and not use_rendered_source,
                "probmap_max_side": pipe_cfg.downscale_probmap,
                "original_encoding": page_encodings.get(page_num, ""),
                "scratch_dir": str(scratch_dir) if scratch_dir else None,
            }
            if use_rendered_source and input_pdf:
                work_item["use_rendered_source"] = True
            work_items.append(work_item)
        return work_items
