"""Chunked OCR Pipeline Mixin — image extraction, preprocessing, and OCR."""

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import refine_ocr_results
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ChunkedOCRMixin:
    """Mixin providing the chunked OCR pipeline for image-only PDFs."""

    def _ocr_chunk_result(self, result, ocr_proc, all_rotation_dicts, c):
        """Run OCR on a single chunk result and set page size on canvas."""
        page_num = result["page_num"]
        abs_idx = page_num - 1

        if abs_idx < len(all_rotation_dicts):
            mb = all_rotation_dicts[abs_idx]["mediabox"]
            c.setPageSize((mb[2] - mb[0], mb[3] - mb[1]))
        else:
            c.setPageSize((595, 842))

        if result.get("success") and result.get("temp_out_path"):
            ocr_path = result.get("temp_ocr_path") or result["temp_out_path"]
            ocr_raw = self._ocr_image_via_subprocess(ocr_proc, ocr_path)

            if ocr_raw and ocr_raw.get("boxes"):

                def _ocr_crop(path, _proc=ocr_proc):
                    return self._ocr_image_via_subprocess(_proc, path)

                ocr_raw = refine_ocr_results(ocr_raw, ocr_path, _ocr_crop)

            result["ocr_raw"] = ocr_raw

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
        from concurrent.futures import ProcessPoolExecutor

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

        logger.info(
            f"Chunked processing: {total_pages} pages in chunks of "
            f"{CHUNK_SIZE}, parallel preprocessing ({max_workers} workers, nice=19), "
            f"1 persistent OCR subprocess (nice=19)"
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

        ocr_proc = self._launch_ocr_subprocess(ocr_threads=pipe_cfg.ocr_threads)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
        ) as executor:
            try:
                self._wait_for_ocr_ready(ocr_proc)

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
                    )

                    chunk_results = list(executor.map(process_page, work_items))

                    for i, result in enumerate(chunk_results):
                        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                            raise InterruptedError("Processing cancelled by user")

                        self._ocr_chunk_result(result, ocr_proc, all_rotation_dicts, c)
                        page_num = result["page_num"]

                        if progress_callback:
                            pct = 5 + int((page_num / total_pages) * 75)
                            progress_callback(
                                pct,
                                100,
                                _("Processing page {0}/{1}...").format(page_num, total_pages),
                            )

                        try:
                            confidence, needs_standalone = self._process_page_result(
                                c,
                                result,
                                work_items[i],
                                all_rotation_dicts,
                                page_num,
                                stats,
                            )
                            total_confidence += confidence
                            page_standalone_flags.append(needs_standalone)
                            enc = result.get("original_encoding", "")
                            if enc:
                                page_result_encodings[page_num] = enc
                        except Exception as page_err:
                            logger.error(f"Error processing page {page_num}: {page_err}")
                            stats.warnings.append(f"Page {page_num} failed: {page_err}")
                            c.setPageSize((595, 842))
                            c.showPage()
                            page_standalone_flags.append(False)

                        del result

                        if pipe_cfg.gc_after_page:
                            gc.collect()

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} done "
                        f"(pages {chunk_start + 1}-{chunk_end})"
                    )

                    if pipe_cfg.gc_after_chunk:
                        gc.collect()

            finally:
                self._stop_ocr_subprocess(ocr_proc)

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
    ) -> list[dict[str, Any]]:
        """Build work items for parallel preprocessing of a chunk."""
        work_items = []
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

            work_items.append({
                "page_num": page_num,
                "img_path": (str(effective_path) if effective_path else None),
                "config": self.config,
                "pdf_rotation": rot.original_pdf_rotation,
                "run_ocr": False,
                "probmap_max_side": pipe_cfg.downscale_probmap,
                "original_encoding": page_encodings.get(page_num, ""),
            })
        return work_items
