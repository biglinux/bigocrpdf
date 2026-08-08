"""PDF Processing Pipeline Mixin for ProfessionalPDFOCR."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pikepdf

from bigocrpdf.services.rapidocr_service.backend_embedded_image_pipeline import (
    BackendEmbeddedImagePipelineMixin,
)
from bigocrpdf.services.rapidocr_service.backend_text_layer_geometry import (
    _extract_image_rects_from_pdf,
)
from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_assembly import smart_merge_pdfs
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    get_page_image_encodings,
    get_pages_with_native_text,
)
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import MixedContentMixin
from bigocrpdf.services.rapidocr_service.rotation import (
    apply_editor_modifications as apply_editor_mods_to_rotations,
)
from bigocrpdf.services.rapidocr_service.rotation import (
    apply_final_rotation_to_pdf,
    extract_page_rotations,
)
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger


def _reserve_output_pdf(output_dir: Path, role: str) -> Path:
    """Reserve a private, unpredictable PDF path on the output filesystem."""
    descriptor, path = tempfile.mkstemp(
        prefix=f".bigocr_{role}_",
        suffix=".pdf",
        dir=output_dir,
    )
    os.close(descriptor)
    return Path(path)


class BackendPipelineMixin(BackendEmbeddedImagePipelineMixin, ChunkedOCRMixin, MixedContentMixin):
    """Mixin providing PDF processing pipeline methods."""

    def _process_image_only_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process a PDF where pages are just images (scanned).

        Architecture (memory-optimized):
        - ThreadPoolExecutor for image preprocessing (no fork = no memory duplication)
        - Single persistent OCR subprocess (one model instance = ~400 MB)
        - Chunked extraction to minimize disk usage

        Total memory: ~600-700 MB vs ~4-10 GB with the old fork+subprocess approach.
        """
        from bigocrpdf.services.rapidocr_service.resource_manager import (
            compute_pipeline_config,
            detect_resources,
        )

        res_profile = detect_resources()
        pipe_cfg = compute_pipeline_config(res_profile)

        logger.info(f"Processing image-only PDF: {input_pdf}")

        try:
            os.nice(19)
            logger.debug("Main process priority set to nice=19")
        except OSError:
            pass

        stats = ProcessingStats()
        start_time = time.time()

        output_dir = output_pdf.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        images_temp_dir = tempfile.TemporaryDirectory(prefix="rapidocr_imgs_")
        merged_pdf: Path | None = None
        text_layer_pdf: Path | None = None

        try:
            images_dir = Path(images_temp_dir.name) / "chunk_imgs"
            images_dir.mkdir()
            worker_scratch_dir = Path(images_temp_dir.name) / "page_workers"
            worker_scratch_dir.mkdir()

            merged_pdf = _reserve_output_pdf(output_dir, "processing")
            text_layer_pdf = _reserve_output_pdf(output_dir, "textlayer")

            # Phase 1: Analyze PDF metadata
            ctx = self._analyze_pdf_metadata(input_pdf, stats, pipe_cfg, progress_callback)
            if ctx["total_pages"] == 0:
                return stats

            # Phase 2: Chunked extraction + preprocessing + OCR
            self._run_chunked_ocr_pipeline(
                input_pdf,
                text_layer_pdf,
                images_dir,
                worker_scratch_dir,
                ctx,
                pipe_cfg,
                res_profile,
                stats,
                progress_callback,
            )

            # Phase 3: Merge, post-process, finalize
            self._post_process_pdf(
                input_pdf, output_pdf, merged_pdf, text_layer_pdf, ctx, stats, progress_callback
            )

        except InterruptedError:
            logger.info("Processing cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Error in image-only PDF processing: {e}")
            stats.error = str(e)
            raise
        finally:
            try:
                images_temp_dir.cleanup()
            except OSError as error:
                logger.warning("Could not fully clean OCR scratch directory: %s", error)
            for temp_pdf in (merged_pdf, text_layer_pdf):
                try:
                    if temp_pdf is not None and temp_pdf.exists():
                        temp_pdf.unlink()
                except OSError:
                    pass

        self._calculate_final_stats(stats, start_time)

        if progress_callback:
            progress_callback(100, 100, _("Done!"))

        logger.info(f"Processing complete in {stats.processing_time_seconds:.1f}s")
        logger.info(
            f"Pages: {stats.pages_processed}, "
            f"Text regions: {stats.total_text_regions}, "
            f"Avg confidence: {stats.average_confidence:.2%}"
        )

        return stats

    def _analyze_pdf_metadata(
        self,
        input_pdf: Path,
        stats: ProcessingStats,
        pipe_cfg: Any,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> dict[str, Any]:
        """Phase 1: Extract rotations, detect native text, and image encodings."""
        if progress_callback:
            progress_callback(0, 100, _("Analyzing PDF..."))

        page_rotations = extract_page_rotations(input_pdf)
        if self.config.page_modifications:
            page_rotations = apply_editor_mods_to_rotations(
                page_rotations, self.config.page_modifications
            )

        total_pages = len(page_rotations)
        stats.pages_total = total_pages

        image_rects = _extract_image_rects_from_pdf(input_pdf, page_count=total_pages)
        all_rotation_dicts = []
        for i, rot in enumerate(page_rotations):
            all_rotation_dicts.append(
                {
                    "rotation": rot.original_pdf_rotation,
                    "mediabox": rot.mediabox,
                    "page_rotation": rot,
                    "image_rect": image_rects[i],
                }
            )
        native_text_pages: set[int] = set()
        if self.config.force_full_ocr:
            native_text_pages = get_pages_with_native_text(input_pdf, total_pages)
            if native_text_pages:
                logger.info(
                    f"PDF has {len(native_text_pages)} page(s) with native text "
                    f"that will be preserved: {sorted(native_text_pages)}"
                )

        page_encodings = get_page_image_encodings(input_pdf)
        bilevel_encs = {p for p, e in page_encodings.items() if e in ("jbig2", "ccitt")}
        if bilevel_encs:
            logger.info(
                f"Detected bilevel encoding on {len(bilevel_encs)} page(s): {sorted(bilevel_encs)}"
            )

        return {
            "total_pages": total_pages,
            "page_rotations": page_rotations,
            "all_rotation_dicts": all_rotation_dicts,
            "native_text_pages": native_text_pages,
            "page_encodings": page_encodings,
        }

    def _post_process_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        merged_pdf: Path,
        text_layer_pdf: Path,
        ctx: dict[str, Any],
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        """Phase 3: Merge text layer, apply editor mods, bilevel optimize, finalize."""
        page_rotations = ctx["page_rotations"]
        native_text_pages = ctx["native_text_pages"]
        page_standalone_flags = self._page_standalone_flags
        page_result_encodings = self._page_original_encodings

        # Merge text layer with original
        if progress_callback:
            progress_callback(85, 100, _("Merging text layer..."))

        self._merge_text_layer_pdf(input_pdf, text_layer_pdf, merged_pdf, page_standalone_flags)

        # OCR images within native text pages (editor-merged files)
        self._process_native_text_pages(merged_pdf, native_text_pages, stats, progress_callback)

        # Apply editor modifications
        start_page = 1
        if self.config.page_range:
            start_page = self.config.page_range[0]

        if self.config.page_modifications:
            apply_final_rotation_to_pdf(merged_pdf, page_rotations, start_page)

        # Optimize bilevel images (JBIG2/CCITT re-encoding)
        self._optimize_bilevel_pages(merged_pdf, page_result_encodings, stats, progress_callback)

        split_parts = self._finalize_output(merged_pdf, output_pdf, progress_callback)
        if split_parts:
            stats.split_output_files = [str(p) for p in split_parts]

    def _merge_text_layer_pdf(
        self,
        input_pdf: Path,
        text_layer_pdf: Path,
        merged_pdf: Path,
        page_standalone_flags: list[bool],
    ) -> None:
        any_standalone = any(page_standalone_flags) if page_standalone_flags else False
        all_standalone = all(page_standalone_flags) if page_standalone_flags else False
        if self.config.image_export_format not in ("original", "") or all_standalone:
            import shutil

            shutil.copy2(text_layer_pdf, merged_pdf)
        elif any_standalone:
            smart_merge_pdfs(input_pdf, text_layer_pdf, merged_pdf, page_standalone_flags)
        else:
            self._overlay_text_on_original(input_pdf, text_layer_pdf, merged_pdf)

    def _process_native_text_pages(
        self,
        merged_pdf: Path,
        native_text_pages: set[int],
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        if not native_text_pages or not merged_pdf.exists():
            return
        if progress_callback:
            progress_callback(87, 100, _("Processing text page images..."))
        self._ocr_native_text_page_images(merged_pdf, native_text_pages, stats, progress_callback)

        native_text = self._extract_native_text(merged_pdf).strip()
        if not native_text:
            return
        stats.full_text = native_text + "\n\n" + stats.full_text if stats.full_text else native_text

    def _optimize_bilevel_pages(
        self,
        merged_pdf: Path,
        page_result_encodings: dict,
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        if not self.config.enable_bilevel_compression or not page_result_encodings:
            return
        if progress_callback:
            progress_callback(88, 100, _("Optimizing image compression..."))
        from bigocrpdf.services.rapidocr_service.bilevel_optimizer import (
            optimize_bilevel_images,
        )

        n_opt = optimize_bilevel_images(
            merged_pdf,
            page_result_encodings,
            force_bilevel=self.config.force_bilevel_compression,
        )
        if n_opt:
            stats.warnings.append(
                ngettext(
                    "{count} page re-encoded with JBIG2",
                    "{count} pages re-encoded with JBIG2",
                    n_opt,
                ).format(count=n_opt)
            )

    def _finalize_output(
        self,
        merged_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[Path]:
        """Finalize output PDF, optionally converting to PDF/A and splitting by size.

        Uses shutil.move when source and destination are on the same
        filesystem for near-instant transfer (rename), avoiding
        large temporary copies in /tmp.

        Args:
            merged_pdf: Path to merged PDF
            output_pdf: Path for final output PDF
            progress_callback: Optional progress callback

        Returns:
            List of split part paths if splitting occurred, empty list otherwise
        """
        import shutil

        if self.config.convert_to_pdfa:
            if progress_callback:
                progress_callback(90, 100, _("Converting to PDF/A..."))
            # PDF/A path sets /PageLayout inside convert_to_pdfa's careful save.
            self._ocr.convert_to_pdfa(merged_pdf, output_pdf)
        elif getattr(self.config, "page_layout", "default") != "default":
            # Non-PDF/A: write /PageLayout into the catalog. Only re-save via
            # pikepdf when a non-default layout was requested; the common
            # default path keeps the instant rename below.
            import pikepdf

            from bigocrpdf.utils.pdf_utils import set_root_page_layout

            with pikepdf.open(merged_pdf) as pdf:
                set_root_page_layout(pdf, self.config.page_layout)
                pdf.save(output_pdf)
            merged_pdf.unlink(missing_ok=True)
        else:
            # Use move instead of copy — instant rename on same filesystem
            shutil.move(str(merged_pdf), str(output_pdf))

        # Split by file size if limit is configured
        max_mb = self.config.max_file_size_mb
        if max_mb > 0:
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                if progress_callback:
                    progress_callback(92, 100, _("Splitting PDF by size limit..."))
                return self._split_pdf_by_size(output_pdf, max_mb)

        return []

    def _split_pdf_by_size(self, output_pdf: Path, max_mb: int) -> list[Path]:
        """Split a PDF into numbered parts under the configured size limit."""
        import io

        max_bytes = max_mb * 1024 * 1024

        with pikepdf.open(output_pdf) as source_pdf:
            total_pages = len(source_pdf.pages)

            if total_pages <= 1:
                logger.info(
                    f"PDF has only 1 page ({output_pdf.stat().st_size / 1024 / 1024:.1f} MB), "
                    f"cannot split further"
                )
                return []

            stem = output_pdf.stem
            suffix = output_pdf.suffix
            parent = output_pdf.parent
            parts: list[Path] = []
            current_pages: list[int] = []

            # Pre-compute per-page size estimates (O(n) total)
            page_sizes: list[int] = []
            for i in range(total_pages):
                single = pikepdf.new()
                single.pages.append(source_pdf.pages[i])
                buf = io.BytesIO()
                single.save(buf)
                page_sizes.append(buf.tell())
                single.close()

            for page_idx in range(total_pages):
                current_pages.append(page_idx)

                # Estimate size from pre-computed per-page sizes
                est_size = sum(page_sizes[idx] for idx in current_pages)

                if est_size > max_bytes and len(current_pages) > 1:
                    # Remove the page that pushed us over the limit
                    current_pages.pop()

                    # Save this part
                    part_num = len(parts) + 1
                    part_path = parent / f"{stem}-{part_num:02d}{suffix}"
                    part_pdf = pikepdf.new()
                    for idx in current_pages:
                        part_pdf.pages.append(source_pdf.pages[idx])
                    part_pdf.save(str(part_path))
                    part_pdf.close()
                    parts.append(part_path)

                    logger.info(
                        f"Split part {part_num}: {len(current_pages)} pages, "
                        f"{part_path.stat().st_size / 1024 / 1024:.1f} MB"
                    )

                    # Start new part with the page that didn't fit
                    current_pages = [page_idx]

            # Save the last part
            if current_pages:
                part_num = len(parts) + 1
                part_path = parent / f"{stem}-{part_num:02d}{suffix}"
                part_pdf = pikepdf.new()
                for idx in current_pages:
                    part_pdf.pages.append(source_pdf.pages[idx])
                part_pdf.save(str(part_path))
                part_pdf.close()
                parts.append(part_path)

                logger.info(
                    f"Split part {part_num}: {len(current_pages)} pages, "
                    f"{part_path.stat().st_size / 1024 / 1024:.1f} MB"
                )

        # Remove the original oversized file
        if parts:
            output_pdf.unlink()
            logger.info(
                f"PDF split into {len(parts)} parts "
                f"(max {max_mb} MB each): {[p.name for p in parts]}"
            )
            return parts

        return []

    def _calculate_final_stats(self, stats: ProcessingStats, start_time: float) -> None:
        """Calculate final processing statistics.

        Args:
            stats: Stats object to update
            start_time: Processing start time
        """
        stats.processing_time_seconds = time.time() - start_time
        if stats.total_text_regions > 0:
            stats.average_confidence = stats.average_confidence / stats.total_text_regions
