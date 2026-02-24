"""PDF Processing Pipeline Mixin for ProfessionalPDFOCR."""

import gc
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pikepdf

from bigocrpdf.services.rapidocr_service.config import OCRBoxData, OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import refine_ocr_results
from bigocrpdf.services.rapidocr_service.pdf_assembly import (
    smart_merge_pdfs,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_image_positions,
    get_pages_with_native_text,
    load_image_with_exif_rotation,
)
from bigocrpdf.services.rapidocr_service.rotation import (
    PageRotation,
    apply_final_rotation_to_pdf,
    extract_page_rotations,
)
from bigocrpdf.services.rapidocr_service.rotation import (
    apply_editor_modifications as apply_editor_mods_to_rotations,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class BackendPipelineMixin:
    """Mixin providing PDF processing pipeline methods."""

    def _build_ocr_subprocess_cmd(self, ocr_threads: int = 0) -> list[str]:
        """Build command-line args for persistent OCR subprocess.

        Args:
            ocr_threads: Number of inference threads. 0 = auto (cpu_count / 2).

        Returns:
            Command list ready for subprocess.Popen.
        """
        import multiprocessing

        worker_script = str(Path(__file__).parent / "ocr_worker.py")
        cpu_count = multiprocessing.cpu_count()

        if ocr_threads <= 0:
            ocr_threads = max(2, cpu_count // 2)

        cmd = [
            "python3",
            worker_script,
            "--persistent",
            "--language",
            self.config.language,
            "--limit_side_len",
            str(self.config.detection_limit_side_len),
            "--box-thresh",
            str(self.config.box_thresh),
            "--unclip-ratio",
            str(self.config.unclip_ratio),
            "--text-score",
            str(self.config.text_score_threshold),
            "--score-mode",
            self.config.score_mode,
            "--threads",
            str(ocr_threads),
        ]

        try:
            if not self._check_openvino_available():
                cmd.append("--no-openvino")
        except Exception:
            cmd.append("--no-openvino")

        for flag, getter in [
            ("--rec-model-path", self.config.get_rec_model_path),
            ("--rec-keys-path", self.config.get_rec_keys_path),
            ("--det-model-path", self.config.get_det_model_path),
            ("--font-path", self.config.get_font_path),
        ]:
            path = getter()
            if path:
                cmd.extend([flag, str(path)])

        return cmd

    @staticmethod
    def _low_priority_preexec():
        """Set low CPU and I/O priority for subprocess (desktop-friendly)."""
        try:
            os.nice(19)
        except OSError:
            pass
        try:
            # ionice idle class (class 3)
            import subprocess as _sp

            _sp.run(["ionice", "-c", "3", "-p", str(os.getpid())], capture_output=True, timeout=2)
        except Exception:
            pass

    def _launch_ocr_subprocess(self, ocr_threads: int = 0) -> subprocess.Popen:
        """Start the OCR subprocess without waiting for it to be ready.

        This allows the model to load in the background while we do other tasks
        (like image extraction/preprocessing).
        The subprocess runs with nice=19 and ionice idle for desktop responsiveness.

        Args:
            ocr_threads: Number of inference threads. 0 = auto.
        """
        cmd = self._build_ocr_subprocess_cmd(ocr_threads=ocr_threads)
        logger.debug(f"Launching OCR subprocess (background): {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit stderr
            text=True,
            bufsize=1,
            preexec_fn=self._low_priority_preexec,
        )
        return proc

    def _wait_for_ocr_ready(self, proc: subprocess.Popen) -> None:
        """Wait for the OCR subprocess to signal it is ready (model loaded)."""
        if not proc:
            return

        logger.debug("Waiting for OCR subprocess ready signal...")
        assert proc.stdout is not None
        ready_line = proc.stdout.readline()

        if not ready_line:
            raise RuntimeError("OCR subprocess failed to start (check stderr)")

        try:
            ready = json.loads(ready_line.strip())
            if ready.get("fatal"):
                raise RuntimeError(f"OCR subprocess fatal error: {ready['fatal']}")
        except json.JSONDecodeError:
            logger.warning(f"Unexpected OCR subprocess output: {ready_line.strip()}")

        logger.info("OCR subprocess ready (model loaded)")

    def _ocr_image_via_subprocess(self, proc: subprocess.Popen, image_path: str) -> dict | None:
        """Send an image to the persistent OCR subprocess and get results.

        Args:
            proc: Running OCR subprocess with stdin/stdout pipes
            image_path: Path to preprocessed image file

        Returns:
            Dict with boxes/txts/scores, or None on failure.
        """
        if not proc or proc.poll() is not None:
            logger.error("OCR subprocess not running")
            return None

        try:
            # Send image path
            # logger.debug(f"Sending path to subprocess: {image_path}")
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(f"{image_path}\n")
            proc.stdin.flush()

            # Wait for response
            line = proc.stdout.readline()

            if not line:
                logger.error("OCR subprocess stdout closed")
                return None

            result = json.loads(line.strip())
            if result.get("error"):
                logger.error(f"OCR error for {image_path}: {result['error']}")
                return None

            return result

        except (BrokenPipeError, OSError) as e:
            logger.error(f"OCR subprocess pipe error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"OCR subprocess invalid JSON: {e}")
            return None

    def _stop_ocr_subprocess(self, proc: subprocess.Popen) -> None:
        """Gracefully stop the persistent OCR subprocess."""
        if proc is None:
            return

        logger.debug(f"Stopping OCR subprocess PID {proc.pid}...")
        try:
            if proc.stdin:
                proc.stdin.close()
            logger.debug("Waiting for subprocess to exit...")
            proc.wait(timeout=10)
            logger.debug("Subprocess exited gracefully")
        except Exception as e:
            logger.debug(f"Error stopping subprocess: {e}, killing...")
            proc.kill()
            proc.wait(timeout=5)
            logger.debug("Subprocess killed")

    def _prepare_pages_for_processing(
        self,
        extracted_images: list[Path | None],
        page_rotations: list,
    ) -> tuple[list[Path | None], list[dict]]:
        """Filter extracted images by deletion flags and build rotation dicts.

        Returns:
            Tuple of (valid_images, valid_rotation_dicts)
        """
        start_page = 1
        if self.config.page_range:
            start_page = self.config.page_range[0]

        valid_images: list[Path | None] = []
        valid_rotations: list[dict[str, Any]] = []

        for i, img_path in enumerate(extracted_images):
            page_num = start_page + i
            original_idx = page_num - 1

            if 0 <= original_idx < len(page_rotations):
                rot = page_rotations[original_idx]
            else:
                rot = PageRotation(page_number=page_num)
                logger.warning(f"Page {page_num}: No rotation info found")

            if rot.deleted:
                logger.info(f"Skipping deleted page {page_num} (adding placeholder)")
                valid_images.append(None)
            elif img_path is None:
                logger.warning(f"No image found for page {page_num}, using placeholder")
                valid_images.append(None)
            else:
                valid_images.append(img_path)

            valid_rotations.append({
                "rotation": rot.original_pdf_rotation,
                "mediabox": rot.mediabox,
                "page_rotation": rot,
            })

        return valid_images, valid_rotations

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

        # Detect available resources and compute adaptive configuration
        res_profile = detect_resources()
        pipe_cfg = compute_pipeline_config(res_profile)

        CHUNK_SIZE = pipe_cfg.chunk_size

        logger.info(f"Processing image-only PDF: {input_pdf}")

        # Set low priority for the main process (desktop responsiveness)
        try:
            os.nice(19)
            logger.debug("Main process priority set to nice=19")
        except OSError:
            pass  # Already at max niceness or no permission

        stats = ProcessingStats()
        start_time = time.time()

        output_dir = output_pdf.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Images go to /tmp (RAM on Linux) for faster I/O
        images_temp_dir = tempfile.TemporaryDirectory(prefix="rapidocr_imgs_")
        images_dir = Path(images_temp_dir.name) / "chunk_imgs"
        images_dir.mkdir()

        # Temporary PDFs stay in the output directory (same filesystem = fast rename)
        # Use descriptive names so the user knows what they are
        merged_pdf = output_dir / f".{output_pdf.stem}_processing.pdf"
        text_layer_pdf = output_dir / f".{output_pdf.stem}_textlayer.pdf"

        try:
            if progress_callback:
                progress_callback(0, 100, _("Analyzing PDF..."))

            # 1. Metadata pass (fast, no images extracted)
            page_rotations = extract_page_rotations(input_pdf)
            if self.config.page_modifications:
                page_rotations = apply_editor_mods_to_rotations(
                    page_rotations, self.config.page_modifications
                )

            total_pages = len(page_rotations)
            stats.pages_total = total_pages

            if total_pages == 0:
                logger.warning("No pages found in PDF!")
                return stats

            # Build rotation dicts for all pages (metadata only)
            all_rotation_dicts = []
            for rot in page_rotations:
                all_rotation_dicts.append({
                    "rotation": rot.original_pdf_rotation,
                    "mediabox": rot.mediabox,
                    "page_rotation": rot,
                })

            # Detect pages with native text so they can be preserved as-is
            # (skip image extraction + OCR). Without this, native text pages
            # would have their background images extracted and used as
            # standalone pages, losing all text content.
            # Only for force_full_ocr (editor-merged files). For
            # replace_existing_ocr, all pages go through the normal
            # pipeline — old invisible text is stripped during merge by
            # _strip_invisible_text().
            native_text_pages: set[int] = set()
            if self.config.force_full_ocr:
                native_text_pages = get_pages_with_native_text(input_pdf, total_pages)
                if native_text_pages:
                    logger.info(
                        f"PDF has {len(native_text_pages)} page(s) with native text "
                        f"that will be preserved: {sorted(native_text_pages)}"
                    )

            # 2. Setup persistent OCR subprocess (start launching in background)
            from concurrent.futures import ProcessPoolExecutor

            from reportlab.pdfgen import canvas

            from bigocrpdf.services.rapidocr_service.page_worker import (
                process_page,
                worker_init,
            )

            # Launch persistent OCR subprocess (stays alive for the entire PDF)
            # This avoids ~2s model-loading penalty per chunk
            ocr_proc = self._launch_ocr_subprocess(ocr_threads=pipe_cfg.ocr_threads)

            # Use ProcessPoolExecutor for CPU-bound preprocessing (rotation, deskew, etc)
            # Worker count and chunk size are determined by resource_manager
            # based on available RAM and CPU cores.
            max_workers = pipe_cfg.max_workers

            logger.info(
                f"Chunked processing: {total_pages} pages in chunks of "
                f"{CHUNK_SIZE}, parallel preprocessing ({max_workers} workers, nice=19), "
                f"1 persistent OCR subprocess (nice=19)"
            )

            # 3. Chunked extraction + preprocessing + OCR loop
            c = canvas.Canvas(str(text_layer_pdf))
            page_standalone_flags: list[bool] = []
            total_confidence = 0.0
            num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE

            if progress_callback:
                progress_callback(5, 100, _("Starting OCR..."))

            # Create executor once for the whole process
            # Workers run with low priority (nice=19 + ionice idle)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=worker_init,
            ) as executor:
                try:
                    # Wait for OCR model to finish loading before first use
                    self._wait_for_ocr_ready(ocr_proc)

                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * CHUNK_SIZE  # 0-based
                        chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
                        chunk_page_range = (chunk_start + 1, chunk_end)  # 1-based

                        # Clean images dir for this chunk
                        for f in images_dir.glob("*"):
                            try:
                                f.unlink()
                            except OSError:
                                pass

                        # Extract images directly using pdfimages -all (no re-encoding)
                        chunk_images = self.extractor.extract(
                            input_pdf,
                            output_dir=images_dir,
                            page_range=chunk_page_range,
                        )

                        # Prepare work items for parallel preprocessing
                        work_items = []
                        for i, img_path in enumerate(chunk_images):
                            abs_idx = chunk_start + i
                            page_num = abs_idx + 1
                            rot = page_rotations[abs_idx]

                            # Skip native text pages — their images will be
                            # OCR'd in a separate pass that processes ALL images
                            # per page (not just the first one).
                            if page_num in native_text_pages:
                                effective_path = None
                            elif rot.deleted:
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
                            })

                        # Step 1: Preprocess images in PARALLEL (workers at nice=19)
                        chunk_results = list(executor.map(process_page, work_items))

                        # Step 3: Sequential OCR (subprocess) and Rendering (canvas not thread-safe)
                        for i, result in enumerate(chunk_results):
                            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                                raise InterruptedError("Processing cancelled by user")

                            page_num = result["page_num"]
                            # logger.debug(f"OCR and Rendering page {page_num}")
                            abs_idx = page_num - 1  # 0-based for rotation dicts

                            # Get page dimensions for canvas (must be done sequentially for canvas)
                            if abs_idx < len(all_rotation_dicts):
                                mb = all_rotation_dicts[abs_idx]["mediabox"]
                                w_pt = mb[2] - mb[0]
                                h_pt = mb[3] - mb[1]
                                c.setPageSize((w_pt, h_pt))
                            else:
                                c.setPageSize((595, 842))

                            # Step 2: OCR via persistent subprocess
                            if result.get("success") and result.get("temp_out_path"):
                                ocr_raw = self._ocr_image_via_subprocess(
                                    ocr_proc, result["temp_out_path"]
                                )

                                # Step 2.5: Refine low-confidence oversized detections.
                                # Re-OCR cropped regions where the detector merged
                                # multiple text lines into one box (common on
                                # photographed pages with spine curvature).
                                if ocr_raw and ocr_raw.get("boxes"):

                                    def _ocr_crop(path, _proc=ocr_proc):
                                        return self._ocr_image_via_subprocess(_proc, path)

                                    ocr_raw = refine_ocr_results(
                                        ocr_raw,
                                        result["temp_out_path"],
                                        _ocr_crop,
                                    )

                                result["ocr_raw"] = ocr_raw

                            # Step 3: Render text layer
                            progress_pct = 5 + int((page_num / total_pages) * 75)
                            if progress_callback:
                                progress_callback(
                                    progress_pct,
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
                            except Exception as page_err:
                                logger.error(f"Error processing page {page_num}: {page_err}")
                                stats.warnings.append(f"Page {page_num} failed: {page_err}")
                                c.setPageSize((595, 842))
                                c.showPage()
                                page_standalone_flags.append(False)

                            # Free page data immediately
                            del result

                            # Adaptive GC: on constrained systems, collect after
                            # every page to keep peak memory low.
                            if pipe_cfg.gc_after_page:
                                gc.collect()

                        # OCR subprocess stays alive for next chunk (no restart penalty)

                        logger.info(
                            f"Chunk {chunk_idx + 1}/{num_chunks} done "
                            f"(pages {chunk_start + 1}-{chunk_end})"
                        )

                        # Free accumulated memory between chunks
                        if pipe_cfg.gc_after_chunk:
                            gc.collect()

                finally:
                    self._stop_ocr_subprocess(ocr_proc)

            c.save()
            stats.average_confidence = total_confidence
            self._page_standalone_flags = page_standalone_flags

            # 4. Merge text layer with original
            if progress_callback:
                progress_callback(85, 100, _("Merging text layer..."))

            any_standalone = any(page_standalone_flags) if page_standalone_flags else False
            format_changed = self.config.image_export_format not in ("original", "")

            if format_changed:
                import shutil

                shutil.copy2(text_layer_pdf, merged_pdf)
            elif any_standalone:
                smart_merge_pdfs(input_pdf, text_layer_pdf, merged_pdf, page_standalone_flags)
            else:
                self._overlay_text_on_original(input_pdf, text_layer_pdf, merged_pdf)

            # 5. OCR images within native text pages (editor-merged files)
            # These pages were skipped by the image-only pipeline to preserve
            # their original text content. Now run OCR on their embedded images
            # (logos, graphics with text) and add invisible text overlays.
            if native_text_pages and merged_pdf.exists():
                if progress_callback:
                    progress_callback(87, 100, _("Processing text page images..."))
                self._ocr_native_text_page_images(
                    merged_pdf, native_text_pages, stats, progress_callback
                )

                # Extract native text from the output PDF so the text
                # viewer has content to display for native text pages.
                native_text = self._extract_native_text(merged_pdf)
                if native_text.strip():
                    if stats.full_text:
                        stats.full_text = native_text.strip() + "\n\n" + stats.full_text
                    else:
                        stats.full_text = native_text.strip()

            # 6. Apply editor modifications
            start_page = 1
            if self.config.page_range:
                start_page = self.config.page_range[0]

            if self.config.page_modifications:
                apply_final_rotation_to_pdf(merged_pdf, page_rotations, start_page)

            split_parts = self._finalize_output(merged_pdf, output_pdf, progress_callback)
            if split_parts:
                stats.split_output_files = [str(p) for p in split_parts]

        except InterruptedError:
            logger.info("Processing cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Error in image-only PDF processing: {e}")
            stats.error = str(e)
            raise
        finally:
            if "images_temp_dir" in locals():
                images_temp_dir.cleanup()
            # Clean up temporary PDFs in the output directory
            for temp_pdf in (merged_pdf, text_layer_pdf):
                try:
                    if temp_pdf.exists():
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

    @staticmethod
    def _get_page_end_ctm(page) -> tuple[float, ...]:
        """Return the active CTM at the end of a page's content stream.

        Tracks q/Q (graphics state save/restore) and cm (concat matrix)
        operators to determine what transform is in effect when new
        content is appended to the page.
        """
        identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        ctm = identity
        stack: list[tuple[float, ...]] = []

        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception:
            return identity

        for operands, operator in ops:
            op = str(operator)
            if op == "q":
                stack.append(ctm)
            elif op == "Q":
                if stack:
                    ctm = stack.pop()
            elif op == "cm" and len(operands) == 6:
                vals = [float(v) for v in operands]
                a2, b2, c2, d2, e2, f2 = vals
                a1, b1, c1, d1, e1, f1 = ctm
                # CTM' = M × CTM  (PDF pre-multiplication)
                ctm = (
                    a2 * a1 + b2 * c1,
                    a2 * b1 + b2 * d1,
                    c2 * a1 + d2 * c1,
                    c2 * b1 + d2 * d1,
                    e2 * a1 + f2 * c1 + e1,
                    e2 * b1 + f2 * d1 + f1,
                )

        return ctm

    @staticmethod
    def _invert_ctm(
        ctm: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        """Return the inverse of a 2D affine CTM, or None if identity."""
        a, b, c, d, e, f = ctm
        # If already identity, no correction needed
        if (
            abs(a - 1) < 1e-6
            and abs(b) < 1e-6
            and abs(c) < 1e-6
            and abs(d - 1) < 1e-6
            and abs(e) < 1e-6
            and abs(f) < 1e-6
        ):
            return None

        det = a * d - b * c
        if abs(det) < 1e-12:
            return None

        return (
            d / det,
            -b / det,
            -c / det,
            a / det,
            (c * f - d * e) / det,
            (b * e - a * f) / det,
        )

    def _ocr_native_text_page_images(
        self,
        merged_pdf_path: Path,
        native_text_pages: set[int],
        stats: ProcessingStats,
        progress_callback=None,
    ) -> None:
        """OCR images within native text pages after the main merge.

        Opens the merged PDF with pikepdf, extracts images directly via
        PdfImage (no pdfimages subprocess, no preprocessing), runs OCR,
        and adds invisible text overlays. This preserves the original
        page content while making embedded images searchable.
        """
        from pikepdf import PdfImage

        image_positions = extract_image_positions(merged_pdf_path)

        # Filter to only native text pages that have images
        pages_to_process = {
            p: imgs for p, imgs in image_positions.items() if p in native_text_pages and imgs
        }

        if not pages_to_process:
            logger.info("No images found in native text pages, skipping")
            return

        total_images = sum(len(imgs) for imgs in pages_to_process.values())
        logger.info(
            f"OCR'ing {total_images} images in {len(pages_to_process)} "
            f"native text pages: {sorted(pages_to_process.keys())}"
        )

        ocr_proc = self._launch_ocr_subprocess()
        try:
            self._wait_for_ocr_ready(ocr_proc)

            with pikepdf.open(merged_pdf_path, allow_overwriting_input=True) as pdf:
                ocr_count = 0
                pages_done = 0
                total_native = len(pages_to_process)
                for page_num, img_positions in sorted(pages_to_process.items()):
                    if page_num > len(pdf.pages):
                        continue

                    if progress_callback:
                        pct = 87 + int(8 * pages_done / total_native)
                        progress_callback(
                            pct,
                            100,
                            _("Processing page {0}/{1}...").format(pages_done + 1, total_native),
                        )
                    pages_done += 1

                    page = pdf.pages[page_num - 1]
                    mediabox = page.mediabox
                    page_width = float(mediabox[2]) - float(mediabox[0])
                    page_height = float(mediabox[3]) - float(mediabox[1])

                    xobjects = {}
                    if "/Resources" in page and "/XObject" in page.Resources:
                        xobjects = page.Resources.XObject

                    # Compute the active CTM at end of existing content stream
                    # so we can undo it before our text overlay commands.
                    end_ctm = self._get_page_end_ctm(page)

                    page_text_commands: list[str] = []

                    for img_pos in img_positions:
                        # Skip very small images (icons, spacers).
                        # Threshold in PDF points: 15pt ≈ 5mm.
                        if img_pos.width < 15 or img_pos.height < 15:
                            continue

                        # Extract image directly from PDF via pikepdf
                        xobj_name = img_pos.name
                        if xobj_name not in xobjects:
                            continue

                        xobj = xobjects[xobj_name]
                        if not hasattr(xobj, "Width"):
                            continue

                        # Also skip if pixel dimensions are too small for OCR
                        px_w, px_h = int(xobj.Width), int(xobj.Height)
                        if px_w < 50 or px_h < 30:
                            continue

                        try:
                            pil_img = PdfImage(xobj).as_pil_image().convert("RGB")
                            img_array = np.array(pil_img)
                        except Exception as e:
                            logger.debug(f"Could not extract {xobj_name}: {e}")
                            continue

                        # OCR directly without preprocessing (these are web
                        # graphics, not scanned documents)
                        ocr_results = self._ocr_via_persistent(img_array, ocr_proc)
                        if not ocr_results:
                            continue

                        img_h, img_w = img_array.shape[:2]
                        scale_x = img_pos.width / img_w if img_w else 1
                        scale_y = img_pos.height / img_h if img_h else 1

                        text_commands = self._create_text_layer_commands(
                            ocr_results,
                            img_pos.x,
                            img_pos.y,
                            img_pos.width,
                            img_pos.height,
                            scale_x,
                            scale_y,
                        )
                        # Collect text commands (skip the leading q and trailing Q)
                        for cmd in text_commands:
                            if cmd not in ("q", "Q"):
                                page_text_commands.append(cmd)

                        stats.total_text_regions += len(ocr_results)
                        ocr_count += len(ocr_results)
                        logger.debug(
                            f"Page {page_num}: OCR'd {xobj_name} ({len(ocr_results)} text regions)"
                        )

                    if page_text_commands:
                        # Wrap in q/Q with CTM reset to page coordinates
                        wrapped = ["q"]
                        inv = self._invert_ctm(end_ctm)
                        if inv:
                            a, b, c, d, e, f = inv
                            wrapped.append(f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} cm")
                        wrapped.extend(page_text_commands)
                        wrapped.append("Q")
                        self._append_text_to_page(pdf, page, wrapped)

                pdf.save(merged_pdf_path)

        finally:
            self._stop_ocr_subprocess(ocr_proc)

        logger.info(f"Native text page image OCR complete ({total_images} images)")

    def _ocr_via_persistent(
        self,
        image: np.ndarray,
        ocr_proc: subprocess.Popen,
    ) -> list[OCRResult]:
        """Run OCR on a numpy image using the persistent subprocess.

        Writes the image to a temp file, sends the path to the persistent
        subprocess, and converts the raw dict result to OCRResult objects.
        """
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(temp_path, image)
            raw = self._ocr_image_via_subprocess(ocr_proc, temp_path)
            if not raw or raw.get("error") or not raw.get("boxes"):
                return []

            # Refine via persistent subprocess
            def _persistent_ocr(crop_path: str) -> dict | None:
                return self._ocr_image_via_subprocess(ocr_proc, crop_path)

            raw = refine_ocr_results(raw, temp_path, _persistent_ocr)

            results = []
            for i in range(len(raw["boxes"])):
                results.append(OCRResult(raw["txts"][i], raw["boxes"][i], raw["scores"][i]))

            min_score = self.config.text_score_threshold
            results = [r for r in results if r.confidence >= min_score]
            return results
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _ocr_image_in_page(
        self,
        img_path: Path,
        img_pos,
        pdf: "pikepdf.Pdf",
        page: "pikepdf.Page",
        page_num: int,
        page_width: float,
        page_height: float,
        stats: ProcessingStats,
        ocr_proc: subprocess.Popen | None = None,
    ) -> list[str]:
        """OCR a single image within a page and overlay invisible text.

        Args:
            ocr_proc: Optional persistent OCR subprocess.  When provided,
                images are sent over stdin instead of spawning a new process.

        Returns:
            List of OCR text strings found in the image.
        """
        # Skip very small images (likely icons or decorations)
        if img_pos.width < 50 or img_pos.height < 50:
            logger.debug(f"Skipping small image: {img_pos.width}x{img_pos.height}")
            return []

        img = load_image_with_exif_rotation(img_path)
        if img is None:
            logger.warning(f"Could not load image: {img_path}")
            return []

        processed_img = self.preprocessor.process(img)

        if ocr_proc is not None:
            ocr_results = self._ocr_via_persistent(processed_img, ocr_proc)
        else:
            ocr_results = self._run_ocr(processed_img)

        if not ocr_results:
            logger.debug(f"No text found in image at ({img_pos.x}, {img_pos.y})")
            return []

        img_h, img_w = processed_img.shape[:2]
        scale_x = img_pos.width / img_w
        scale_y = img_pos.height / img_h

        text_commands = self._create_text_layer_commands(
            ocr_results,
            img_pos.x,
            img_pos.y,
            img_pos.width,
            img_pos.height,
            scale_x,
            scale_y,
        )
        self._append_text_to_page(pdf, page, text_commands)

        stats.total_text_regions += len(ocr_results)

        formatted_text = self._format_ocr_text(ocr_results, float(img_w))

        # Collect OCR boxes for ODF export
        for r in ocr_results:
            xs = [p[0] for p in r.box]
            ys = [p[1] for p in r.box]
            box_x = min(xs) * scale_x + img_pos.x
            box_y = min(ys) * scale_y + img_pos.y
            box_w = (max(xs) - min(xs)) * scale_x
            box_h = (max(ys) - min(ys)) * scale_y

            x_pct = (box_x / page_width) * 100 if page_width > 0 else 0
            y_pct = (box_y / page_height) * 100 if page_height > 0 else 0
            w_pct = (box_w / page_width) * 100 if page_width > 0 else 0

            stats.ocr_boxes.append(
                OCRBoxData(
                    text=r.text,
                    x=x_pct,
                    y=y_pct,
                    width=w_pct,
                    height=box_h,
                    confidence=r.confidence,
                    page_num=page_num,
                )
            )

        logger.debug(
            f"Added {len(ocr_results)} text regions for image at ({img_pos.x:.1f}, {img_pos.y:.1f})"
        )

        return [formatted_text] if formatted_text else []

    @staticmethod
    def _extract_native_text(input_pdf: Path, pages: set[int] | None = None) -> str:
        """Extract existing text from PDF using pdftotext.

        Uses plain mode (no -layout) for cleaner text flow, then
        post-processes with conservative reflow for mid-sentence joins.

        Args:
            input_pdf: Path to PDF file.
            pages: If given, extract only these page numbers (1-based).
        """
        import re

        try:
            result = subprocess.run(
                ["pdftotext", str(input_pdf), "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return ""
        except Exception as e:
            logger.warning(f"Could not extract native text: {e}")
            return ""

        text = result.stdout

        # Filter to specific pages if requested
        if pages is not None:
            page_texts = text.split("\f")
            selected = []
            for i, page_text in enumerate(page_texts):
                if (i + 1) in pages and page_text.strip():
                    selected.append(page_text)
            if not selected:
                return ""
            text = "\f".join(selected)

        # Remove form-feed characters (page breaks)
        text = text.replace("\f", "\n\n")

        # Conservative reflow: only join mid-sentence continuations
        lines = text.split("\n")
        reflowed: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                reflowed.append("")
                i += 1
                continue

            para = line.rstrip()
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    break
                stripped_next = next_line.strip()
                para_end = para.rstrip()
                if not para_end:
                    break

                # Hyphenated word break: "word-\ncontinuation" → join
                if (
                    para_end.endswith("-")
                    and len(para_end) > 1
                    and para_end[-2].isalpha()
                    and stripped_next
                    and stripped_next[0].islower()
                ):
                    para = para_end[:-1] + stripped_next
                    i += 1
                    continue

                # Mid-sentence continuation: prev ends with letter/comma,
                # next starts with lowercase → clearly same sentence
                last_ch = para_end[-1]
                if (last_ch.isalpha() or last_ch == ",") and stripped_next[0].islower():
                    para = para_end + " " + stripped_next
                    i += 1
                    continue

                # Otherwise, keep as separate lines
                break

            reflowed.append(para)

        text = "\n".join(reflowed)

        # Collapse multiple blank lines into max 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text

    @staticmethod
    def _extract_and_filter_images(input_pdf: Path, images_dir: Path) -> list[Path]:
        """Extract images from PDF and filter out masks/small icons."""
        cmd = ["pdfimages", "-all", str(input_pdf), str(images_dir / "img")]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pdfimages failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract images: {e}") from e

        extracted = sorted(images_dir.glob("img-*"))
        filtered = [
            img
            for img in extracted
            if not (img.stat().st_size < 5000 and img.suffix.lower() == ".png")
        ]
        logger.info(f"Extracted {len(filtered)} images (after filtering masks)")
        return filtered

    def _process_mixed_content_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process mixed content PDF (text + images).

        Preserves original PDF structure and adds invisible OCR text layers
        at image positions.
        """
        import shutil

        start_time = time.time()
        stats = ProcessingStats()

        if progress_callback:
            progress_callback(0, 100, _("Analyzing PDF structure..."))

        image_positions = extract_image_positions(input_pdf)

        if not image_positions:
            logger.info("No images found in PDF. Copying original.")
            shutil.copy2(input_pdf, output_pdf)
            stats.warnings.append("No images found to OCR in mixed content PDF")
            self._calculate_final_stats(stats, start_time)
            return stats

        total_images = sum(len(imgs) for imgs in image_positions.values())
        logger.info(f"Found {total_images} image(s) across {len(image_positions)} page(s)")

        # Count total pages to determine which are text-only
        with pikepdf.open(input_pdf) as pdf_count:
            total_pages = len(pdf_count.pages)
        stats.pages_total = total_pages

        # Extract native text ONLY for pages without images (avoids
        # duplicating OCR content with worse table formatting)
        pages_with_images = set(image_positions.keys())
        text_only_pages = set(range(1, total_pages + 1)) - pages_with_images
        if text_only_pages:
            native_text = self._extract_native_text(input_pdf, text_only_pages)
            logger.info(f"Extracted native text for {len(text_only_pages)} text-only page(s)")
        else:
            native_text = ""
            logger.info("All pages have images; skipping native text extraction")
        ocr_texts = []

        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "images"
            images_dir.mkdir()

            if progress_callback:
                progress_callback(5, 100, _("Extracting images..."))

            extracted_images = self._extract_and_filter_images(input_pdf, images_dir)

            # Launch persistent subprocess once (avoids model reload per image)
            ocr_proc = self._launch_ocr_subprocess()
            try:
                self._wait_for_ocr_ready(ocr_proc)

                with pikepdf.open(input_pdf, allow_overwriting_input=True) as pdf:
                    processed_images = 0
                    current_img_idx = 0

                    for page_num in sorted(image_positions.keys()):
                        # Check for cancellation between pages
                        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                            logger.info("Processing cancelled by user in mixed content mode")
                            raise InterruptedError("Processing cancelled by user")

                        page_imgs = image_positions[page_num]
                        page = pdf.pages[page_num - 1]
                        mediabox = page.mediabox
                        page_height = float(mediabox[3]) - float(mediabox[1])
                        page_width = float(mediabox[2]) - float(mediabox[0])

                        if progress_callback:
                            pct = 10 + int(80 * processed_images / total_images)
                            progress_callback(pct, 100, _("OCR page {0}...").format(page_num))

                        for img_pos in page_imgs:
                            if current_img_idx >= len(extracted_images):
                                logger.warning(
                                    f"Image index {current_img_idx} exceeds extracted images"
                                )
                                break

                            img_path = extracted_images[current_img_idx]
                            current_img_idx += 1

                            try:
                                texts = self._ocr_image_in_page(
                                    img_path,
                                    img_pos,
                                    pdf,
                                    page,
                                    page_num,
                                    page_width,
                                    page_height,
                                    stats,
                                    ocr_proc=ocr_proc,
                                )
                                ocr_texts.extend(texts)
                                if texts:
                                    processed_images += 1
                            except Exception as e:
                                logger.error(f"Error processing image {img_path}: {e}")
                                stats.warnings.append(f"Failed to OCR image: {e}")
                                continue

                    if progress_callback:
                        progress_callback(90, 100, _("Saving PDF..."))

                    stats.pages_processed = len(image_positions)
                    pdf.save(output_pdf)
            finally:
                self._stop_ocr_subprocess(ocr_proc)

        # Split by file size if limit is configured
        max_mb = self.config.max_file_size_mb
        if max_mb > 0:
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                if progress_callback:
                    progress_callback(92, 100, _("Splitting PDF by size limit..."))
                split_parts = self._split_pdf_by_size(output_pdf, max_mb)
                if split_parts:
                    stats.split_output_files = [str(p) for p in split_parts]

        # Combine native text (text-only pages) with OCR text (image pages)
        parts = []
        if native_text and native_text.strip():
            parts.append(native_text.strip())
        if ocr_texts:
            parts.append("\n".join(ocr_texts))
        stats.full_text = "\n\n".join(parts)

        self._calculate_final_stats(stats, start_time)

        if progress_callback:
            progress_callback(100, 100, _("Done!"))

        logger.info(f"Mixed content processing complete in {stats.processing_time_seconds:.1f}s")
        logger.info(f"Pages: {stats.pages_processed}, Text regions: {stats.total_text_regions}")

        return stats

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
            self._convert_to_pdfa(merged_pdf, output_pdf)
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
        """Split a PDF into multiple parts that each stay under the size limit.

        Output files are named: {stem}-01.pdf, {stem}-02.pdf, etc.
        The original file is replaced by the split parts.

        Args:
            output_pdf: Path to the PDF to split
            max_mb: Maximum file size in MB per part

        Returns:
            List of split part paths, or empty list if no split was needed
        """
        import io

        import pikepdf

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
