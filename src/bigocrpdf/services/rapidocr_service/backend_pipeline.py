"""PDF Processing Pipeline Mixin for ProfessionalPDFOCR."""

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

from bigocrpdf.constants import MIN_IMAGE_BOX_SIZE_PX, MIN_TEXT_BOX_HEIGHT_PX, MIN_TEXT_BOX_WIDTH_PX
from bigocrpdf.services.rapidocr_service.config import OCRBoxData, OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import refine_ocr_results
from bigocrpdf.services.rapidocr_service.pdf_assembly import smart_merge_pdfs
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_image_positions,
    get_page_image_encodings,
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
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import MixedContentMixin
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class BackendPipelineMixin(ChunkedOCRMixin, MixedContentMixin):
    """Mixin providing PDF processing pipeline methods."""

    def _build_ocr_subprocess_cmd(self, ocr_threads: int = 0) -> list[str]:
        """Build command-line args for persistent OCR subprocess.

        Args:
            ocr_threads: Number of inference threads. 0 = auto (physical cores).

        Returns:
            Command list ready for subprocess.Popen.
        """
        import multiprocessing

        worker_script = str(Path(__file__).parent / "ocr_worker.py")
        cpu_count = multiprocessing.cpu_count()

        if ocr_threads <= 0:
            # Use all logical cores; OpenVINO handles HT internally.
            ocr_threads = max(2, cpu_count)

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

        if self.config.detection_full_resolution:
            cmd.append("--full-resolution")

        try:
            if not self._check_openvino_available():
                cmd.append("--no-openvino")
        except (ImportError, OSError, AttributeError):
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
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
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
            assert proc.stdin is not None
            assert proc.stdout is not None

            proc.stdin.write(f"{image_path}\n")
            proc.stdin.flush()

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
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
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
        images_dir = Path(images_temp_dir.name) / "chunk_imgs"
        images_dir.mkdir()

        merged_pdf = output_dir / f".{output_pdf.stem}_processing.pdf"
        text_layer_pdf = output_dir / f".{output_pdf.stem}_textlayer.pdf"

        try:
            # Phase 1: Analyze PDF metadata
            ctx = self._analyze_pdf_metadata(input_pdf, stats, pipe_cfg, progress_callback)
            if ctx["total_pages"] == 0:
                return stats

            # Phase 2: Chunked extraction + preprocessing + OCR
            self._run_chunked_ocr_pipeline(
                input_pdf,
                text_layer_pdf,
                images_dir,
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
            if "images_temp_dir" in locals():
                images_temp_dir.cleanup()
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

        all_rotation_dicts = []
        for rot in page_rotations:
            all_rotation_dicts.append({
                "rotation": rot.original_pdf_rotation,
                "mediabox": rot.mediabox,
                "page_rotation": rot,
            })

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

        any_standalone = any(page_standalone_flags) if page_standalone_flags else False
        format_changed = self.config.image_export_format not in ("original", "")

        if format_changed:
            import shutil

            shutil.copy2(text_layer_pdf, merged_pdf)
        elif any_standalone:
            smart_merge_pdfs(input_pdf, text_layer_pdf, merged_pdf, page_standalone_flags)
        else:
            self._overlay_text_on_original(input_pdf, text_layer_pdf, merged_pdf)

        # OCR images within native text pages (editor-merged files)
        if native_text_pages and merged_pdf.exists():
            if progress_callback:
                progress_callback(87, 100, _("Processing text page images..."))
            self._ocr_native_text_page_images(
                merged_pdf, native_text_pages, stats, progress_callback
            )

            native_text = self._extract_native_text(merged_pdf)
            if native_text.strip():
                if stats.full_text:
                    stats.full_text = native_text.strip() + "\n\n" + stats.full_text
                else:
                    stats.full_text = native_text.strip()

        # Apply editor modifications
        start_page = 1
        if self.config.page_range:
            start_page = self.config.page_range[0]

        if self.config.page_modifications:
            apply_final_rotation_to_pdf(merged_pdf, page_rotations, start_page)

        # Optimize bilevel images (JBIG2/CCITT re-encoding)
        if self.config.enable_bilevel_compression and page_result_encodings:
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
                stats.warnings.append(_("{0} pages re-encoded with JBIG2").format(n_opt))

        split_parts = self._finalize_output(merged_pdf, output_pdf, progress_callback)
        if split_parts:
            stats.split_output_files = [str(p) for p in split_parts]

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

    def _extract_xobj_image(self, xobj, xobj_name: str, PdfImage):
        """Extract a numpy RGB array from a PDF XObject, or None."""
        if not hasattr(xobj, "Width"):
            return None
        px_w, px_h = int(xobj.Width), int(xobj.Height)
        if px_w < MIN_TEXT_BOX_WIDTH_PX or px_h < MIN_TEXT_BOX_HEIGHT_PX:
            return None
        try:
            return np.array(PdfImage(xobj).as_pil_image().convert("RGB"))
        except Exception as e:
            logger.debug(f"Could not extract {xobj_name}: {e}")
            return None

    def _ocr_page_embedded_images(
        self,
        img_positions,
        xobjects,
        ocr_proc,
        stats,
        page_num,
        PdfImage,
    ) -> list[str]:
        """OCR all embedded images in one page; return text overlay commands."""
        text_commands: list[str] = []
        for img_pos in img_positions:
            if img_pos.width < 15 or img_pos.height < 15:
                continue
            if img_pos.name not in xobjects:
                continue
            img_array = self._extract_xobj_image(
                xobjects[img_pos.name],
                img_pos.name,
                PdfImage,
            )
            if img_array is None:
                continue
            ocr_results = self._ocr_via_persistent(img_array, ocr_proc)
            if not ocr_results:
                continue
            img_h, img_w = img_array.shape[:2]
            scale_x = img_pos.width / img_w if img_w else 1
            scale_y = img_pos.height / img_h if img_h else 1
            cmds = self._create_text_layer_commands(
                ocr_results,
                img_pos.x,
                img_pos.y,
                img_pos.width,
                img_pos.height,
                scale_x,
                scale_y,
            )
            for cmd in cmds:
                if cmd not in ("q", "Q"):
                    text_commands.append(cmd)
            stats.total_text_regions += len(ocr_results)
            logger.debug(f"Page {page_num}: OCR'd {img_pos.name} ({len(ocr_results)} text regions)")
        return text_commands

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
                    xobjects = {}
                    if "/Resources" in page and "/XObject" in page.Resources:
                        xobjects = page.Resources.XObject

                    page_text_commands = self._ocr_page_embedded_images(
                        img_positions,
                        xobjects,
                        ocr_proc,
                        stats,
                        page_num,
                        PdfImage,
                    )

                    if page_text_commands:
                        end_ctm = self._get_page_end_ctm(page)
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
        skip_preprocessing: bool = False,
    ) -> list[str]:
        """OCR a single image within a page and overlay invisible text.

        Args:
            ocr_proc: Optional persistent OCR subprocess.  When provided,
                images are sent over stdin instead of spawning a new process.
            skip_preprocessing: Skip geometric preprocessing (for mixed content
                images that are already properly aligned).

        Returns:
            List of OCR text strings found in the image.
        """
        # Skip very small images (likely icons or decorations)
        if img_pos.width < MIN_IMAGE_BOX_SIZE_PX or img_pos.height < MIN_IMAGE_BOX_SIZE_PX:
            logger.debug(f"Skipping small image: {img_pos.width}x{img_pos.height}")
            return []

        img = load_image_with_exif_rotation(img_path)
        if img is None:
            logger.warning(f"Could not load image: {img_path}")
            return []

        if skip_preprocessing:
            processed_img = img
        else:
            processed_img = self.preprocessor.process(img)
            self._replace_pdf_image(page, img_pos.name, processed_img)

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
    def _replace_pdf_image(page, img_name: str, img_array: np.ndarray) -> None:
        """Replace a PDF image XObject with a preprocessed image."""
        import io

        from PIL import Image

        try:
            xobj = page.Resources.XObject[img_name]
        except (AttributeError, KeyError):
            logger.warning(f"Could not find image {img_name} to replace")
            return

        h, w = img_array.shape[:2]
        is_gray = len(img_array.shape) == 2

        if is_gray:
            pil_img = Image.fromarray(img_array, mode="L")
            colorspace = pikepdf.Name.DeviceGray
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            colorspace = pikepdf.Name.DeviceRGB

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)

        xobj.write(buf.getvalue(), filter=pikepdf.Name.DCTDecode)
        xobj[pikepdf.Name.Width] = w
        xobj[pikepdf.Name.Height] = h
        xobj[pikepdf.Name.ColorSpace] = colorspace
        xobj[pikepdf.Name.BitsPerComponent] = 8

        # Remove transparency mask — JPEG does not support alpha
        if pikepdf.Name.SMask in xobj:
            del xobj[pikepdf.Name.SMask]

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
