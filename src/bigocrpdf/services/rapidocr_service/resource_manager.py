"""
Adaptive Resource Management for OCR Processing.

Detects available system resources (RAM, CPU) and provides dynamic
configuration for the processing pipeline. Systems with more resources
get higher parallelism and throughput; constrained systems get conservative
settings that avoid swapping and OOM.

Resource tiers:
  - CONSTRAINED: < 2 GB available RAM → single worker, small chunks, aggressive GC
  - MODERATE:    2-6 GB available RAM → balanced workers, medium chunks
  - ABUNDANT:    > 6 GB available RAM → max workers, large chunks

All thresholds are derived from measured memory profiles:
  - Base process overhead:   ~150 MB
  - OCR subprocess (model):  ~400 MB
  - Per-worker preprocessing: ~200 MB peak (large page at 300 DPI)
  - DBNet inference:          ~50 MB (shared model, per-call ~15 MB)
  - Per-page peak (full):     ~350 MB (preprocessing + probmap + deskew)
"""

import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from bigocrpdf.constants import (
    BASE_PROCESS_OVERHEAD_MB,
    OCR_SUBPROCESS_OVERHEAD_MB,
    PER_WORKER_COST_MB,
    RESOURCE_TIER_CONSTRAINED_GB,
    RESOURCE_TIER_MODERATE_GB,
)

logger = logging.getLogger(__name__)


class ResourceTier(Enum):
    """System resource tier for adaptive configuration."""

    CONSTRAINED = auto()  # < 2 GB free RAM
    MODERATE = auto()  # 2-6 GB free RAM
    ABUNDANT = auto()  # > 6 GB free RAM


@dataclass(frozen=True)
class ResourceProfile:
    """Snapshot of available system resources.

    Attributes:
        available_ram_mb: Currently available RAM in MB
        total_ram_mb: Total system RAM in MB
        cpu_count: Number of logical CPU cores
        tier: Computed resource tier
    """

    available_ram_mb: int
    total_ram_mb: int
    cpu_count: int
    tier: ResourceTier


@dataclass(frozen=True)
class PipelineConfig:
    """Dynamic pipeline configuration based on available resources.

    Attributes:
        max_workers: Maximum preprocessing worker processes
        chunk_size: Pages per processing chunk
        ocr_threads: Threads for OCR inference subprocess
        gc_after_page: Force gc.collect() after each page
        downscale_probmap: Max side for probmap inference (lower = less RAM)
    """

    max_workers: int
    chunk_size: int
    ocr_threads: int
    gc_after_page: bool
    downscale_probmap: int


def detect_resources() -> ResourceProfile:
    """Detect current system resources.

    Uses psutil if available for accurate measurement. Falls back to
    os.sysconf for total RAM and assumes 50% available if psutil is
    not installed.

    Returns:
        ResourceProfile with current system state.
    """
    cpu_count = os.cpu_count() or 4

    try:
        import psutil

        mem = psutil.virtual_memory()
        available_mb = int(mem.available / (1024 * 1024))
        total_mb = int(mem.total / (1024 * 1024))
    except ImportError:
        # Fallback: read from /proc/meminfo (Linux)
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])  # kB

            total_mb = meminfo.get("MemTotal", 4 * 1024 * 1024) // 1024
            # MemAvailable is the best metric (includes reclaimable cache)
            available_mb = (
                meminfo.get(
                    "MemAvailable",
                    meminfo.get("MemFree", total_mb // 2 * 1024)
                    + meminfo.get("Buffers", 0)
                    + meminfo.get("Cached", 0),
                )
                // 1024
            )
        except (OSError, ValueError):
            # Last resort
            total_mb = 8192
            available_mb = total_mb // 2

    # Respect cgroup v2 memory limits (containers, Flatpak, systemd slices)
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
            if raw != "max":
                cgroup_limit_mb = int(raw) // (1024 * 1024)
                total_mb = min(total_mb, cgroup_limit_mb)
                available_mb = min(available_mb, cgroup_limit_mb)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            cgroup_used_mb = int(f.read().strip()) // (1024 * 1024)
            available_mb = min(available_mb, max(0, total_mb - cgroup_used_mb))
    except (OSError, ValueError):
        pass

    # Determine tier
    available_gb = available_mb / 1024
    if available_gb < RESOURCE_TIER_CONSTRAINED_GB:
        tier = ResourceTier.CONSTRAINED
    elif available_gb < RESOURCE_TIER_MODERATE_GB:
        tier = ResourceTier.MODERATE
    else:
        tier = ResourceTier.ABUNDANT

    profile = ResourceProfile(
        available_ram_mb=available_mb,
        total_ram_mb=total_mb,
        cpu_count=cpu_count,
        tier=tier,
    )

    logger.info(
        f"Resource detection: {available_mb} MB available / {total_mb} MB total, "
        f"{cpu_count} CPUs → {tier.name}"
    )

    return profile


def compute_pipeline_config(profile: ResourceProfile) -> PipelineConfig:
    """Compute optimal pipeline configuration for the given resource profile.

    The algorithm:
    1. Reserve memory for OCR subprocess (400 MB) + base overhead (150 MB)
    2. Divide remaining available RAM by per-worker cost (200 MB)
    3. Cap by CPU count (reserve 2 cores for OCR + system)
    4. Apply tier-specific constraints

    Args:
        profile: Current system resource profile.

    Returns:
        PipelineConfig with optimal settings.
    """
    available_mb = profile.available_ram_mb
    cpu_count = profile.cpu_count

    # Memory budget
    ocr_overhead_mb = OCR_SUBPROCESS_OVERHEAD_MB
    base_overhead_mb = BASE_PROCESS_OVERHEAD_MB
    worker_cost_mb = PER_WORKER_COST_MB

    usable_mb = available_mb - ocr_overhead_mb - base_overhead_mb

    if profile.tier == ResourceTier.CONSTRAINED:
        # Minimal mode: 1 worker, small chunks, aggressive GC
        max_workers = 1
        chunk_size = 4
        ocr_threads = max(2, cpu_count // 2)
        gc_after_page = True
        downscale_probmap = 1024  # Smaller inference = less RAM

    elif profile.tier == ResourceTier.MODERATE:
        # Balanced mode
        ram_workers = max(1, int(usable_mb * 0.7 / worker_cost_mb))
        cpu_workers = max(1, cpu_count - 2)
        max_workers = min(ram_workers, cpu_workers, 6)
        chunk_size = 8
        ocr_threads = max(2, cpu_count // 2)
        gc_after_page = False
        downscale_probmap = 1536

    else:  # ABUNDANT
        # High-performance mode: maximize throughput
        ram_workers = max(1, int(usable_mb * 0.7 / worker_cost_mb))
        cpu_workers = max(1, cpu_count - 2)
        max_workers = min(ram_workers, cpu_workers, 12)
        chunk_size = min(max_workers * 2, 20)
        # Single OCR subprocess maximizes throughput: OpenVINO
        # inference scales better with more threads in one process
        # than splitting across multiple processes (cache contention).
        ocr_threads = max(2, cpu_count)
        gc_after_page = False
        downscale_probmap = 1536

    config = PipelineConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        ocr_threads=ocr_threads,
        gc_after_page=gc_after_page,
        downscale_probmap=downscale_probmap,
    )

    logger.info(
        f"Pipeline config: workers={max_workers}, chunk={chunk_size}, "
        f"ocr_threads={ocr_threads}, "
        f"gc_page={gc_after_page}, "
        f"probmap_max={downscale_probmap}"
    )

    return config


# PDF points per inch
_PDF_POINTS_PER_INCH = 72.0
# Bytes per pixel (RGB)
_BYTES_PER_PIXEL = 3


def estimate_page_memory_mb(width_pts: float, height_pts: float, render_dpi: int = 300) -> float:
    """Estimate peak memory for rendering + OCR of a single page.

    Args:
        width_pts: Page width in PDF points (1/72 inch).
        height_pts: Page height in PDF points.
        render_dpi: DPI used for rasterization.

    Returns:
        Estimated peak memory in MB.
    """
    px_w = width_pts / _PDF_POINTS_PER_INCH * render_dpi
    px_h = height_pts / _PDF_POINTS_PER_INCH * render_dpi
    raw_mb = px_w * px_h * _BYTES_PER_PIXEL / (1024 * 1024)
    # Account for preprocessing copies (~3x raw) + inference overhead
    return raw_mb * 3.5


def estimate_page_megapixels(width_pts: float, height_pts: float, render_dpi: int = 300) -> float:
    """Estimate rendered page size in megapixels for a PDF page."""
    px_w = abs(width_pts) / _PDF_POINTS_PER_INCH * render_dpi
    px_h = abs(height_pts) / _PDF_POINTS_PER_INCH * render_dpi
    return px_w * px_h / 1_000_000


def select_render_dpi_for_page(
    width_pts: float,
    height_pts: float,
    preferred_dpi: int,
    max_megapixels: float,
    min_dpi: int = 150,
) -> int:
    """Choose a render DPI that stays under the configured megapixel budget."""
    if preferred_dpi <= 0 or max_megapixels <= 0:
        return preferred_dpi
    width_pts = abs(width_pts)
    height_pts = abs(height_pts)
    if (
        not math.isfinite(width_pts)
        or not math.isfinite(height_pts)
        or width_pts <= 0
        or height_pts <= 0
    ):
        raise ValueError("Page has invalid dimensions for the configured render budget")
    if estimate_page_megapixels(width_pts, height_pts, preferred_dpi) <= max_megapixels:
        return preferred_dpi

    page_area_inches = (width_pts / _PDF_POINTS_PER_INCH) * (height_pts / _PDF_POINTS_PER_INCH)
    effective_min_dpi = min(preferred_dpi, max(1, min_dpi))
    minimum_megapixels = estimate_page_megapixels(
        width_pts,
        height_pts,
        effective_min_dpi,
    )
    if minimum_megapixels > max_megapixels + 1e-9:
        raise ValueError(
            f"Page would render at {minimum_megapixels:.1f} MP at minimum "
            f"{effective_min_dpi} DPI; configured limit is {max_megapixels:.1f} MP"
        )

    capped_dpi = int(math.floor(math.sqrt((max_megapixels * 1_000_000) / page_area_inches)))
    return max(effective_min_dpi, min(preferred_dpi, capped_dpi))


def select_pdf_page_render_dpi(
    pdf_path: Path | str,
    page_num: int,
    preferred_dpi: int,
    max_megapixels: float,
    min_dpi: int = 150,
) -> int:
    """Choose render DPI for a PDF page, failing closed when its size cannot be inspected."""
    if max_megapixels <= 0:
        return preferred_dpi
    if page_num <= 0:
        raise ValueError(f"Invalid PDF page number for render budget: {page_num}")
    try:
        import pikepdf

        with pikepdf.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            mediabox = page.mediabox
            user_unit = float(page.get("/UserUnit", 1))
            width_pts = abs(float(mediabox[2]) - float(mediabox[0])) * user_unit
            height_pts = abs(float(mediabox[3]) - float(mediabox[1])) * user_unit
    except Exception as exc:
        raise ValueError(
            f"Could not inspect page {page_num} size required for the configured render budget"
        ) from exc
    if not math.isfinite(user_unit) or user_unit <= 0:
        raise ValueError(f"Page {page_num} has an invalid PDF UserUnit")
    return select_render_dpi_for_page(
        width_pts,
        height_pts,
        preferred_dpi,
        max_megapixels,
        min_dpi,
    )


def enforce_pdf_resource_limits(
    total_pages: int,
    page_dimensions: Iterable[tuple[float, float]],
    config,
    image_dimensions: Iterable[tuple[int, int, int]] = (),
) -> None:
    """Fail early when a PDF would exceed configured OCR resource limits."""
    max_pdf_pages = int(getattr(config, "max_pdf_pages", 0))
    if max_pdf_pages > 0 and total_pages > max_pdf_pages:
        raise ValueError(f"PDF has {total_pages} pages; configured limit is {max_pdf_pages}")

    max_page_megapixels = float(getattr(config, "max_page_megapixels", 0.0))
    if max_page_megapixels > 0:
        render_dpi = int(getattr(config, "dpi", 300))
        for page_index, (width_pts, height_pts) in enumerate(page_dimensions, 1):
            if (
                not math.isfinite(width_pts)
                or not math.isfinite(height_pts)
                or width_pts <= 0
                or height_pts <= 0
            ):
                raise ValueError(f"Page {page_index} has invalid dimensions")
            megapixels = estimate_page_megapixels(width_pts, height_pts, render_dpi)
            if megapixels > max_page_megapixels:
                raise ValueError(
                    f"Page {page_index} would render at {megapixels:.1f} MP "
                    f"({render_dpi} DPI); configured limit is {max_page_megapixels:.1f} MP"
                )

    enforce_image_resource_limits(image_dimensions, config)


def enforce_image_resource_limits(
    image_dimensions: Iterable[tuple[int, int, int]],
    config,
) -> None:
    """Reject embedded images whose declared decoded size exceeds the limit."""
    max_image_megapixels = float(getattr(config, "max_image_megapixels", 0.0))
    if max_image_megapixels <= 0:
        return

    for image_index, (page_num, width_px, height_px) in enumerate(image_dimensions, 1):
        if width_px <= 0 or height_px <= 0:
            raise ValueError(
                f"Image {image_index} on page {page_num} has invalid dimensions: "
                f"{width_px}x{height_px}"
            )
        megapixels = width_px * height_px / 1_000_000
        if megapixels > max_image_megapixels:
            raise ValueError(
                f"Image {image_index} on page {page_num} is {megapixels:.1f} MP; "
                f"configured limit is {max_image_megapixels:.1f} MP"
            )


def adjust_chunk_size(
    base_chunk: int,
    page_dimensions: list[tuple[float, float]],
    available_mb: int,
    render_dpi: int = 300,
) -> int:
    """Adjust chunk size based on actual page dimensions.

    Args:
        base_chunk: Chunk size from pipeline config.
        page_dimensions: List of (width_pts, height_pts) per page.
        available_mb: Available RAM in MB.
        render_dpi: DPI for rasterization.

    Returns:
        Adjusted chunk size (at least 1).
    """
    if not page_dimensions:
        return base_chunk
    max_mem = max(estimate_page_memory_mb(w, h, render_dpi) for w, h in page_dimensions)
    if max_mem <= 0:
        return base_chunk
    # How many pages fit in 60% of available memory
    safe_budget = available_mb * 0.6
    mem_chunk = max(1, int(safe_budget / max_mem))
    adjusted = min(base_chunk, mem_chunk)
    if adjusted != base_chunk:
        logger.info(
            f"Chunk size adjusted {base_chunk} → {adjusted} "
            f"(largest page ~{max_mem:.0f} MB, budget {safe_budget:.0f} MB)"
        )
    return adjusted
