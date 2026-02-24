"""
Adaptive Resource Management for OCR Processing.

Detects available system resources (RAM, CPU) and provides dynamic
configuration for the processing pipeline. Systems with more resources
get higher parallelism and throughput; constrained systems get conservative
settings that avoid swapping and OOM.

Resource tiers:
  - CONSTRAINED: < 2 GB available RAM → single worker, small chunks, aggressive GC
  - MODERATE:    2-6 GB available RAM → balanced workers, medium chunks
  - ABUNDANT:    > 6 GB available RAM → max workers, large chunks, prefetch

All thresholds are derived from measured memory profiles:
  - Base process overhead:   ~150 MB
  - OCR subprocess (model):  ~400 MB
  - Per-worker preprocessing: ~200 MB peak (large page at 300 DPI)
  - DBNet inference:          ~50 MB (shared model, per-call ~15 MB)
  - Per-page peak (full):     ~350 MB (preprocessing + probmap + deskew)
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto

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
        enable_prefetch: Prefetch next chunk while processing current
        gc_after_page: Force gc.collect() after each page
        gc_after_chunk: Force gc.collect() after each chunk
        downscale_probmap: Max side for probmap inference (lower = less RAM)
    """

    max_workers: int
    chunk_size: int
    ocr_threads: int
    enable_prefetch: bool
    gc_after_page: bool
    gc_after_chunk: bool
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

    # Determine tier
    available_gb = available_mb / 1024
    if available_gb < 2.0:
        tier = ResourceTier.CONSTRAINED
    elif available_gb < 6.0:
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
    ocr_overhead_mb = 400  # OCR model in subprocess
    base_overhead_mb = 150  # Main process base
    worker_cost_mb = 200  # Per preprocessing worker peak

    usable_mb = available_mb - ocr_overhead_mb - base_overhead_mb

    if profile.tier == ResourceTier.CONSTRAINED:
        # Minimal mode: 1 worker, small chunks, aggressive GC
        max_workers = 1
        chunk_size = 4
        ocr_threads = max(2, cpu_count // 2)
        enable_prefetch = False
        gc_after_page = True
        gc_after_chunk = True
        downscale_probmap = 1024  # Smaller inference = less RAM

    elif profile.tier == ResourceTier.MODERATE:
        # Balanced mode
        ram_workers = max(1, int(usable_mb * 0.7 / worker_cost_mb))
        cpu_workers = max(1, cpu_count - 2)
        max_workers = min(ram_workers, cpu_workers, 6)
        chunk_size = 8
        ocr_threads = max(2, cpu_count // 2)
        enable_prefetch = False
        gc_after_page = False
        gc_after_chunk = True
        downscale_probmap = 1536

    else:  # ABUNDANT
        # High-performance mode: maximize throughput
        ram_workers = max(1, int(usable_mb * 0.7 / worker_cost_mb))
        cpu_workers = max(1, cpu_count - 2)
        max_workers = min(ram_workers, cpu_workers, 12)
        chunk_size = min(max_workers * 2, 20)
        ocr_threads = max(2, cpu_count)
        enable_prefetch = True
        gc_after_page = False
        gc_after_chunk = True
        downscale_probmap = 1536

    config = PipelineConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        ocr_threads=ocr_threads,
        enable_prefetch=enable_prefetch,
        gc_after_page=gc_after_page,
        gc_after_chunk=gc_after_chunk,
        downscale_probmap=downscale_probmap,
    )

    logger.info(
        f"Pipeline config: workers={max_workers}, chunk={chunk_size}, "
        f"ocr_threads={ocr_threads}, prefetch={enable_prefetch}, "
        f"gc_page={gc_after_page}, gc_chunk={gc_after_chunk}, "
        f"probmap_max={downscale_probmap}"
    )

    return config
