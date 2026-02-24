"""Tests for resource_manager module."""

from unittest.mock import MagicMock, patch

from bigocrpdf.services.rapidocr_service.resource_manager import (
    PipelineConfig,
    ResourceProfile,
    ResourceTier,
    compute_pipeline_config,
    detect_resources,
)


class TestResourceTier:
    def test_enum_members(self):
        assert ResourceTier.CONSTRAINED.name == "CONSTRAINED"
        assert ResourceTier.MODERATE.name == "MODERATE"
        assert ResourceTier.ABUNDANT.name == "ABUNDANT"


class TestResourceProfile:
    def test_frozen(self):
        p = ResourceProfile(
            available_ram_mb=4096, total_ram_mb=16384, cpu_count=8, tier=ResourceTier.MODERATE
        )
        assert p.available_ram_mb == 4096
        assert p.total_ram_mb == 16384
        assert p.cpu_count == 8
        assert p.tier == ResourceTier.MODERATE

    def test_immutable(self):
        p = ResourceProfile(
            available_ram_mb=4096, total_ram_mb=16384, cpu_count=8, tier=ResourceTier.MODERATE
        )
        try:
            p.available_ram_mb = 0
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


class TestDetectResources:
    def test_with_psutil(self):
        mock_mem = MagicMock()
        mock_mem.available = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_mem.total = 16 * 1024 * 1024 * 1024  # 16 GB

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch("os.cpu_count", return_value=8):
                profile = detect_resources()

        assert profile.available_ram_mb == 4096
        assert profile.total_ram_mb == 16384
        assert profile.cpu_count == 8
        assert profile.tier == ResourceTier.MODERATE

    def test_tier_constrained(self):
        mock_mem = MagicMock()
        mock_mem.available = 1 * 1024 * 1024 * 1024  # 1 GB
        mock_mem.total = 4 * 1024 * 1024 * 1024

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch("os.cpu_count", return_value=4):
                profile = detect_resources()

        assert profile.tier == ResourceTier.CONSTRAINED

    def test_tier_abundant(self):
        mock_mem = MagicMock()
        mock_mem.available = 12 * 1024 * 1024 * 1024  # 12 GB
        mock_mem.total = 32 * 1024 * 1024 * 1024

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch("os.cpu_count", return_value=16):
                profile = detect_resources()

        assert profile.tier == ResourceTier.ABUNDANT

    def test_fallback_without_psutil(self):
        meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         2000000 kB\n"
            "MemAvailable:    5000000 kB\n"
            "Buffers:          500000 kB\n"
            "Cached:          2000000 kB\n"
        )
        # Force psutil import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            meminfo_cm = MagicMock()
            meminfo_cm.__enter__ = lambda s: iter(meminfo.splitlines(True))
            meminfo_cm.__exit__ = MagicMock(return_value=False)

            def _open_side_effect(path, *a, **kw):
                if "/proc/meminfo" in str(path):
                    return meminfo_cm
                raise OSError("not available")

            with patch("builtins.open", side_effect=_open_side_effect):
                with patch("os.cpu_count", return_value=4):
                    profile = detect_resources()
                    assert profile.available_ram_mb == 5000000 // 1024
                    assert profile.total_ram_mb == 16384000 // 1024
                    assert profile.cpu_count == 4


class TestComputePipelineConfig:
    def _profile(self, ram_mb, cpu, tier):
        return ResourceProfile(
            available_ram_mb=ram_mb, total_ram_mb=ram_mb * 2, cpu_count=cpu, tier=tier
        )

    def test_constrained_single_worker(self):
        p = self._profile(1500, 4, ResourceTier.CONSTRAINED)
        cfg = compute_pipeline_config(p)
        assert cfg.max_workers == 1
        assert cfg.chunk_size == 4
        assert cfg.enable_prefetch is False
        assert cfg.gc_after_page is True
        assert cfg.gc_after_chunk is True
        assert cfg.downscale_probmap == 1024

    def test_moderate_balanced(self):
        p = self._profile(4096, 8, ResourceTier.MODERATE)
        cfg = compute_pipeline_config(p)
        assert 1 <= cfg.max_workers <= 6
        assert cfg.chunk_size == 8
        assert cfg.enable_prefetch is False
        assert cfg.gc_after_page is False
        assert cfg.gc_after_chunk is True
        assert cfg.downscale_probmap == 1536

    def test_abundant_high_performance(self):
        p = self._profile(16384, 16, ResourceTier.ABUNDANT)
        cfg = compute_pipeline_config(p)
        assert cfg.max_workers > 1
        assert cfg.max_workers <= 12
        assert cfg.enable_prefetch is True
        assert cfg.gc_after_page is False
        assert cfg.gc_after_chunk is True
        assert cfg.downscale_probmap == 1536

    def test_abundant_respects_cpu_cap(self):
        p = self._profile(65536, 4, ResourceTier.ABUNDANT)  # Huge RAM, few CPUs
        cfg = compute_pipeline_config(p)
        assert cfg.max_workers <= 4  # capped by cpu - 2

    def test_moderate_respects_ram_cap(self):
        p = self._profile(700, 16, ResourceTier.MODERATE)  # Low RAM, many CPUs
        cfg = compute_pipeline_config(p)
        assert cfg.max_workers >= 1

    def test_returns_pipeline_config_type(self):
        p = self._profile(8192, 8, ResourceTier.ABUNDANT)
        cfg = compute_pipeline_config(p)
        assert isinstance(cfg, PipelineConfig)

    def test_ocr_threads_always_at_least_2(self):
        for tier in ResourceTier:
            p = self._profile(2000, 2, tier)
            cfg = compute_pipeline_config(p)
            assert cfg.ocr_threads >= 2
