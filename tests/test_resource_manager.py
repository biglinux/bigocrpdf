"""Tests for resource_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf.services.rapidocr_service.resource_manager import (
    PipelineConfig,
    ResourceProfile,
    ResourceTier,
    compute_pipeline_config,
    detect_resources,
    enforce_pdf_resource_limits,
    estimate_page_megapixels,
    select_pdf_page_render_dpi,
    select_render_dpi_for_page,
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
        with pytest.raises(AttributeError):
            p.available_ram_mb = 0  # pyright: ignore[reportAttributeAccessIssue]


class TestDetectResources:
    def test_with_psutil(self):
        mock_mem = MagicMock()
        mock_mem.available = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_mem.total = 16 * 1024 * 1024 * 1024  # 16 GB

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch("builtins.open", side_effect=OSError):
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
            with patch("builtins.open", side_effect=OSError):
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
            with patch("builtins.open", side_effect=OSError):
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
        assert cfg.gc_after_page is True
        assert cfg.downscale_probmap == 1024

    def test_moderate_balanced(self):
        p = self._profile(4096, 8, ResourceTier.MODERATE)
        cfg = compute_pipeline_config(p)
        assert 1 <= cfg.max_workers <= 6
        assert cfg.chunk_size == 8
        assert cfg.gc_after_page is False
        assert cfg.downscale_probmap == 1536

    def test_abundant_high_performance(self):
        p = self._profile(16384, 16, ResourceTier.ABUNDANT)
        cfg = compute_pipeline_config(p)
        assert cfg.max_workers > 1
        assert cfg.max_workers <= 12
        assert cfg.gc_after_page is False
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


class TestPdfResourceLimits:
    def test_estimates_page_megapixels_from_points_and_dpi(self):
        megapixels = estimate_page_megapixels(612, 792, 300)
        assert 8.0 < megapixels < 9.0

    def test_accepts_pdf_within_limits(self):
        config = type("Config", (), {"max_pdf_pages": 10, "dpi": 300})()
        enforce_pdf_resource_limits(2, config)

    def test_accepts_embedded_image_within_limit(self):
        config = type(
            "Config",
            (),
            {"max_pdf_pages": 10, "max_image_megapixels": 128, "dpi": 300},
        )()
        enforce_pdf_resource_limits(1, config, image_dimensions=[(1, 4000, 3000)])

    def test_page_count_limit_is_disabled_by_default(self):
        config = type("Config", (), {"max_pdf_pages": 0, "dpi": 300})()
        enforce_pdf_resource_limits(10_000, config)

    def test_rejects_too_many_pages(self):
        config = type("Config", (), {"max_pdf_pages": 1, "dpi": 300})()
        with pytest.raises(ValueError, match="configured limit is 1"):
            enforce_pdf_resource_limits(2, config)

    def test_rejects_too_large_embedded_image(self):
        config = type(
            "Config",
            (),
            {"max_pdf_pages": 10, "max_image_megapixels": 128, "dpi": 300},
        )()
        with pytest.raises(ValueError, match=r"page 1.*200\.0 MP.*128\.0 MP"):
            enforce_pdf_resource_limits(1, config, image_dimensions=[(1, 20_000, 10_000)])

    def test_render_dpi_stays_preferred_when_within_budget(self):
        assert select_render_dpi_for_page(612, 792, 300, 45) == 300

    def test_render_dpi_is_reduced_to_megapixel_budget(self):
        dpi = select_render_dpi_for_page(612, 792, 300, 1, min_dpi=72)
        assert 90 <= dpi <= 110

    def test_render_dpi_stops_at_floor_when_floor_fits_budget(self):
        floor_budget = estimate_page_megapixels(612, 792, 150)
        assert select_render_dpi_for_page(612, 792, 300, floor_budget, min_dpi=150) == 150

    def test_render_dpi_rejects_page_when_floor_exceeds_budget(self):
        with pytest.raises(ValueError, match=r"minimum 150 DPI.*configured limit is 1\.0 MP"):
            select_render_dpi_for_page(612, 792, 300, 1, min_dpi=150)

    def test_render_dpi_downscales_photo_sized_page_box(self):
        """A photo PDF placing one point per source pixel must still render."""
        assert select_render_dpi_for_page(3864, 2814, 300, 45) == 146

    def test_render_dpi_rejects_invalid_page_dimensions(self):
        with pytest.raises(ValueError, match="invalid dimensions"):
            select_render_dpi_for_page(0, 792, 300, 45)

    def test_pdf_render_budget_accounts_for_user_unit(self, tmp_path):
        import pikepdf

        pdf_path = tmp_path / "large-user-unit.pdf"
        pdf = pikepdf.Pdf.new()
        page = pdf.add_blank_page(page_size=(612, 792))
        page["/UserUnit"] = 10
        pdf.save(pdf_path)

        with pytest.raises(ValueError, match=r"minimum 100 DPI.*configured limit is 45\.0 MP"):
            select_pdf_page_render_dpi(pdf_path, 1, 300, 45)

    def test_pdf_render_budget_rejects_invalid_page_number(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid PDF page number"):
            select_pdf_page_render_dpi(tmp_path / "unused.pdf", 0, 300, 45)
