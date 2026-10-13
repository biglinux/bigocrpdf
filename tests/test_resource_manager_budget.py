"""Memory budgeting: cgroup limits, chunk sizing, and the tier arithmetic.

The cgroup v2 branches are the only thing standing between a container or
Flatpak with a memory limit and the OOM killer, so they run here against a fake
cgroup filesystem.
"""

import builtins

import pytest

from bigocrpdf.services.ocr.resource_manager import (
    ResourceProfile,
    ResourceTier,
    adjust_chunk_size,
    compute_pipeline_config,
    detect_resources,
    estimate_page_memory_mb,
)

A4 = (595.0, 842.0)
A0 = (3370.0, 4780.0)


class TestAdjustChunkSize:
    """Chosen so the largest page in the document still fits in memory."""

    def test_no_pages_leaves_the_base_chunk(self):
        assert adjust_chunk_size(8, [], 8000) == 8

    def test_plenty_of_memory_leaves_the_base_chunk(self):
        assert adjust_chunk_size(8, [A4] * 20, 64000) == 8

    def test_the_result_never_exceeds_the_base(self):
        """The pipeline's own configuration is the ceiling."""
        assert adjust_chunk_size(4, [A4], 1_000_000) == 4

    def test_the_largest_page_decides_for_the_whole_document(self):
        """One oversized page shrinks the chunk for every page in the document,
        including the ninety-nine ordinary ones next to it."""
        mixed = [A4] * 99 + [A0]

        assert adjust_chunk_size(8, mixed, 4000) < adjust_chunk_size(8, [A4] * 99, 4000)

    def test_it_never_returns_less_than_one(self):
        """A zero chunk would process nothing at all."""
        assert adjust_chunk_size(8, [A0] * 4, 100) == 1

    @pytest.mark.parametrize("dimensions", [(0.0, 0.0), (-595.0, 842.0)])
    def test_degenerate_dimensions_do_not_raise(self, dimensions):
        assert adjust_chunk_size(8, [dimensions], 8000) >= 1

    def test_memory_scales_with_the_square_of_dpi(self):
        """Doubling DPI quadruples the pixels, and so the budget."""
        low = estimate_page_memory_mb(*A4, 150)
        high = estimate_page_memory_mb(*A4, 300)

        assert high / low == pytest.approx(4.0, rel=0.01)

    def test_the_chunk_fits_sixty_percent_of_available_memory(self):
        """The budget property itself, so the assertion does not restate the
        formula's rounding: the chosen chunk fits, and one more would not."""
        per_page = estimate_page_memory_mb(*A4, 300)
        available = 4000

        chunk = adjust_chunk_size(50, [A4] * 50, available)

        assert chunk * per_page <= 0.6 * available
        assert (chunk + 1) * per_page > 0.6 * available

    def test_more_memory_allows_a_larger_chunk(self):
        small = adjust_chunk_size(50, [A4] * 50, 2000)
        large = adjust_chunk_size(50, [A4] * 50, 8000)

        assert large > small


class _FakeCgroupFS:
    """Serves the given cgroup v2 files.

    Other cgroup paths and ``/proc/meminfo`` raise ``OSError``; anything else
    goes to the real ``open``.
    """

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self._real_open = builtins.open

    def __call__(self, path, *args, **kwargs):
        name = str(path)
        if name in self.files:
            import io

            return io.StringIO(self.files[name])
        if name.startswith("/sys/fs/cgroup/") or name == "/proc/meminfo":
            raise OSError(f"no such file: {name}")
        return self._real_open(path, *args, **kwargs)


@pytest.fixture
def host_memory(monkeypatch):
    """Pin the host to 16 GB total / 12 GB available, before cgroup clamping."""

    class _Memory:
        total = 16 * 1024 * 1024 * 1024
        available = 12 * 1024 * 1024 * 1024

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Memory())
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(8)))


# The process sits in a nested cgroup; the hierarchy root carries no limit.
OWN_CGROUP = "/sys/fs/cgroup/user.slice/app.slice/bigocr.scope"


def _with_cgroup(monkeypatch, **files: str) -> ResourceProfile:
    files = {"/proc/self/cgroup": "0::/user.slice/app.slice/bigocr.scope\n", **files}
    monkeypatch.setattr(builtins, "open", _FakeCgroupFS(files))
    return detect_resources()


class TestCgroupV2Limits:
    def test_a_memory_limit_clamps_the_available_ram(self, monkeypatch, host_memory):
        """1 GiB limit, 200 MiB already used, on a 16 GB host."""
        profile = _with_cgroup(
            monkeypatch,
            **{
                f"{OWN_CGROUP}/memory.max": str(1024 * 1024 * 1024),
                f"{OWN_CGROUP}/memory.current": str(200 * 1024 * 1024),
            },
        )

        assert profile.total_ram_mb == 1024
        assert profile.available_ram_mb == 1024 - 200
        assert profile.tier is ResourceTier.CONSTRAINED

    def test_the_smallest_ancestor_limit_applies(self, monkeypatch, host_memory):
        """A slice limit binds every scope below it, even unlimited ones."""
        slice_dir = "/sys/fs/cgroup/user.slice"
        profile = _with_cgroup(
            monkeypatch,
            **{
                f"{OWN_CGROUP}/memory.max": "max",
                f"{OWN_CGROUP}/memory.current": str(100 * 1024 * 1024),
                f"{slice_dir}/memory.max": str(2048 * 1024 * 1024),
                f"{slice_dir}/memory.current": str(1536 * 1024 * 1024),
            },
        )

        assert profile.total_ram_mb == 2048
        assert profile.available_ram_mb == 512

    def test_an_unlimited_cgroup_leaves_the_host_values(self, monkeypatch, host_memory):
        """``max`` is the literal cgroup writes when there is no limit."""
        profile = _with_cgroup(monkeypatch, **{f"{OWN_CGROUP}/memory.max": "max\n"})

        assert profile.total_ram_mb == 16 * 1024
        assert profile.tier is ResourceTier.ABUNDANT

    def test_usage_above_the_limit_never_goes_negative(self, monkeypatch, host_memory):
        profile = _with_cgroup(
            monkeypatch,
            **{
                f"{OWN_CGROUP}/memory.max": str(512 * 1024 * 1024),
                f"{OWN_CGROUP}/memory.current": str(900 * 1024 * 1024),
            },
        )

        assert profile.available_ram_mb == 0
        assert profile.tier is ResourceTier.CONSTRAINED

    @pytest.mark.parametrize("contents", ["", "abc", "1e9", "   "])
    def test_unparseable_contents_fall_back_to_the_host(self, monkeypatch, host_memory, contents):
        profile = _with_cgroup(monkeypatch, **{f"{OWN_CGROUP}/memory.max": contents})

        assert profile.total_ram_mb == 16 * 1024

    def test_a_limit_larger_than_the_host_does_not_inflate_it(self, monkeypatch, host_memory):
        profile = _with_cgroup(monkeypatch, **{f"{OWN_CGROUP}/memory.max": str(64 * 1024**3)})

        assert profile.total_ram_mb == 16 * 1024

    def test_no_cgroup_files_is_the_ordinary_case(self, monkeypatch, host_memory):
        profile = _with_cgroup(monkeypatch)

        assert profile.total_ram_mb == 16 * 1024
        assert profile.available_ram_mb == 12 * 1024


class TestPipelineConfigEdges:
    def _profile(self, available_mb: int, tier: ResourceTier, cpu_count: int = 8):
        return ResourceProfile(
            available_ram_mb=available_mb,
            total_ram_mb=max(available_mb, 1024),
            cpu_count=cpu_count,
            tier=tier,
        )

    def test_a_negative_budget_still_yields_a_usable_config(self):
        """550 MB of overhead against 500 MB available leaves -50."""
        config = compute_pipeline_config(self._profile(500, ResourceTier.MODERATE))

        assert config.max_workers >= 1
        assert config.chunk_size >= 1

    def test_the_constrained_tier_serialises_and_collects(self):
        config = compute_pipeline_config(self._profile(1500, ResourceTier.CONSTRAINED))

        assert config.max_workers == 1
        assert config.chunk_size == 4
        assert config.gc_after_page is True
        assert config.downscale_probmap == 1024

    def test_the_moderate_tier_caps_workers_at_what_the_ocr_takes(self):
        config = compute_pipeline_config(self._profile(5000, ResourceTier.MODERATE, cpu_count=32))

        assert config.max_workers == 2
        assert config.downscale_probmap == 1536

    def test_the_abundant_tier_caps_workers_and_chunk(self):
        config = compute_pipeline_config(self._profile(64000, ResourceTier.ABUNDANT, cpu_count=32))

        assert config.max_workers == 2
        assert config.chunk_size == 20
        assert config.gc_after_page is False

    @pytest.mark.parametrize("cpu_count", [1, 2, 3])
    def test_tiny_cpu_counts_never_produce_zero_workers(self, cpu_count):
        """Two cores are reserved, so a 1-core host must not end up at -1."""
        config = compute_pipeline_config(
            self._profile(64000, ResourceTier.ABUNDANT, cpu_count=cpu_count)
        )

        assert config.max_workers >= 1

    @pytest.mark.parametrize("tier", list(ResourceTier))
    def test_ocr_threads_are_never_below_two(self, tier):
        config = compute_pipeline_config(self._profile(4000, tier, cpu_count=1))

        assert config.ocr_threads >= 2

    def test_the_per_worker_cost_governs_the_worker_count(self):
        """Pins the 0.7 safety factor and the 900 MB per-worker estimate.

        These are inline literals; editing one should fail loudly here rather
        than quietly change how much memory the pipeline commits to.
        """
        config = compute_pipeline_config(self._profile(2000, ResourceTier.MODERATE, cpu_count=32))

        usable = 2000 - 400 - 150
        assert config.max_workers == min(int(usable * 0.7 / 900), 2) == 1

    @pytest.mark.parametrize("tier", list(ResourceTier))
    def test_a_requested_worker_count_is_honoured(self, tier):
        config = compute_pipeline_config(self._profile(64000, tier, cpu_count=32), 5)

        assert config.max_workers == 5
