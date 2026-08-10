"""Ground truth must describe what the generator actually drew.

``two_columns`` and ``table`` used to declare their ground truth as a literal
separate from the drawing code, and the two had drifted: the declared text had
different line breaks from the text on the page. Both samples therefore carried
a permanent non-zero error floor that no OCR improvement could ever clear,
which quietly made them useless as canaries.

The fix was structural -- ground truth is now derived from the segments the
generator draws -- and these tests are what keeps it that way.
"""

from pathlib import Path

import pytest

from benchmarks.make_synthetic_ocr_fixtures import (
    COLUMN_SEGMENTS,
    SAMPLES,
    TABLE_ROWS,
    sample_ground_truth,
)
from benchmarks.tests.test_prepare_benchmark_datasets import requires_sample_fonts

SAMPLE_IDS = [sample_id for sample_id, _, _, _ in SAMPLES]


def _ground_truth(sample_id: str) -> str:
    for candidate, _language, _tags, declared in SAMPLES:
        if candidate == sample_id:
            return sample_ground_truth(candidate, declared)
    raise AssertionError(f"unknown sample {sample_id!r}")


class TestDerivedGroundTruth:
    def test_every_column_segment_appears_in_the_ground_truth(self):
        truth = _ground_truth("two_columns")

        for segment, _position in COLUMN_SEGMENTS:
            for line in segment.splitlines():
                assert line in truth

    def test_every_table_cell_appears_in_the_ground_truth(self):
        truth = _ground_truth("table")

        for row in TABLE_ROWS:
            for cell in row:
                assert cell in truth

    def test_table_rows_read_across_not_down(self):
        """Row-major is how a reader meets a table, and how OCR emits it."""
        truth = _ground_truth("table")

        assert truth.splitlines()[0] == "Produto Quantidade Valor"

    def test_column_line_breaks_match_the_drawing(self):
        """The exact drift that made these samples unusable."""
        truth = _ground_truth("two_columns")

        assert "Coluna esquerda\ncom texto." in truth
        assert "Coluna esquerda com texto." not in truth

    @pytest.mark.parametrize("sample_id", ["pt_accented", "pt_legal", "cjk", "small_text"])
    def test_plain_samples_keep_their_declared_text(self, sample_id):
        declared = next(text for sid, _, _, text in SAMPLES if sid == sample_id)

        assert _ground_truth(sample_id) == declared


class TestSampleDeclarations:
    def test_sample_ids_are_unique(self):
        assert len(set(SAMPLE_IDS)) == len(SAMPLE_IDS)

    @pytest.mark.parametrize("sample_id", SAMPLE_IDS)
    def test_no_sample_has_empty_ground_truth(self, sample_id):
        """An empty ground truth would score every read as a total failure."""
        assert _ground_truth(sample_id).strip()

    def test_ambiguous_layouts_are_tagged(self):
        """So the harness can gate them on the order-insensitive metric."""
        for sample_id, _language, tags, _text in SAMPLES:
            if sample_id in {"two_columns", "table"}:
                assert "layout_order_uncertain" in tags

    def test_unsupported_scripts_are_tagged(self):
        """PP-OCRv6 cannot read these; they are diagnostic, not gating."""
        tags_by_id = {sample_id: tags for sample_id, _, tags, _ in SAMPLES}

        assert "unsupported_ppocrv6" in tags_by_id["greek"]
        assert "unsupported_ppocrv6" in tags_by_id["arabic"]


class TestTiers:
    """The curated subsets that make the corpus runnable."""

    def test_the_tiers_grow_monotonically(self):
        from benchmarks.make_synthetic_ocr_fixtures import TIERS

        smoke, gate, full = TIERS["smoke"], TIERS["gate"], TIERS["full"]

        assert set(smoke["axes"]) <= set(gate["axes"]) <= set(full["axes"])
        assert set(smoke["levels"]) <= set(gate["levels"]) <= set(full["levels"])

    def test_only_the_full_tier_includes_the_diagnostic_severity(self):
        """Level 3 shows where the cliff is; it is not a pass/fail line."""
        from benchmarks.make_synthetic_ocr_fixtures import TIERS

        assert 3 not in TIERS["smoke"]["levels"]
        assert 3 not in TIERS["gate"]["levels"]
        assert 3 in TIERS["full"]["levels"]

    def test_the_full_tier_covers_every_sample(self):
        from benchmarks.make_synthetic_ocr_fixtures import TIERS

        assert TIERS["full"]["samples"] is None

    def test_the_smoke_tier_stays_small_enough_to_re_run(self):
        """One sample, six axes, one level, plus the composite: fourteen PDFs."""
        from benchmarks.make_synthetic_ocr_fixtures import TIERS

        spec = TIERS["smoke"]
        variants = len(spec["samples"]) * len(spec["axes"]) * len(spec["levels"]) + len(
            spec["samples"]
        )
        assert variants <= 16


@requires_sample_fonts
class TestDegradedGeneration:
    """Variants share the clean sample's ground truth, and say what they are."""

    @staticmethod
    def _clean_row(tmp_path: Path) -> list[dict]:
        from benchmarks.make_synthetic_ocr_fixtures import generate_rows

        return generate_rows(tmp_path)

    def test_variants_reuse_the_clean_ground_truth(self, tmp_path: Path):
        """The degradation touched pixels only, so the expected text is unchanged.

        Rewriting it per variant is exactly how a corpus stops being a
        measuring instrument.
        """
        from benchmarks.make_synthetic_ocr_fixtures import generate_degraded_rows

        clean = self._clean_row(tmp_path)
        variants = generate_degraded_rows(tmp_path, clean, "smoke")

        by_id = {row["id"]: row for row in clean}
        assert variants
        for variant in variants:
            origin = variant["id"].split("__")[0]
            assert variant["gt_text"] == by_id[origin]["gt_text"]

    def test_every_variant_gets_its_own_image_and_pdf(self, tmp_path: Path):
        from benchmarks.make_synthetic_ocr_fixtures import generate_degraded_rows

        variants = generate_degraded_rows(tmp_path, self._clean_row(tmp_path), "smoke")

        for variant in variants:
            assert (tmp_path / variant["image"]).is_file()
            assert (tmp_path / variant["pdf"]).is_file()
        assert len({v["image"] for v in variants}) == len(variants)

    def test_the_degradation_parameters_are_recorded(self, tmp_path: Path):
        """A metric shift has to be traceable to the axis that caused it."""
        from benchmarks.make_synthetic_ocr_fixtures import generate_degraded_rows

        variants = generate_degraded_rows(tmp_path, self._clean_row(tmp_path), "smoke")

        for variant in variants:
            assert variant["source"]["degradation"]

    def test_variants_carry_axis_and_tier_tags(self, tmp_path: Path):
        """The harness selects rows by tag, so an untagged row is unreachable."""
        from benchmarks.make_synthetic_ocr_fixtures import generate_degraded_rows

        variants = generate_degraded_rows(tmp_path, self._clean_row(tmp_path), "smoke")

        assert all("tier:smoke" in variant["tags"] for variant in variants)
        assert any(tag.startswith("axis:") for v in variants for tag in v["tags"])
        assert any("photo_geometry" in variant["tags"] for variant in variants)

    def test_generation_is_reproducible(self, tmp_path: Path):
        """Same inputs, same bytes -- otherwise the corpus cannot be a baseline."""
        from benchmarks.make_synthetic_ocr_fixtures import generate_degraded_rows

        clean = self._clean_row(tmp_path)
        first = generate_degraded_rows(tmp_path, clean, "smoke")
        digests = {row["id"]: (tmp_path / row["image"]).read_bytes()[:2048] for row in first}

        second = generate_degraded_rows(tmp_path, clean, "smoke")

        for row in second:
            assert (tmp_path / row["image"]).read_bytes()[:2048] == digests[row["id"]]
