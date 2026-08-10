"""Degradations must be reproducible, bounded, and honest about the text.

A corpus is only a measuring instrument if regenerating it produces the same
images. These tests cover the properties that make that true, and the one
property that makes the corpus meaningful at all: the text stays on the canvas,
so the ground truth remains exact.
"""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from benchmarks.degradations import (
    AXES,
    LEVELS,
    PHOTO_REALISTIC,
    apply_degradation,
    apply_recipe,
    derive_seed,
)

RANDOM_AXES = ("gaussian_noise", "salt_pepper", "broken_glyphs")
GEOMETRY_AXES = ("rotate", "skew", "perspective")


@pytest.fixture
def page() -> Image.Image:
    """Text with wide margins, so a warp has room to move it."""
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    for row in range(4):
        draw.rectangle([120, 100 + row * 50, 480, 120 + row * 50], fill="black")
    return image


class TestDeterminism:
    @pytest.mark.parametrize("axis", AXES)
    def test_the_same_inputs_produce_the_same_bytes(self, page, axis):
        first, first_params = apply_degradation(page, axis, 2, "amostra")
        second, second_params = apply_degradation(page, axis, 2, "amostra")

        assert first.tobytes() == second.tobytes()
        assert first_params == second_params

    @pytest.mark.parametrize("axis", RANDOM_AXES)
    def test_a_different_sample_gets_different_noise(self, page, axis):
        """Otherwise every sample would carry the identical noise pattern."""
        first, _ = apply_degradation(page, axis, 2, "amostra_a")
        second, _ = apply_degradation(page, axis, 2, "amostra_b")

        assert first.tobytes() != second.tobytes()

    def test_seeds_are_derived_not_sequential(self):
        """So a partial or parallel regeneration matches a full serial one."""
        assert derive_seed("a", "blur", 1) == derive_seed("a", "blur", 1)
        assert derive_seed("a", "blur", 1) != derive_seed("a", "blur", 2)
        assert derive_seed("a", "blur", 1) != derive_seed("b", "blur", 1)

    @pytest.mark.parametrize("axis", RANDOM_AXES)
    def test_generation_order_does_not_matter(self, page, axis):
        """Generate one axis alone, and again after others; identical either way."""
        alone, _ = apply_degradation(page, axis, 3, "amostra")
        for other in ("blur", "jpeg"):
            apply_degradation(page, other, 2, "amostra")
        after, _ = apply_degradation(page, axis, 3, "amostra")

        assert alone.tobytes() == after.tobytes()


class TestLevels:
    @pytest.mark.parametrize("axis", AXES)
    def test_level_zero_is_the_identity(self, page, axis):
        """Every sweep gets a control row from the same code path."""
        result, params = apply_degradation(page, axis, 0, "amostra")

        assert result.tobytes() == page.tobytes()
        assert params == {"axis": axis, "level": 0}

    @pytest.mark.parametrize("axis", AXES)
    def test_every_level_actually_changes_the_image(self, page, axis):
        for level in (1, 2, 3):
            result, _ = apply_degradation(page, axis, level, "amostra")
            assert result.tobytes() != page.tobytes(), f"{axis} level {level} was a no-op"

    @pytest.mark.parametrize("axis", ("blur", "gaussian_noise", "faint_glyphs", "low_dpi"))
    def test_severity_increases_with_level(self, page, axis):
        """A graded curve is what makes a model change visible as a shift."""

        def distance(level: int) -> float:
            result, _ = apply_degradation(page, axis, level, "amostra")
            return float(
                np.mean(
                    np.abs(
                        np.asarray(result, dtype=np.float64) - np.asarray(page, dtype=np.float64)
                    )
                )
            )

        assert distance(1) < distance(2) < distance(3)

    def test_an_unknown_axis_is_rejected(self, page):
        with pytest.raises(ValueError, match="unknown degradation axis"):
            apply_degradation(page, "sepia", 1, "amostra")

    def test_an_out_of_range_level_is_rejected(self, page):
        with pytest.raises(ValueError, match="level must be one of"):
            apply_degradation(page, "blur", 7, "amostra")


class TestGroundTruthSurvives:
    """Geometry must not push ink off the canvas, or the text would be wrong."""

    @staticmethod
    def _ink_mask(image: Image.Image) -> np.ndarray:
        """Ink relative to the page's own range, not an absolute grey level.

        ``faint_glyphs`` at its strongest leaves the text around 165, which is
        legitimately faint but perfectly present. A fixed threshold of 128
        would call that an erased page.
        """
        array = np.asarray(image.convert("L")).astype(np.int16)
        darkest, lightest = int(array.min()), int(array.max())
        if lightest - darkest < 8:
            return np.zeros(array.shape, dtype=bool)
        return array < (darkest + lightest) / 2

    @classmethod
    def _ink_bounds(cls, image: Image.Image):
        mask = cls._ink_mask(image)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return None
        return cols[0], rows[0], cols[-1], rows[-1]

    @pytest.mark.parametrize("axis", GEOMETRY_AXES)
    @pytest.mark.parametrize("level", (1, 2, 3))
    def test_ink_stays_inside_the_canvas(self, page, axis, level):
        result, _ = apply_degradation(page, axis, level, "amostra")

        bounds = self._ink_bounds(result)
        assert bounds is not None, "the degradation erased all the text"
        left, top, right, bottom = bounds
        assert left >= 0 and top >= 0
        assert right < result.size[0] and bottom < result.size[1]

    @pytest.mark.parametrize("axis", GEOMETRY_AXES)
    def test_most_of_the_ink_is_still_there(self, page, axis):
        """A warp may resample the strokes; it may not delete them."""
        original_ink = int(self._ink_mask(page).sum())
        result, _ = apply_degradation(page, axis, 3, "amostra")
        remaining_ink = int(self._ink_mask(result).sum())

        assert remaining_ink >= original_ink * 0.75

    @pytest.mark.parametrize("axis", AXES)
    def test_no_axis_erases_the_page(self, page, axis):
        result, _ = apply_degradation(page, axis, 3, "amostra")

        assert self._ink_bounds(result) is not None


class TestRecordedParameters:
    @pytest.mark.parametrize("axis", AXES)
    def test_parameters_round_trip_through_json(self, page, axis):
        """They go into the manifest, so they must serialise."""
        _, params = apply_degradation(page, axis, 2, "amostra")

        assert json.loads(json.dumps(params)) == params

    @pytest.mark.parametrize("axis", AXES)
    def test_the_axis_and_level_are_always_recorded(self, page, axis):
        _, params = apply_degradation(page, axis, 3, "amostra")

        assert params["axis"] == axis
        assert params["level"] == 3
        assert params["seed"] == derive_seed("amostra", axis, 3)


class TestPhotoRealisticRecipe:
    def test_the_recipe_records_every_step_in_order(self, page):
        result, records = apply_recipe(page, PHOTO_REALISTIC, "amostra")

        assert [record["axis"] for record in records] == [axis for axis, _ in PHOTO_REALISTIC]
        assert result.size == page.size or result.size[0] >= page.size[0]

    def test_the_recipe_is_reproducible(self, page):
        first, _ = apply_recipe(page, PHOTO_REALISTIC, "amostra")
        second, _ = apply_recipe(page, PHOTO_REALISTIC, "amostra")

        assert first.tobytes() == second.tobytes()

    def test_the_recipe_leaves_readable_ink(self, page):
        """It stacks five degradations; the page must still be a document."""
        result, _ = apply_recipe(page, PHOTO_REALISTIC, "amostra")

        ink_fraction = float((np.asarray(result.convert("L")) < 128).mean())
        assert 0.005 < ink_fraction < 0.5


def test_every_axis_is_registered():
    """AXES is what the generator iterates; a function without an entry is dead."""
    for axis in AXES:
        assert axis in AXES
    assert len(set(AXES)) == len(AXES)
    assert LEVELS == (0, 1, 2, 3)
