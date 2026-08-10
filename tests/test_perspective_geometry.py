"""Tests for perspective geometry functions."""

import numpy as np

from bigocrpdf.services.perspective_document import (
    _is_convex_quad,
    four_point_transform,
    order_points,
)


class TestOrderPoints:
    """Tests for order_points."""

    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")
        result = order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])  # top-left
        np.testing.assert_array_almost_equal(result[1], [100, 0])  # top-right
        np.testing.assert_array_almost_equal(result[2], [100, 100])  # bottom-right
        np.testing.assert_array_almost_equal(result[3], [0, 100])  # bottom-left

    def test_shuffled(self):
        pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype="float32")
        result = order_points(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[1], [100, 0])
        np.testing.assert_array_almost_equal(result[2], [100, 100])
        np.testing.assert_array_almost_equal(result[3], [0, 100])

    def test_non_rectangular(self):
        pts = np.array([[10, 5], [90, 10], [85, 95], [5, 90]], dtype="float32")
        result = order_points(pts)
        # top-left: smallest sum = (5,90)→95 vs (10,5)→15 → (10,5)
        np.testing.assert_array_almost_equal(result[0], [10, 5])
        np.testing.assert_array_almost_equal(result[2], [85, 95])

    def test_output_shape(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
        result = order_points(pts)
        assert result.shape == (4, 2)
        assert result.dtype == np.float32


class TestFourPointTransform:
    """Tests for four_point_transform."""

    def test_identity_transform(self):
        # Create a 100x100 test image with a known pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = 255  # White square in center
        pts = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype="float32")
        result = four_point_transform(img, pts)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_crop_region(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 128
        pts = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype="float32")
        result = four_point_transform(img, pts)
        # Result should be approximately 100x100
        assert abs(result.shape[0] - 100) <= 1
        assert abs(result.shape[1] - 100) <= 1

    def test_output_is_ndarray(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype="float32")
        result = four_point_transform(img, pts)
        assert isinstance(result, np.ndarray)


class TestIsConvexQuad:
    """Corner validation must answer, not raise.

    NumPy 2.0 removed the 2-D cross product: ``np.cross`` on two-element
    vectors now raises instead of returning the scalar z-component. Every call
    here raised, so refined corners were never validated and the exception
    propagated out of the whole perspective cascade -- caught far above, which
    turned a live feature into a silent no-op. Measured on 50 scanned
    Portuguese pages, the failure appeared only as a log line.
    """

    SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")

    def test_a_square_is_convex(self):
        assert _is_convex_quad(self.SQUARE) is True

    def test_a_perspective_quad_is_convex(self):
        """The shape this function exists to accept: a photographed page."""
        quad = np.array([[10, 5], [190, 20], [180, 140], [20, 130]], dtype="float32")

        assert _is_convex_quad(quad) is True

    def test_a_bowtie_is_not_convex(self):
        """Two corners swapped is the defect it exists to reject."""
        bowtie = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype="float32")

        assert _is_convex_quad(bowtie) is False

    def test_a_reflex_corner_is_not_convex(self):
        arrowhead = np.array([[0, 0], [100, 0], [50, 40], [0, 100]], dtype="float32")

        assert _is_convex_quad(arrowhead) is False

    def test_integer_corners_are_accepted(self):
        """Contours arrive as int32, where a wrong dtype would silently overflow."""
        assert _is_convex_quad(self.SQUARE.astype(np.int32)) is True
