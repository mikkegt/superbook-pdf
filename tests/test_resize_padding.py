"""Tests for resize_padding module."""

from __future__ import annotations

import numpy as np

from superbook.processing.resize_padding import (
    _make_gradient_background,
    resize_and_pad,
)


class TestMakeGradientBackground:
    def test_shape_and_dtype(self):
        bg = _make_gradient_background(
            100, 200,
            (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (128, 128, 128),
        )
        assert bg.shape == (200, 100, 3)
        assert bg.dtype == np.uint8

    def test_corners_match(self):
        tl, tr, bl, br = (200, 200, 200), (100, 100, 100), (50, 50, 50), (150, 150, 150)
        bg = _make_gradient_background(50, 50, tl, tr, bl, br)
        # Top-left corner
        np.testing.assert_array_equal(bg[0, 0], [200, 200, 200])
        # Top-right corner
        np.testing.assert_array_equal(bg[0, -1], [100, 100, 100])
        # Bottom-left corner
        np.testing.assert_array_equal(bg[-1, 0], [50, 50, 50])
        # Bottom-right corner
        np.testing.assert_array_equal(bg[-1, -1], [150, 150, 150])


class TestResizeAndPad:
    def test_output_shape(self, scanned_page_like):
        result = resize_and_pad(scanned_page_like, 4960, 7016)
        assert result.shape == (7016, 4960, 3)
        assert result.dtype == np.uint8

    def test_small_to_large(self):
        small = np.full((100, 50, 3), 128, dtype=np.uint8)
        result = resize_and_pad(small, 200, 300)
        assert result.shape == (300, 200, 3)

    def test_shift(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = resize_and_pad(img, 200, 200, shift_x=10, shift_y=5)
        assert result.shape == (200, 200, 3)
