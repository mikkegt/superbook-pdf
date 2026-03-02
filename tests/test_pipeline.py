"""Tests for pipeline helpers."""

from __future__ import annotations

import numpy as np

from superbook.models import BoundingBox
from superbook.pipeline import _equalize_crop_regions, _trim_edges


class TestTrimEdges:
    def test_reduces_size(self):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        trimmed = _trim_edges(img)
        assert trimmed.shape[0] < 1000
        assert trimmed.shape[1] < 800

    def test_small_image_unchanged(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        trimmed = _trim_edges(img)
        assert trimmed.shape == (5, 5, 3)


class TestEqualizeCropRegions:
    def test_y_unified(self):
        odd = BoundingBox(100, 200, 300, 400)
        even = BoundingBox(120, 150, 280, 500)
        odd_out, even_out = _equalize_crop_regions(odd, even, margin_percent=7)
        # Y should be unified: both start at min(200, 150) = 150
        assert odd_out.y == even_out.y

    def test_dimensions_match(self):
        odd = BoundingBox(100, 100, 300, 400)
        even = BoundingBox(150, 100, 250, 450)
        odd_out, even_out = _equalize_crop_regions(odd, even, margin_percent=7)
        assert odd_out.width == even_out.width
        assert odd_out.height == even_out.height

    def test_clamped_to_canvas(self):
        # Very large crop that could exceed canvas
        odd = BoundingBox(0, 0, 4900, 7000)
        even = BoundingBox(0, 0, 4900, 7000)
        odd_out, even_out = _equalize_crop_regions(odd, even, margin_percent=10)
        assert odd_out.width <= 4960
        assert odd_out.height <= 7016
