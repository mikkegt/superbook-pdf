"""Tests for deskew module."""

from __future__ import annotations

import numpy as np

from superbook.processing.deskew import _prepare_otsu_image


class TestPrepareOtsuImage:
    def test_output_is_binary(self, dark_text_on_light_bg):
        otsu = _prepare_otsu_image(dark_text_on_light_bg)
        assert otsu.ndim == 2
        unique = set(np.unique(otsu))
        assert unique.issubset({0, 255})

    def test_shape_preserved(self, scanned_page_like):
        otsu = _prepare_otsu_image(scanned_page_like)
        assert otsu.shape == scanned_page_like.shape[:2]
