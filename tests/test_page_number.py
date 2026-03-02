"""Tests for OCR page_number module."""

from __future__ import annotations

import numpy as np

from superbook.ocr.page_number import _extract_margin_regions, _prepare_for_ocr


class TestPrepareForOcr:
    def test_output_is_binary(self, dark_text_on_light_bg):
        otsu = _prepare_for_ocr(dark_text_on_light_bg)
        assert otsu.ndim == 2
        unique = set(np.unique(otsu))
        assert unique.issubset({0, 255})


class TestExtractMarginRegions:
    def test_returns_four_regions(self, scanned_page_like):
        regions = _extract_margin_regions(scanned_page_like)
        assert len(regions) == 4

    def test_regions_not_empty(self, scanned_page_like):
        for region, rect in _extract_margin_regions(scanned_page_like):
            assert region.size > 0
            assert rect[2] > 0 and rect[3] > 0
