"""Tests for color_analysis module."""

from __future__ import annotations

import pytest

from superbook.processing.color_analysis import (
    _percentile,
    calculate_color_stats,
    decide_global_color_adjustment,
    exclude_outliers,
)
from superbook.models import ColorStats


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([10.0], 50) == 10.0

    def test_median_odd(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_median_even(self):
        result = _percentile([1.0, 2.0, 3.0, 4.0], 50)
        assert abs(result - 2.5) < 1e-9

    def test_boundaries(self):
        assert _percentile([10.0, 20.0, 30.0], 0) == 10.0
        assert _percentile([10.0, 20.0, 30.0], 100) == 30.0


class TestCalculateColorStats:
    def test_white_image(self, white_image):
        stats = calculate_color_stats(white_image, page_number=1)
        assert stats.page_number == 1
        assert stats.paper_r == pytest.approx(255.0, abs=1)
        assert stats.paper_g == pytest.approx(255.0, abs=1)
        assert stats.paper_b == pytest.approx(255.0, abs=1)

    def test_dark_on_light(self, dark_text_on_light_bg):
        stats = calculate_color_stats(dark_text_on_light_bg, page_number=2)
        # Paper should be near 230, ink near 30
        assert stats.paper_r > 200
        assert stats.ink_r < 80

    def test_scanned_page(self, scanned_page_like):
        stats = calculate_color_stats(scanned_page_like, page_number=3)
        # Paper ~(240, 235, 220)
        assert stats.paper_r > 200
        assert stats.paper_g > 200
        assert stats.paper_b > 180
        # Ink ~(20, 20, 25)
        assert stats.ink_r < 60
        assert stats.ink_g < 60


class TestExcludeOutliers:
    def test_small_list_unchanged(self):
        stats = [ColorStats(i, 200, 200, 200, 10, 10, 10) for i in range(2)]
        result = exclude_outliers(stats)
        assert len(result) == 2

    def test_trims_extremes(self):
        # 10 pages, should trim top/bottom 20% (2 each), leaving 6
        stats = [ColorStats(i, float(i * 25), 200, 200, 10, 10, 10) for i in range(10)]
        result = exclude_outliers(stats)
        assert len(result) == 6


class TestDecideGlobalColorAdjustment:
    def test_empty(self):
        param = decide_global_color_adjustment([])
        assert param.scale_r == 1.0
        assert param.offset_r == 0.0

    def test_normal_pages(self, scanned_page_like):
        stats = [calculate_color_stats(scanned_page_like, page_number=i) for i in range(5)]
        param = decide_global_color_adjustment(stats)
        # Scale should be > 1 (stretching paper→255, ink→0)
        assert param.scale_r > 0.8
        assert param.scale_r <= 4.0
        # Paper color should be close to (240, 235, 220)
        assert param.paper_r > 200
