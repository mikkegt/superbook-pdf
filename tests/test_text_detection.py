"""Tests for text_detection module."""

from __future__ import annotations


from superbook.models import BoundingBox, PageBoundingBox
from superbook.processing.text_detection import (
    decide_group_crop_region,
    detect_text_bounding_box,
)


class TestDetectTextBoundingBox:
    def test_white_image_returns_empty(self, white_image):
        bbox = detect_text_bounding_box(white_image)
        assert bbox.width == 0 and bbox.height == 0

    def test_dark_rect_detected(self, dark_text_on_light_bg):
        bbox = detect_text_bounding_box(dark_text_on_light_bg)
        # The dark rectangle is at (40,50)-(160,150)
        assert bbox.width > 0
        assert bbox.height > 0
        # Should roughly cover the dark region
        assert bbox.x <= 50
        assert bbox.y <= 60
        assert bbox.right >= 140
        assert bbox.bottom >= 140

    def test_scanned_page(self, scanned_page_like):
        bbox = detect_text_bounding_box(scanned_page_like)
        assert bbox.width > 300
        assert bbox.height > 400


class TestDecideGroupCropRegion:
    def test_empty_returns_zero(self):
        result = decide_group_crop_region([])
        assert result.width == 0

    def test_single_page(self):
        pbb = PageBoundingBox(1, BoundingBox(100, 50, 300, 600))
        result = decide_group_crop_region([pbb])
        assert result.x == 100
        assert result.y == 50

    def test_outlier_excluded(self):
        # 9 consistent pages + 1 outlier
        consistent = [
            PageBoundingBox(i, BoundingBox(100, 50, 300, 600))
            for i in range(9)
        ]
        outlier = PageBoundingBox(99, BoundingBox(10, 10, 480, 690))
        all_pages = consistent + [outlier]
        result = decide_group_crop_region(all_pages)
        # Should match the consistent pages, not the outlier
        assert result.x == 100
        assert result.y == 50

    def test_all_blank_returns_zero(self):
        blanks = [PageBoundingBox(i, BoundingBox(0, 0, 0, 0)) for i in range(5)]
        result = decide_group_crop_region(blanks)
        assert result.width == 0
