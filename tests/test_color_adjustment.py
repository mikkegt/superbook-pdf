"""Tests for color_adjustment module."""

from __future__ import annotations

import numpy as np

from superbook.models import GlobalColorParam
from superbook.processing.color_adjustment import apply_global_color_adjustment


def _identity_param() -> GlobalColorParam:
    """A param that leaves the image mostly unchanged (scale=1, offset=0)."""
    return GlobalColorParam(
        scale_r=1.0, scale_g=1.0, scale_b=1.0,
        offset_r=0.0, offset_g=0.0, offset_b=0.0,
        ghost_suppress_lum_threshold=200,
        paper_r=240, paper_g=235, paper_b=220,
        sat_threshold=55,
        color_dist_threshold=35,
        bleed_hue_min=20.0,
        bleed_hue_max=65.0,
        bleed_value_min=0.35,
    )


class TestApplyGlobalColorAdjustment:
    def test_identity_preserves_dark_pixels(self):
        """Dark ink pixels should remain dark under identity correction."""
        img = np.full((10, 10, 3), 30, dtype=np.uint8)
        param = _identity_param()
        result = apply_global_color_adjustment(img, param)
        # Dark pixels shouldn't be whitened (lum < ghostThr)
        assert result.mean() < 100

    def test_paper_pixels_whitened(self):
        """Pixels near paper color should become whiter via smoothstep."""
        img = np.full((10, 10, 3), dtype=np.uint8, fill_value=0)
        img[:, :, 0] = 240
        img[:, :, 1] = 235
        img[:, :, 2] = 220
        param = _identity_param()
        param.ghost_suppress_lum_threshold = 100  # low threshold → more whitening
        result = apply_global_color_adjustment(img, param)
        # Should be pushed towards 255
        assert result[:, :, 0].mean() > 240

    def test_pastel_pink_suppressed(self):
        """Light pastel pink pixels should be converted to white."""
        img = np.full((10, 10, 3), dtype=np.uint8, fill_value=0)
        # Pastel pink: high lum, low sat, hue ≤ 40
        img[:, :, 0] = 250  # R
        img[:, :, 1] = 240  # G
        img[:, :, 2] = 238  # B
        param = _identity_param()
        result = apply_global_color_adjustment(img, param)
        assert np.all(result == 255)

    def test_output_shape_preserved(self, dark_text_on_light_bg):
        param = _identity_param()
        result = apply_global_color_adjustment(dark_text_on_light_bg, param)
        assert result.shape == dark_text_on_light_bg.shape
        assert result.dtype == np.uint8
