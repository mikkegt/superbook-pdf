"""Color statistics computation and outlier exclusion.

Ports the C# ``CalculateColorStats``, ``ExcludeOutliers``, and
``DecideGlobalColorAdjustment`` methods.
"""

from __future__ import annotations

import math

import numpy as np

from superbook.config import (
    BLEED_HUE_MAX,
    BLEED_HUE_MIN,
    BLEED_VALUE_MIN,
    COLOR_DIST_THRESHOLD,
    INK_PERCENTILE,
    MAD_MULTIPLIER,
    OUTLIER_TRIM_RATIO,
    PAPER_PERCENTILE,
    SAMPLE_STEP,
    SAT_THRESHOLD,
    SCALE_CLAMP_MAX,
    SCALE_CLAMP_MIN,
)
from superbook.models import ColorStats, GlobalColorParam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Linearly-interpolated percentile (0–100 scale), matching the C# helper."""
    if not values:
        return 0.0
    values = sorted(values)
    rank = (p / 100.0) * (len(values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


# ---------------------------------------------------------------------------
# Per-page color statistics
# ---------------------------------------------------------------------------

def calculate_color_stats(image: np.ndarray, page_number: int = 0) -> ColorStats:
    """Compute paper / ink RGB averages from an RGB uint8 image.

    *image* is expected to be ``(H, W, 3)`` in **RGB** channel order.
    """
    # Sub-sample every SAMPLE_STEP pixels in both axes
    sampled = image[::SAMPLE_STEP, ::SAMPLE_STEP]  # (h', w', 3)
    r = sampled[:, :, 0].astype(np.float64)
    g = sampled[:, :, 1].astype(np.float64)
    b = sampled[:, :, 2].astype(np.float64)

    # BT.601 luminance (float version, matching C# cast to int)
    lum = (0.299 * r + 0.587 * g + 0.114 * b + 0.5).astype(np.int32)
    lum = np.clip(lum, 0, 255)

    # Build luminance histogram
    hist = np.bincount(lum.ravel(), minlength=256)
    total = hist.sum()

    # Find 5th percentile (ink) and 95th percentile (paper) luminance
    cum = np.cumsum(hist)
    low_target = int(total * INK_PERCENTILE)
    high_target = int(total * PAPER_PERCENTILE)

    low_lum = 0
    for i in range(256):
        if cum[i] >= low_target:
            low_lum = i
            break

    high_lum = 255
    for i in range(256):
        if cum[i] >= high_target:
            high_lum = i
            break

    # Average RGB for paper (lum >= high_lum) and ink (lum <= low_lum)
    paper_mask = lum >= high_lum
    ink_mask = lum <= low_lum

    cnt_paper = paper_mask.sum()
    cnt_ink = ink_mask.sum()
    if cnt_paper == 0:
        cnt_paper = 1
    if cnt_ink == 0:
        cnt_ink = 1

    paper_r = r[paper_mask].sum() / cnt_paper
    paper_g = g[paper_mask].sum() / cnt_paper
    paper_b = b[paper_mask].sum() / cnt_paper
    ink_r = r[ink_mask].sum() / cnt_ink
    ink_g = g[ink_mask].sum() / cnt_ink
    ink_b = b[ink_mask].sum() / cnt_ink

    return ColorStats(
        page_number=page_number,
        paper_r=paper_r,
        paper_g=paper_g,
        paper_b=paper_b,
        ink_r=ink_r,
        ink_g=ink_g,
        ink_b=ink_b,
    )


# ---------------------------------------------------------------------------
# Group-level outlier exclusion (simple top/bottom 20% trim on MeanR)
# ---------------------------------------------------------------------------

def exclude_outliers(stats_list: list[ColorStats]) -> list[ColorStats]:
    """Remove top/bottom 20% of pages sorted by paper-R."""
    if len(stats_list) < 3:
        return list(stats_list)
    sorted_stats = sorted(stats_list, key=lambda s: s.paper_r)
    skip = int(len(sorted_stats) * OUTLIER_TRIM_RATIO)
    take = len(sorted_stats) - skip * 2
    if take < 1:
        take = 1
    return sorted_stats[skip : skip + take]


# ---------------------------------------------------------------------------
# Global color-correction parameters (per odd/even group)
# ---------------------------------------------------------------------------

def decide_global_color_adjustment(stats_list: list[ColorStats]) -> GlobalColorParam:
    """Compute per-channel linear correction and ghost-suppression params."""
    if not stats_list:
        return GlobalColorParam(
            scale_r=1, scale_g=1, scale_b=1,
            offset_r=0, offset_g=0, offset_b=0,
            ghost_suppress_lum_threshold=200,
            paper_r=255, paper_g=255, paper_b=255,
            sat_threshold=SAT_THRESHOLD,
            color_dist_threshold=COLOR_DIST_THRESHOLD,
            bleed_hue_min=BLEED_HUE_MIN,
            bleed_hue_max=BLEED_HUE_MAX,
            bleed_value_min=BLEED_VALUE_MIN,
        )

    # 1) MAD-based page outlier removal
    paper_y = [0.299 * s.paper_r + 0.587 * s.paper_g + 0.114 * s.paper_b for s in stats_list]
    med_y = _percentile(paper_y, 50)
    mad = _percentile([abs(v - med_y) for v in paper_y], 50)
    thr = mad * MAD_MULTIPLIER

    main_pages = [
        s for s in stats_list
        if abs((0.299 * s.paper_r + 0.587 * s.paper_g + 0.114 * s.paper_b) - med_y) <= thr
    ]
    if not main_pages:
        main_pages = list(stats_list)

    # 2) Per-channel median of paper and ink
    bg_r = _percentile([s.paper_r for s in main_pages], 50)
    bg_g = _percentile([s.paper_g for s in main_pages], 50)
    bg_b = _percentile([s.paper_b for s in main_pages], 50)
    ink_r = _percentile([s.ink_r for s in main_pages], 50)
    ink_g = _percentile([s.ink_g for s in main_pages], 50)
    ink_b = _percentile([s.ink_b for s in main_pages], 50)

    # 3) Linear scale: ink→0, paper→255
    def _lin(bg: float, ink: float) -> tuple[float, float]:
        diff = bg - ink
        if diff < 1:
            return (1.0, 0.0)
        s = max(SCALE_CLAMP_MIN, min(SCALE_CLAMP_MAX, 255.0 / diff))
        o = -ink * s
        return (s, o)

    s_r, o_r = _lin(bg_r, ink_r)
    s_g, o_g = _lin(bg_g, ink_g)
    s_b, o_b = _lin(bg_b, ink_b)

    # 4) Ghost suppression threshold (midpoint of scaled ink & paper luminance)
    def _sc_clamp(v: float) -> float:
        return max(0.0, min(255.0, v))

    bg_lum_scaled = (
        0.299 * _sc_clamp(bg_r * s_r + o_r)
        + 0.587 * _sc_clamp(bg_g * s_g + o_g)
        + 0.114 * _sc_clamp(bg_b * s_b + o_b)
    )
    ink_lum_scaled = (
        0.299 * _sc_clamp(ink_r * s_r + o_r)
        + 0.587 * _sc_clamp(ink_g * s_g + o_g)
        + 0.114 * _sc_clamp(ink_b * s_b + o_b)
    )
    ghost_thr = int(max(0, min(255, (ink_lum_scaled + bg_lum_scaled) * 0.5)))

    return GlobalColorParam(
        scale_r=s_r, scale_g=s_g, scale_b=s_b,
        offset_r=o_r, offset_g=o_g, offset_b=o_b,
        ghost_suppress_lum_threshold=ghost_thr,
        paper_r=int(round(bg_r)),
        paper_g=int(round(bg_g)),
        paper_b=int(round(bg_b)),
        sat_threshold=SAT_THRESHOLD,
        color_dist_threshold=COLOR_DIST_THRESHOLD,
        bleed_hue_min=BLEED_HUE_MIN,
        bleed_hue_max=BLEED_HUE_MAX,
        bleed_value_min=BLEED_VALUE_MIN,
    )
