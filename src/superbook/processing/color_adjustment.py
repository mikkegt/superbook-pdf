"""Per-pixel color adjustment: linear correction, smoothstep whitening, ghost suppression.

Ports the C# ``ApplyGlobalColorAdjustment`` method using numpy vectorized operations.
"""

from __future__ import annotations

import numpy as np

from superbook.config import WHITE_CLIP_RANGE
from superbook.models import GlobalColorParam


def _rgb_to_hsv_vectorized(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB (0-255 uint8) arrays to HSV (h: 0-360, s: 0-1, v: 0-1).

    Replicates the C# ``RgbToHsv`` helper exactly.
    """
    rf = r.astype(np.float32) / 255.0
    gf = g.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0

    maxc = np.maximum(rf, np.maximum(gf, bf))
    minc = np.minimum(rf, np.minimum(gf, bf))
    d = maxc - minc

    v = maxc
    safe_maxc = np.where(maxc == 0, 1.0, maxc)
    s = np.where(maxc == 0, 0.0, d / safe_maxc)

    # Hue calculation
    h = np.zeros_like(d)
    mask_d = d > 0
    mask_r = mask_d & (maxc == rf)
    mask_g = mask_d & (maxc == gf) & ~mask_r
    mask_b = mask_d & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((gf[mask_r] - bf[mask_r]) / d[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((bf[mask_g] - rf[mask_g]) / d[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((rf[mask_b] - gf[mask_b]) / d[mask_b]) + 4.0)
    h[h < 0] += 360.0

    return h, s, v


def apply_global_color_adjustment(image: np.ndarray, p: GlobalColorParam) -> np.ndarray:
    """Apply linear correction + smoothstep whitening + ghost suppression.

    *image* is ``(H, W, 3)`` RGB uint8.  Returns a new array of the same shape.
    """
    h, w = image.shape[:2]
    paper_r, paper_g, paper_b = p.paper_r, p.paper_g, p.paper_b
    clip_start = p.ghost_suppress_lum_threshold
    clip_end = max(0, min(255, 255 - WHITE_CLIP_RANGE))

    # Work in float64 for precision during computation
    src = image.astype(np.float64)

    # --- 1) Linear correction ---
    r = np.clip(src[:, :, 0] * p.scale_r + p.offset_r, 0, 255)
    g = np.clip(src[:, :, 1] * p.scale_g + p.offset_g, 0, 255)
    b = np.clip(src[:, :, 2] * p.scale_b + p.offset_b, 0, 255)

    # --- 2) Smoothstep whitening for paper-like pixels ---
    lum = (r * 299 + g * 587 + b * 114) / 1000.0
    high_lum_mask = lum >= clip_start

    if np.any(high_lum_mask):
        r_u8 = r[high_lum_mask]
        g_u8 = g[high_lum_mask]
        b_u8 = b[high_lum_mask]

        maxc = np.maximum(r_u8, np.maximum(g_u8, b_u8))
        minc = np.minimum(r_u8, np.minimum(g_u8, b_u8))
        safe_maxc = np.where(maxc == 0, 1.0, maxc)
        sat = np.where(maxc == 0, 0, (maxc - minc) * 255 / safe_maxc)
        dist = np.abs(r_u8 - paper_r) + np.abs(g_u8 - paper_g) + np.abs(b_u8 - paper_b)

        paper_like = (sat < p.sat_threshold) & (dist < p.color_dist_threshold)
        if np.any(paper_like):
            lum_sub = lum[high_lum_mask][paper_like]
            t = np.clip((lum_sub - clip_start) / (clip_end - clip_start + 1e-6), 0.0, 1.0)
            wgt = t * t * (3.0 - 2.0 * t)  # smoothstep

            # Build index for nested mask
            hl_indices = np.where(high_lum_mask)
            pl_indices = (hl_indices[0][paper_like], hl_indices[1][paper_like])

            r[pl_indices] = np.clip(r[pl_indices] + (255 - r[pl_indices]) * wgt, 0, 255)
            g[pl_indices] = np.clip(g[pl_indices] + (255 - g[pl_indices]) * wgt, 0, 255)
            b[pl_indices] = np.clip(b[pl_indices] + (255 - b[pl_indices]) * wgt, 0, 255)

    # --- 3) Pastel pink ghost suppression ---
    r_u8 = np.clip(r, 0, 255).astype(np.uint8)
    g_u8 = np.clip(g, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b, 0, 255).astype(np.uint8)

    hue, _, _ = _rgb_to_hsv_vectorized(r_u8, g_u8, b_u8)

    maxc2 = np.maximum(r, np.maximum(g, b))
    minc2 = np.minimum(r, np.minimum(g, b))
    safe_maxc2 = np.where(maxc2 == 0, 1.0, maxc2)
    sat2 = np.where(maxc2 == 0, 0, (maxc2 - minc2) * 255 / safe_maxc2)
    lum2 = (r * 299 + g * 587 + b * 114) / 1000.0

    is_pastel_pink = (lum2 > 230) & (sat2 < 30) & ((hue <= 40.0) | (hue >= 330.0))
    r[is_pastel_pink] = 255
    g[is_pastel_pink] = 255
    b[is_pastel_pink] = 255

    # Assemble output
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)
