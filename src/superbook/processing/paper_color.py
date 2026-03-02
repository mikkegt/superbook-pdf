"""Paper color estimation via luminance histograms and corner sampling.

Ports the C# ``EstimatePaperColor``, ``AveragePaperColor``, and
``SampleCornerColors`` methods.
"""

from __future__ import annotations

import numpy as np

from superbook.config import CORNER_PATCH_PERCENT, PAPER_LUM_FLOOR, PAPER_SAT_MAX


def estimate_paper_color(image: np.ndarray) -> tuple[int, int, int]:
    """Estimate the dominant paper color from an RGB uint8 image.

    Returns ``(r, g, b)`` as integers.
    """
    # Sub-sample every 2nd pixel
    sampled = image[::2, ::2]
    r = sampled[:, :, 0].astype(np.int32)
    g = sampled[:, :, 1].astype(np.int32)
    b = sampled[:, :, 2].astype(np.int32)

    lum = (r * 299 + g * 587 + b * 114) // 1000

    # Luminance histogram → top 5% threshold
    hist = np.bincount(lum.ravel(), minlength=256)
    total = hist.sum()
    target = int(total * 0.05)
    acc = 0
    thr = 255
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= target:
            thr = i
            break

    # Among bright pixels, keep only low-saturation ones
    bright = lum >= thr
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    safe_maxc = np.where(maxc == 0, 1, maxc)
    sat = np.where(maxc == 0, 0, (maxc - minc) * 255 // safe_maxc)
    mask = bright & (sat < PAPER_SAT_MAX)

    cnt = mask.sum()
    if cnt == 0:
        return (255, 255, 255)

    return (
        int(r[mask].sum() // cnt),
        int(g[mask].sum() // cnt),
        int(b[mask].sum() // cnt),
    )


def _average_paper_color(image: np.ndarray, sx: int, sy: int, w: int, h: int) -> tuple[int, int, int]:
    """Local paper-color average within the rectangle ``(sx, sy, w, h)``.

    Falls back to :func:`estimate_paper_color` on the full image if the
    patch is too dark or yields no pixels.
    """
    img_h, img_w = image.shape[:2]
    sx = max(0, min(sx, img_w - 1))
    sy = max(0, min(sy, img_h - 1))
    w = max(1, min(w, img_w - sx))
    h = max(1, min(h, img_h - sy))

    patch = image[sy : sy + h : 2, sx : sx + w : 2]
    if patch.size == 0:
        return estimate_paper_color(image)

    r = patch[:, :, 0].astype(np.int32)
    g = patch[:, :, 1].astype(np.int32)
    b = patch[:, :, 2].astype(np.int32)

    lum = (r * 299 + g * 587 + b * 114) // 1000

    # Histogram → top 5% threshold
    hist = np.bincount(lum.ravel(), minlength=256)
    total = hist.sum()
    target = int(total * 0.05)
    acc = 0
    thr = 255
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= target:
            thr = i
            break

    if thr < PAPER_LUM_FLOOR:
        return estimate_paper_color(image)

    # Low-saturation bright pixels
    bright = lum >= thr
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    safe_maxc = np.where(maxc == 0, 1, maxc)
    sat = np.where(maxc == 0, 0, (maxc - minc) * 255 // safe_maxc)
    mask = bright & (sat < PAPER_SAT_MAX)

    cnt = mask.sum()
    if cnt == 0:
        return estimate_paper_color(image)

    return (
        int(r[mask].sum() // cnt),
        int(g[mask].sum() // cnt),
        int(b[mask].sum() // cnt),
    )


def sample_corner_colors(
    image: np.ndarray,
    percent: int = CORNER_PATCH_PERCENT,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Sample paper colors from the 4 corners.

    Returns ``(top_left, top_right, bottom_left, bottom_right)`` each as ``(r, g, b)``.
    """
    img_h, img_w = image.shape[:2]
    patch_w = max(img_w * percent // 100, 8)
    patch_h = max(img_h * percent // 100, 8)

    tl = _average_paper_color(image, 0, 0, patch_w, patch_h)
    tr = _average_paper_color(image, img_w - patch_w, 0, patch_w, patch_h)
    bl = _average_paper_color(image, 0, img_h - patch_h, patch_w, patch_h)
    br = _average_paper_color(image, img_w - patch_w, img_h - patch_h, patch_w, patch_h)

    return tl, tr, bl, br
