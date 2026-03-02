"""Text bounding-box detection and IQR-based crop region determination.

Ports the C# ``DetectTextBoundingBox`` and ``DecideGroupCropRegion`` methods.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from superbook.config import CONTOUR_MIN_AREA_RATIO, IQR_FENCE_MULTIPLIER
from superbook.models import BoundingBox, PageBoundingBox


# ---------------------------------------------------------------------------
# Per-page text bounding box
# ---------------------------------------------------------------------------

def detect_text_bounding_box(image: np.ndarray) -> BoundingBox:
    """Detect the text region in an RGB uint8 image via Otsu + contours.

    Returns a :class:`BoundingBox` covering all significant contours.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = gray.shape

    # 2. Mask border 1% on all 4 edges (fill with white=255)
    border_x = max(cols // 100, 1)
    border_y = max(rows // 100, 1)
    gray[:border_y, :] = 255
    gray[rows - border_y :, :] = 255
    gray[:, :border_x] = 255
    gray[:, cols - border_x :] = 255

    # 3. Otsu binary threshold (inverted: text=white, background=black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 4. Morphological opening (3x3 rect) to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Find external contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filter by min area (0.0025% of image area, min 10px)
    img_area = rows * cols
    min_area = max(int(img_area * CONTOUR_MIN_AREA_RATIO), 10)

    rects: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            rects.append((x, y, w, h))

    if not rects:
        return BoundingBox(0, 0, 0, 0)

    # 7. Encompassing bounding box
    min_x = min(r[0] for r in rects)
    min_y = min(r[1] for r in rects)
    max_x = max(r[0] + r[2] - 1 for r in rects)
    max_y = max(r[1] + r[3] - 1 for r in rects)

    return BoundingBox(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


# ---------------------------------------------------------------------------
# IQR-based crop region (group of pages)
# ---------------------------------------------------------------------------

def _percentile_int(values: list[int], p: float) -> int:
    """Linearly-interpolated percentile (p in 0.0–1.0) for sorted int list."""
    if not values:
        return 0
    idx = p * (len(values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return round(values[lo] + (values[hi] - values[lo]) * frac)


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) // 2


def decide_group_crop_region(bboxes: list[PageBoundingBox]) -> BoundingBox:
    """Compute the consensus crop region from per-page bounding boxes.

    Uses Tukey IQR fences (k=1.5) to detect and exclude outlier pages,
    then takes the median of each edge.
    """
    if not bboxes:
        return BoundingBox(0, 0, 0, 0)

    # Filter zero-area bboxes (blank pages)
    valid = [b for b in bboxes if b.bbox.width > 0 and b.bbox.height > 0]
    if not valid:
        return BoundingBox(0, 0, 0, 0)

    # 1) Sorted edge lists
    lefts = sorted(b.bbox.left for b in valid)
    tops = sorted(b.bbox.top for b in valid)
    rights = sorted(b.bbox.right for b in valid)
    bottoms = sorted(b.bbox.bottom for b in valid)

    # 2) Q1, Q3, IQR per edge
    q1_l = _percentile_int(lefts, 0.25)
    q3_l = _percentile_int(lefts, 0.75)
    iqr_l = max(q3_l - q1_l, 1)

    q1_t = _percentile_int(tops, 0.25)
    q3_t = _percentile_int(tops, 0.75)
    iqr_t = max(q3_t - q1_t, 1)

    q1_r = _percentile_int(rights, 0.25)
    q3_r = _percentile_int(rights, 0.75)
    iqr_r = max(q3_r - q1_r, 1)

    q1_b = _percentile_int(bottoms, 0.25)
    q3_b = _percentile_int(bottoms, 0.75)
    iqr_b = max(q3_b - q1_b, 1)

    k = IQR_FENCE_MULTIPLIER

    def _is_outlier(v: int, q1: int, q3: int, iqr: int) -> bool:
        return (v < q1 - k * iqr) or (v > q3 + k * iqr)

    # 3) Remove pages where ANY edge is an outlier
    inliers = [
        b for b in valid
        if not (
            _is_outlier(b.bbox.left, q1_l, q3_l, iqr_l)
            or _is_outlier(b.bbox.top, q1_t, q3_t, iqr_t)
            or _is_outlier(b.bbox.right, q1_r, q3_r, iqr_r)
            or _is_outlier(b.bbox.bottom, q1_b, q3_b, iqr_b)
        )
    ]

    # Fallback if too few inliers
    if len(inliers) < max(3, len(valid) // 2):
        inliers = valid

    # 4) Final region = median of inlier edges
    left = _median_int([b.bbox.left for b in inliers])
    top = _median_int([b.bbox.top for b in inliers])
    right = _median_int([b.bbox.right for b in inliers])
    bottom = _median_int([b.bbox.bottom for b in inliers])

    w = max(right - left, 0)
    h = max(bottom - top, 0)

    if w == 0 or h == 0:
        return BoundingBox(0, 0, 0, 0)

    return BoundingBox(left, top, w, h)
