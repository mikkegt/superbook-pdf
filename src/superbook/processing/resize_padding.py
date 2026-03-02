"""Resize, gradient-background padding, and edge feathering.

Ports the C# ``ResizeAndMakePaddingWithNaturalPaperColor`` and related helpers.
"""

from __future__ import annotations

import cv2
import numpy as np

from superbook.config import CORNER_PATCH_PERCENT, FEATHER_RANGE_PX
from superbook.processing.paper_color import sample_corner_colors


def _make_gradient_background(
    target_w: int,
    target_h: int,
    tl: tuple[int, int, int],
    tr: tuple[int, int, int],
    bl: tuple[int, int, int],
    br: tuple[int, int, int],
) -> np.ndarray:
    """Create a bilinear-interpolated gradient background (RGB uint8).

    Matches the C# ``Bilinear`` helper applied to every pixel of the canvas.
    """
    # Build u (horizontal) and v (vertical) interpolation grids
    u = np.linspace(0.0, 1.0, target_w, dtype=np.float32)  # (W,)
    v = np.linspace(0.0, 1.0, target_h, dtype=np.float32)  # (H,)
    u_grid, v_grid = np.meshgrid(u, v)  # both (H, W)

    canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
    for ch in range(3):
        top_row = tl[ch] * (1 - u_grid) + tr[ch] * u_grid
        bot_row = bl[ch] * (1 - u_grid) + br[ch] * u_grid
        canvas[:, :, ch] = (top_row * (1 - v_grid) + bot_row * v_grid).astype(np.uint8)

    return canvas


def _feather(
    canvas: np.ndarray,
    off_x: int,
    off_y: int,
    fitted_w: int,
    fitted_h: int,
    range_px: int,
) -> None:
    """Blend the boundary between the placed image and gradient background.

    Modifies *canvas* in-place.  Operates on the border strip of width
    *range_px* around the placed image rectangle.
    """
    if range_px <= 0:
        return

    canvas_h, canvas_w = canvas.shape[:2]

    # We need to keep a copy of the canvas background before the image was placed
    # so we can blend.  Since the image was already composited, we reconstruct:
    # - Inside the placed rectangle, the pixel is the foreground
    # - Outside, the pixel is already gradient background
    # The feathering replaces the sharp seam with a smooth transition.

    y_start = max(off_y - range_px, 0)
    y_end = min(off_y + fitted_h + range_px, canvas_h)
    x_start = max(off_x - range_px, 0)
    x_end = min(off_x + fitted_w + range_px, canvas_w)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            # Distance to the nearest edge of the placed-image rectangle
            dx = max(0, max(off_x - x, x - (off_x + fitted_w - 1)))
            dy = max(0, max(off_y - y, y - (off_y + fitted_h - 1)))
            d = max(dx, dy)
            if d == 0 or d >= range_px:
                continue  # fully inside or fully outside

            alpha = d / range_px  # 0 = boundary, 1 = fully background
            # The pixel in canvas at (y, x) was written as the foreground
            # when we composited.  We need the background color at this spot.
            # For the border strip *outside* the image rect, canvas already
            # has the gradient background.  For inside-border pixels, we
            # just skip (d==0 above).  So here d>0 means we're outside the
            # rect, so canvas[y,x] is already background.  We actually need
            # to blend the boundary, which means looking at the nearest
            # foreground pixel.  The simplest correct approach: read the
            # edge pixel from the placed image.
            # Clamp to the image boundary
            fx = max(off_x, min(x, off_x + fitted_w - 1))
            fy = max(off_y, min(y, off_y + fitted_h - 1))
            fg = canvas[fy, fx].astype(np.float32)
            bg = canvas[y, x].astype(np.float32)
            blended = bg + (fg - bg) * (1.0 - alpha)
            canvas[y, x] = np.clip(blended, 0, 255).astype(np.uint8)


def resize_and_pad(
    src: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    shift_x: int = 0,
    shift_y: int = 0,
    corner_patch_percent: int = CORNER_PATCH_PERCENT,
    feather: int = FEATHER_RANGE_PX,
) -> np.ndarray:
    """Resize *src* to fit inside *target_w × target_h* and pad with gradient background.

    This is the main entry point corresponding to the C# method
    ``ResizeAndMakePaddingWithNaturalPaperColor``.  *src* is RGB uint8.
    """
    src_h, src_w = src.shape[:2]

    # 1) Aspect-preserving fit
    scale = min(target_w / src_w, target_h / src_h)
    fitted_w = round(src_w * scale)
    fitted_h = round(src_h * scale)
    fitted = cv2.resize(src, (fitted_w, fitted_h), interpolation=cv2.INTER_LANCZOS4)

    # Offset (centred) + optional shift
    off_x = (target_w - fitted_w) // 2 + round(shift_x * scale)
    off_y = (target_h - fitted_h) // 2 + round(shift_y * scale)

    # 2) Sample 4 corner colors from the fitted image
    tl, tr, bl, br = sample_corner_colors(fitted, corner_patch_percent)

    # 3) Gradient background
    canvas = _make_gradient_background(target_w, target_h, tl, tr, bl, br)

    # 4) Composite the fitted image onto the canvas
    # Clamp placement to canvas bounds
    src_y1 = max(0, -off_y)
    src_x1 = max(0, -off_x)
    src_y2 = min(fitted_h, target_h - off_y)
    src_x2 = min(fitted_w, target_w - off_x)
    dst_y1 = max(0, off_y)
    dst_x1 = max(0, off_x)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = fitted[src_y1:src_y2, src_x1:src_x2]

    # 5) Edge feathering
    _feather(canvas, off_x, off_y, fitted_w, fitted_h, feather)

    return canvas


def resize_and_pad_with_crop(
    src: np.ndarray,
    target_w: int,
    target_h: int,
    crop_x: int,
    crop_y: int,
    scale: float,
    *,
    corner_patch_percent: int = CORNER_PATCH_PERCENT,
    feather: int = FEATHER_RANGE_PX,
) -> np.ndarray:
    """Variant with explicit scale and shift (``ResizeAndMakePaddingWithNaturalPaperColor2``).

    *crop_x* / *crop_y* are in the source coordinate system.
    """
    src_h, src_w = src.shape[:2]
    fitted_w = round(src_w * scale)
    fitted_h = round(src_h * scale)
    fitted = cv2.resize(src, (fitted_w, fitted_h), interpolation=cv2.INTER_LANCZOS4)

    off_x = round(crop_x * scale)
    off_y = round(crop_y * scale)

    tl, tr, bl, br = sample_corner_colors(fitted, corner_patch_percent)
    canvas = _make_gradient_background(target_w, target_h, tl, tr, bl, br)

    src_y1 = max(0, -off_y)
    src_x1 = max(0, -off_x)
    src_y2 = min(fitted_h, target_h - off_y)
    src_x2 = min(fitted_w, target_w - off_x)
    dst_y1 = max(0, off_y)
    dst_x1 = max(0, off_x)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = fitted[src_y1:src_y2, src_x1:src_x2]

    _feather(canvas, off_x, off_y, fitted_w, fitted_h, feather)
    return canvas
