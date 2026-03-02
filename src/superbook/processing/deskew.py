"""Image deskew: angle detection via ImageMagick + rotation via OpenCV.

Ports the C# ``DeskewImageWithOpenCvAsync`` method.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from superbook.tools.imagemagick import get_deskew_angle

logger = logging.getLogger(__name__)


def _prepare_otsu_image(image: np.ndarray) -> np.ndarray:
    """Produce a contrast-enhanced Otsu-binarised image for deskew detection.

    Mirrors the C# ``PerformOtsuForPaperPage``: contrast × 1.5 → grayscale → Otsu.
    """
    # Simple contrast increase (scale by 1.5, clamp to 255)
    enhanced = np.clip(image.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return otsu


async def deskew_image(
    magick: str,
    image: np.ndarray,
) -> np.ndarray:
    """Detect skew and rotate *image* to correct it.

    *image* is ``(H, W, 3)`` RGB uint8.  Returns the deskewed image.
    """
    # 1. Prepare Otsu-binarised image for angle detection
    otsu = _prepare_otsu_image(image)

    # 2. Write temp PNG for ImageMagick
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), otsu)

    try:
        angle = await get_deskew_angle(magick, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("    Deskew angle detected: %.4f°", angle)

    # 3. Skip if angle is negligible
    if abs(angle) < 0.001:
        return image

    # Negate for correction (C#: angle = -angle)
    angle = -angle

    # 4. Rotate using OpenCV warpAffine
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated
