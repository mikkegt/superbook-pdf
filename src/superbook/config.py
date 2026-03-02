"""Constants and external tool path detection."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Internal canvas dimensions (high-res working space)
# ---------------------------------------------------------------------------
INTERNAL_WIDTH = 4960
INTERNAL_HEIGHT = 7016

# Final output height (A4 @ 300 dpi)
FINAL_OUTPUT_HEIGHT = 3508

# ---------------------------------------------------------------------------
# RealESRGAN
# ---------------------------------------------------------------------------
REALESRGAN_SCALE = 2.0

# ---------------------------------------------------------------------------
# Edge trim ratio (0.5% border removal after PDF extraction)
# ---------------------------------------------------------------------------
EDGE_TRIM_RATIO = 0.005

# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------
SAMPLE_STEP = 4  # subsampling interval for color statistics
PAPER_PERCENTILE = 0.95  # luminance threshold for paper pixels
INK_PERCENTILE = 0.05  # luminance threshold for ink pixels
OUTLIER_TRIM_RATIO = 0.20  # top/bottom trim for per-group stats

# ---------------------------------------------------------------------------
# MAD-based page outlier detection
# ---------------------------------------------------------------------------
MAD_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# Linear color correction
# ---------------------------------------------------------------------------
SCALE_CLAMP_MIN = 0.8
SCALE_CLAMP_MAX = 4.0
WHITE_CLIP_RANGE = 30

# ---------------------------------------------------------------------------
# Smoothstep whitening & ghost suppression
# ---------------------------------------------------------------------------
SAT_THRESHOLD = 55  # HSV saturation (0-255 scale)
COLOR_DIST_THRESHOLD = 35  # L1 distance to paper color

# ---------------------------------------------------------------------------
# Pastel-pink ghost detection
# ---------------------------------------------------------------------------
BLEED_HUE_MIN = 20.0  # degrees
BLEED_HUE_MAX = 65.0  # degrees
BLEED_VALUE_MIN = 0.35

# ---------------------------------------------------------------------------
# Text detection (OpenCV contour analysis)
# ---------------------------------------------------------------------------
CONTOUR_MIN_AREA_RATIO = 0.000025  # min contour area / image area
BORDER_MASK_RATIO = 0.01  # 1% border masking

# ---------------------------------------------------------------------------
# Paper color estimation
# ---------------------------------------------------------------------------
CORNER_PATCH_PERCENT = 3  # 3% of image for corner patches
PAPER_SAT_MAX = 40  # max saturation for paper-color pixels (0-255)
PAPER_LUM_FLOOR = 150  # fallback threshold floor

# ---------------------------------------------------------------------------
# Resize / padding
# ---------------------------------------------------------------------------
FEATHER_RANGE_PX = 4  # edge feather width in pixels

# ---------------------------------------------------------------------------
# Crop region (IQR Tukey fence)
# ---------------------------------------------------------------------------
IQR_FENCE_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# ImageMagick PDF extraction
# ---------------------------------------------------------------------------
EXTRACT_DPI = 300
JPEG_QUALITY = 70

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_IGNORE_MARGIN_PERCENT = 17
OCR_CHAR_MIN_HEIGHT = 30
OCR_SHIFT_SEARCH_RANGE = 300
OCR_MIN_MATCHES = 5
SCAN_SHIFT_MARGIN_WIDTH_RATIO = 0.030
SCAN_SHIFT_MARGIN_HEIGHT_RATIO = 0.025


# ---------------------------------------------------------------------------
# External tool path detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolPaths:
    """Resolved paths to required external CLI tools."""

    magick: str
    exiftool: str
    qpdf: str
    pdfcpu: str
    tesseract: str


def detect_tools() -> ToolPaths:
    """Locate external tools via ``shutil.which``; raise if any are missing."""
    names = ["magick", "exiftool", "qpdf", "pdfcpu", "tesseract"]
    paths: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        path = shutil.which(name)
        if path is None:
            missing.append(name)
        else:
            paths[name] = path
    if missing:
        raise RuntimeError(
            f"Required external tools not found: {', '.join(missing)}. "
            "Install them via: brew install imagemagick ghostscript exiftool qpdf pdfcpu tesseract tesseract-lang"
        )
    return ToolPaths(**paths)
