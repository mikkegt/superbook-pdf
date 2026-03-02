"""ImageMagick CLI wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

from superbook.config import EXTRACT_DPI, JPEG_QUALITY
from superbook.tools.runner import run_tool_async

logger = logging.getLogger(__name__)


async def extract_images_from_pdf(
    magick: str,
    src_pdf: Path,
    output_dir: Path,
    *,
    dpi: int = EXTRACT_DPI,
    max_pages: int | None = None,
) -> list[Path]:
    """Extract pages from *src_pdf* as BMP images at *dpi* resolution.

    Returns the list of extracted image paths sorted by page number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    page_spec = f"[0-{max_pages - 1}]" if max_pages else ""
    output_pattern = str(output_dir / "page_%05d.bmp")
    args: list[str] = [
        magick,
        "-density", str(dpi),
        f"{src_pdf}{page_spec}",
        "-depth", "8",
        "-type", "TrueColor",
        output_pattern,
    ]
    await run_tool_async(args)
    return sorted(output_dir.glob("page_*.bmp"))


async def get_deskew_angle(magick: str, image_path: Path) -> float:
    """Detect the skew angle of an image using ImageMagick's ``-deskew``.

    Returns the angle in degrees.  A value near 0 means no significant skew.
    """
    args = [
        magick,
        str(image_path),
        "-resize", "1920x1920>",
        "-deskew", "40%",
        "-format", "%[deskew:angle]",
        "info:",
    ]
    stdout = await run_tool_async(args)
    text = stdout.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        logger.warning("Could not parse deskew angle from: %r", text)
        return 0.0


async def build_pdf_from_images(
    magick: str,
    image_dir: Path,
    dst_pdf: Path,
    *,
    dpi: int = EXTRACT_DPI,
    quality: int = JPEG_QUALITY,
    pattern: str = "*.png",
) -> None:
    """Assemble PNG images in *image_dir* into a single PDF."""
    images = sorted(image_dir.glob(pattern))
    if not images:
        raise FileNotFoundError(f"No images matching {pattern} in {image_dir}")
    args: list[str] = [
        magick,
        "-density", str(dpi),
        "-units", "PixelsPerInch",
        *[str(p) for p in images],
        "-compress", "jpeg",
        "-quality", str(quality),
        str(dst_pdf),
    ]
    await run_tool_async(args, timeout_s=1800)
