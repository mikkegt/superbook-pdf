"""Pipeline orchestrator: the 6-step processing pipeline.

Ports the C# ``PerformPdfMainAsync`` → ``PerformPagesYohakuAsync`` flow.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from superbook.config import (
    EDGE_TRIM_RATIO,
    FINAL_OUTPUT_HEIGHT,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
)
from superbook.models import (
    BoundingBox,
    PageBoundingBox,
    PageInfo,
    ProcessingOptions,
)
from superbook.processing.color_adjustment import apply_global_color_adjustment
from superbook.processing.color_analysis import (
    calculate_color_stats,
    decide_global_color_adjustment,
    exclude_outliers,
)
from superbook.processing.deskew import deskew_image
from superbook.processing.resize_padding import resize_and_pad, resize_and_pad_with_crop
from superbook.processing.text_detection import (
    decide_group_crop_region,
    detect_text_bounding_box,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallel helper
# ---------------------------------------------------------------------------

async def _run_parallel(items: list, func, *, max_workers: int | None = None):
    """Run *func(item)* in threads, limited to *max_workers* concurrent tasks."""
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    sem = asyncio.Semaphore(max_workers)

    async def _limited(item):
        async with sem:
            return await asyncio.to_thread(func, item)

    return await asyncio.gather(*[_limited(i) for i in items])


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    """Load an image file as RGB uint8 numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB uint8 numpy array as PNG."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _trim_edges(image: np.ndarray) -> np.ndarray:
    """Trim 0.5% from each edge (scan artifact removal)."""
    h, w = image.shape[:2]
    if w < 10 or h < 10:
        return image
    mx = int(w * EDGE_TRIM_RATIO)
    my = int(h * EDGE_TRIM_RATIO)
    return image[my : h - my, mx : w - mx].copy()


def _equalize_crop_regions(
    odd_crop: BoundingBox,
    even_crop: BoundingBox,
    margin_percent: int,
) -> tuple[BoundingBox, BoundingBox]:
    """Unify Y extents, equalize sizes, and add margin."""
    # Unify Y across groups
    total_top = min(odd_crop.top, even_crop.top)
    total_bottom = max(odd_crop.bottom, even_crop.bottom)

    odd_crop = BoundingBox(odd_crop.x, total_top, odd_crop.width, total_bottom - total_top)
    even_crop = BoundingBox(even_crop.x, total_top, even_crop.width, total_bottom - total_top)

    # Max width/height + margin
    max_w = max(odd_crop.width, even_crop.width)
    max_h = max(odd_crop.height, even_crop.height)
    max_w += max_w * margin_percent // 100
    max_h += max_h * margin_percent // 100

    def _adjust(crop: BoundingBox) -> BoundingBox:
        dw = max_w - crop.width
        dh = max_h - crop.height
        new_left = crop.x - dw // 2
        new_top = crop.y - dh // 2

        clamped_w = min(max_w, INTERNAL_WIDTH)
        new_left = max(0, min(new_left, INTERNAL_WIDTH - clamped_w))

        clamped_h = min(max_h, INTERNAL_HEIGHT)
        new_top = max(0, min(new_top, INTERNAL_HEIGHT - clamped_h))

        return BoundingBox(new_left, new_top, clamped_w, clamped_h)

    return _adjust(odd_crop), _adjust(even_crop)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    src_pdf: Path,
    dst_pdf: Path,
    options: ProcessingOptions,
    magick: str,
    exiftool: str,
    qpdf: str,
    pdfcpu: str,
) -> None:
    """Execute the full 6-step processing pipeline."""
    from superbook.tools.imagemagick import build_pdf_from_images, extract_images_from_pdf
    from superbook.tools.exiftool import strip_metadata
    from superbook.tools.pdfcpu import set_viewer_preferences

    # Temp directory structure
    tmp_root = dst_pdf.parent / ".superbook_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    dirs = {
        "extracted": tmp_root / "1_extracted",
        "trimmed": tmp_root / "2_trimmed",
        "upscaled": tmp_root / "3_upscaled",
        "deskewed": tmp_root / "4_deskewed",
        "color_adj": tmp_root / "5_color_adj",
        "final": tmp_root / "6_final",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: PDF → images ──
    logger.info("Step 1: Extracting images from PDF...")
    extracted = await extract_images_from_pdf(
        magick, src_pdf, dirs["extracted"], max_pages=options.max_pages
    )
    if not extracted:
        raise RuntimeError("No pages extracted from PDF")
    logger.info("  Extracted %d pages", len(extracted))

    # ── Step 2: Edge trimming ──
    logger.info("Step 2: Trimming edges...")
    trimmed_paths: list[Path] = []
    for p in extracted:
        img = _load_rgb(p)
        trimmed = _trim_edges(img)
        out_path = dirs["trimmed"] / p.with_suffix(".bmp").name
        _save_png(out_path, trimmed)
        trimmed_paths.append(out_path)

    # ── Step 3: AI upscaling ──
    if options.skip_upscale:
        logger.info("Step 3: Skipping upscale")
        upscaled_paths = trimmed_paths
    else:
        logger.info("Step 3: AI upscaling (RealESRGAN)...")
        try:
            from superbook.upscaler import Upscaler
            upscaler = Upscaler()
            upscaled_paths = []
            for i, p in enumerate(trimmed_paths):
                logger.info("  Upscaling page %d/%d", i + 1, len(trimmed_paths))
                img = _load_rgb(p)
                up = upscaler.upscale(img)
                out_path = dirs["upscaled"] / f"upscaled_{i:05d}.png"
                _save_png(out_path, up)
                upscaled_paths.append(out_path)
        except ImportError:
            logger.warning("RealESRGAN not available, skipping upscale")
            upscaled_paths = trimmed_paths

    # ── Step 4: Page processing (the core) ──
    logger.info("Step 4: Page processing...")
    total_pages = len(upscaled_paths)
    page_infos = [
        PageInfo(
            file_path=upscaled_paths[i],
            page_number=i + 1,
            is_odd=((i + 1) % 2 == 1),
        )
        for i in range(total_pages)
    ]
    # Phase 4-1/4-2: Resize to internal canvas + deskew + color stats
    logger.info("  Phase 4-2: Deskew + color stats...")
    odd_stats = []
    even_stats = []

    for page in page_infos:
        logger.info("    Page %d: resize + deskew", page.page_number)
        img = _load_rgb(page.file_path)

        # Resize to internal canvas with gradient padding
        padded = resize_and_pad(img, INTERNAL_WIDTH, INTERNAL_HEIGHT)

        # Deskew
        deskewed = await deskew_image(magick, padded)

        # Save deskewed
        deskew_path = dirs["deskewed"] / f"deskew_{page.page_number:04d}.png"
        _save_png(deskew_path, deskewed)

        # Color stats
        stats = calculate_color_stats(deskewed, page.page_number)
        if page.is_odd:
            odd_stats.append(stats)
        else:
            even_stats.append(stats)

    # Phase 4-3: Global color parameters
    logger.info("  Phase 4-3: Global color params...")
    filtered_even = exclude_outliers(even_stats)
    filtered_odd = exclude_outliers(odd_stats)
    param_even = decide_global_color_adjustment(filtered_even)
    param_odd = decide_global_color_adjustment(filtered_odd)

    # Phase 4-4: Color adjustment + text detection
    logger.info("  Phase 4-4: Color adjustment + text detection...")
    odd_bboxes: list[PageBoundingBox] = []
    even_bboxes: list[PageBoundingBox] = []

    for page in page_infos:
        logger.info("    Page %d: color adj + bbox", page.page_number)
        deskew_path = dirs["deskewed"] / f"deskew_{page.page_number:04d}.png"
        img = _load_rgb(deskew_path)

        param = param_odd if page.is_odd else param_even
        adjusted = apply_global_color_adjustment(img, param)

        # Text bounding box
        bbox = detect_text_bounding_box(adjusted)
        pbb = PageBoundingBox(page.page_number, bbox)
        if page.is_odd:
            odd_bboxes.append(pbb)
        else:
            even_bboxes.append(pbb)

        # Save color-adjusted
        adj_path = dirs["color_adj"] / f"coloradj_{page.page_number:04d}.png"
        _save_png(adj_path, adjusted)
        page.color_adj_path = adj_path

    # Phase 4-5: Page number OCR (optional)
    ocr_metadata = None
    if options.perform_ocr:
        logger.info("  Phase 4-5: Page number OCR...")
        from superbook.ocr.page_number import ocr_detect_page_numbers
        adj_paths = [dirs["color_adj"] / f"coloradj_{p.page_number:04d}.png" for p in page_infos]
        ocr_metadata = ocr_detect_page_numbers(adj_paths)

    # Phase 4-6: Crop region determination
    logger.info("  Phase 4-6: Crop regions...")
    odd_crop = decide_group_crop_region(sorted(odd_bboxes, key=lambda b: b.page_number))
    even_crop = decide_group_crop_region(sorted(even_bboxes, key=lambda b: b.page_number))

    # Handle empty crop regions (all blank pages)
    if odd_crop.width == 0 or odd_crop.height == 0:
        odd_crop = BoundingBox(0, 0, INTERNAL_WIDTH, INTERNAL_HEIGHT)
    if even_crop.width == 0 or even_crop.height == 0:
        even_crop = BoundingBox(0, 0, INTERNAL_WIDTH, INTERNAL_HEIGHT)

    odd_crop, even_crop = _equalize_crop_regions(odd_crop, even_crop, options.margin_percent)

    # Compute final output dimensions
    crop_w = max(odd_crop.width, even_crop.width)
    crop_h = max(odd_crop.height, even_crop.height)
    final_height = FINAL_OUTPUT_HEIGHT
    final_width = crop_w * FINAL_OUTPUT_HEIGHT // crop_h

    # Phase 4-7: Final output
    logger.info("  Phase 4-7: Final output (%dx%d)...", final_width, final_height)
    for page in page_infos:
        logger.info("    Page %d: final resize + pad", page.page_number)
        adj_path = dirs["color_adj"] / f"coloradj_{page.page_number:04d}.png"
        img = _load_rgb(adj_path)

        crop = odd_crop if page.is_odd else even_crop

        # OCR shift (if available)
        shift_x = 0
        shift_y = 0
        if ocr_metadata and page.page_number - 1 < len(ocr_metadata.pages):
            pm = ocr_metadata.pages[page.page_number - 1]
            shift_x = pm.shift_x
            shift_y = pm.shift_y

        scale = final_width / crop.width
        final = resize_and_pad_with_crop(
            img,
            final_width,
            final_height,
            crop_x=-crop.x + shift_x,
            crop_y=-crop.y + shift_y,
            scale=scale,
        )

        out_path = dirs["final"] / f"page_{page.page_number:05d}.png"
        _save_png(out_path, final)

        if options.save_debug_png:
            debug_dir = dst_pdf.parent / "debug"
            debug_dir.mkdir(exist_ok=True)
            _save_png(debug_dir / f"page_{page.page_number:05d}.png", final)

    # ── Step 5: PDF reconstruction ──
    logger.info("Step 5: Building PDF...")
    await build_pdf_from_images(magick, dirs["final"], dst_pdf)

    # Strip metadata and set viewer preferences
    await strip_metadata(exiftool, dst_pdf)
    await set_viewer_preferences(
        pdfcpu, dst_pdf,
        binding=options.binding,
        page_layout=options.page_layout,
    )

    # ── Step 6: Japanese OCR (optional) ──
    if options.perform_ocr:
        logger.info("Step 6: Running YomiToku OCR...")
        try:
            from superbook.ocr.yomitoku import run_yomitoku_ocr
            ocr_dir = dst_pdf.parent / "ocr_output"
            await run_yomitoku_ocr(dst_pdf, ocr_dir)
        except Exception:
            logger.warning("YomiToku OCR failed or not available", exc_info=True)

    # Clean up
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info("Done! Output: %s", dst_pdf)
