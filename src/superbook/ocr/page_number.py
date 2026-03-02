"""Tesseract-based page number detection and shift estimation.

Simplified port of the C# ``OcrProcessForBookAsync`` and related methods.
The core idea:
1. For each page, OCR the margin regions to find digit strings.
2. Try all shift offsets (-300..+300) to find the best physical→logical mapping.
3. Compute per-page X/Y shift from the detected page-number bounding box
   relative to a standard position.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from superbook.config import (
    OCR_CHAR_MIN_HEIGHT,
    OCR_IGNORE_MARGIN_PERCENT,
    OCR_MIN_MATCHES,
    OCR_SHIFT_SEARCH_RANGE,
)
from superbook.models import BookOcrMetadata, PageOcrMetadata

logger = logging.getLogger(__name__)

# Regex: 1-4 digits
_DIGIT_RE = re.compile(r"\b(\d{1,4})\b")


def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    """Contrast-enhance, grayscale, and Otsu-binarise for OCR."""
    enhanced = np.clip(image.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return otsu


def _extract_margin_regions(image: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Extract the 4 margin strips (top, bottom, left, right) for OCR.

    Returns list of (crop, (x, y, w, h)) tuples.
    """
    h, w = image.shape[:2]
    margin_x = int(w * OCR_IGNORE_MARGIN_PERCENT / 100)
    margin_y = int(h * OCR_IGNORE_MARGIN_PERCENT / 100)

    regions = [
        # Top strip
        (image[:margin_y, :], (0, 0, w, margin_y)),
        # Bottom strip
        (image[h - margin_y :, :], (0, h - margin_y, w, margin_y)),
        # Left strip
        (image[margin_y : h - margin_y, :margin_x], (0, margin_y, margin_x, h - 2 * margin_y)),
        # Right strip
        (image[margin_y : h - margin_y, w - margin_x :], (w - margin_x, margin_y, margin_x, h - 2 * margin_y)),
    ]
    return [(r, rect) for r, rect in regions if r.size > 0]


def _ocr_digits_in_region(region: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Run Tesseract on a margin region and extract digit strings.

    Returns list of (number, confidence, (x, y, w, h)) tuples.
    """
    if region.ndim == 3:
        otsu = _prepare_for_ocr(region)
    else:
        otsu = region

    # Tesseract with digit whitelist
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    try:
        data = pytesseract.image_to_data(otsu, config=custom_config, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return []

    results: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        m = _DIGIT_RE.match(text)
        if not m:
            continue
        num = int(m.group(1))
        if num < 1:
            continue
        conf = float(data["conf"][i]) / 100.0
        if conf < 0.1:
            continue
        bbox = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        if bbox[3] < OCR_CHAR_MIN_HEIGHT:
            continue
        results.append((num, conf, bbox))

    return results


def _detect_page_numbers(image: np.ndarray) -> dict[int, float]:
    """Detect candidate page numbers and their confidence scores."""
    regions = _extract_margin_regions(image)
    candidates: dict[int, float] = defaultdict(float)

    for region_img, (rx, ry, rw, rh) in regions:
        for num, conf, (lx, ly, lw, lh) in _ocr_digits_in_region(region_img):
            candidates[num] = max(candidates[num], conf)

    return dict(candidates)


def ocr_detect_page_numbers(
    image_paths: list[Path],
) -> BookOcrMetadata:
    """Run OCR on all pages and estimate the physical→logical page shift.

    Returns :class:`BookOcrMetadata` with per-page shift information.
    """
    # 1) Per-page OCR
    page_candidates: list[dict[int, float]] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            page_candidates.append({})
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        page_candidates.append(_detect_page_numbers(rgb))

    num_pages = len(image_paths)

    # 2) Shift search: try offsets to find best physical→logical mapping
    best_shift = 0
    best_score = 0.0
    best_match_count = 0

    for shift in range(-OCR_SHIFT_SEARCH_RANGE, OCR_SHIFT_SEARCH_RANGE):
        score = 0.0
        match_count = 0
        for phys_idx, candidates in enumerate(page_candidates):
            logical = phys_idx - shift
            if logical >= 1 and logical in candidates:
                score += candidates[logical]
                match_count += 1
        if score > best_score:
            best_score = score
            best_shift = shift
            best_match_count = match_count

    # 3) Check minimum quality
    max_phys = num_pages
    if best_match_count < OCR_MIN_MATCHES or (best_match_count * 3) < max_phys:
        # Insufficient OCR quality — return metadata without shift
        pages = [
            PageOcrMetadata(
                physical_page_number=i,
                logical_page_number=i,
                is_vertical_writing=False,
                shift_x=0,
                shift_y=0,
            )
            for i in range(num_pages)
        ]
        return BookOcrMetadata(pages=pages, is_vertical_writing=False)

    # 4) Build metadata with shift
    pages = [
        PageOcrMetadata(
            physical_page_number=i,
            logical_page_number=i - best_shift,
            is_vertical_writing=False,
            shift_x=0,
            shift_y=0,
        )
        for i in range(num_pages)
    ]

    return BookOcrMetadata(pages=pages, is_vertical_writing=False)
