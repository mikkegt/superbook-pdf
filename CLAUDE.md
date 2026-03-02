# superbook-pdf: Project Instructions for Claude Code

## Overview

Python rewrite of DN_SuperBook_PDF_Converter (C# .NET 6.0). Processes scanned book PDFs: deskew, color correction, AI upscaling, OCR. Runs on macOS (Apple Silicon).

## Project Structure

```
src/superbook/
  cli.py                     # Click CLI entry point
  pipeline.py                # 6-step pipeline orchestrator
  config.py                  # Named constants + tool path detection
  models.py                  # Dataclasses (ProcessingOptions, ColorStats, etc.)
  upscaler.py                # RealESRGAN wrapper (optional)
  tools/
    runner.py                # Shared async subprocess runner
    imagemagick.py           # PDF extraction, deskew angle, PDF assembly
    exiftool.py              # Metadata stripping
    qpdf.py                  # Page labels (unused currently)
    pdfcpu.py                # Viewer preferences (direction, layout)
  processing/
    deskew.py                # ImageMagick angle detection + OpenCV warpAffine
    color_analysis.py        # Per-page color stats, MAD outlier exclusion
    color_adjustment.py      # Linear correction, smoothstep whitening, ghost suppression
    text_detection.py        # Otsu+contour BBox, IQR crop region
    paper_color.py           # Paper color estimation, corner sampling
    resize_padding.py        # Gradient background + Lanczos resize + feather
  ocr/
    page_number.py           # Tesseract page number detection
    yomitoku.py              # YomiToku Japanese OCR wrapper
tests/
  test_*.py                  # Unit tests (pytest + pytest-asyncio)
```

## Tech Stack

- **Python 3.10+**, managed with **uv** (pyproject.toml)
- **CLI**: click
- **Image processing**: OpenCV (cv2), numpy
- **AI upscaling**: RealESRGAN + PyTorch (optional)
- **OCR**: pytesseract, yomitoku (optional)
- **External tools**: ImageMagick, Ghostscript, ExifTool, qpdf, pdfcpu

## Key Commands

```bash
uv sync --group dev          # Install dependencies
uv run pytest tests/ -v      # Run tests
uv run ruff check src/ tests/  # Lint
uv run superbook input.pdf output.pdf --skip-upscale -v  # Run CLI
```

## Architecture Notes

- Pipeline processes odd/even pages separately (duplex scanners produce different characteristics per side)
- All image processing uses numpy arrays (H, W, 3) in RGB uint8 format
- OpenCV reads BGR; conversion happens at I/O boundaries (_load_rgb, _save_png in pipeline.py)
- Constants are in config.py — never use magic numbers in processing code
- External tools are called via async subprocess (tools/runner.py)
- numpy divide-by-zero: always use `safe_x = np.where(x == 0, 1, x)` pattern before division in np.where

## Tuning Guide for Output Quality

These are the key parameters to adjust when the output doesn't look right. All constants are in `config.py` unless noted otherwise.

### Deskew (tilt correction)
- **Current approach**: ImageMagick detects angle, OpenCV rotates
- Duplex scanning causes alternating tilt direction per page — this is normal
- If deskew is too aggressive/weak, the threshold in `deskew.py` line 56 (`abs(angle) < 0.001`) controls the minimum angle to correct
- ImageMagick's `-deskew 40%` threshold is in `tools/imagemagick.py`

### Edge trimming (scan artifacts)
- `EDGE_TRIM_RATIO = 0.005` (0.5%) — trims ~5-7px from each edge at 300dpi
- If paper edge shadows are still visible, increase this (e.g., 0.01 for 1%)
- Root cause is often the scanning/cutting technique, not the software

### Color correction
- `SAT_THRESHOLD = 55` — saturation below this is considered "paper-like"
- `COLOR_DIST_THRESHOLD = 35` — L1 distance to paper color for whitening
- `WHITE_CLIP_RANGE = 30` — smoothstep luminance range for whitening
- `BLEED_HUE_MIN/MAX` (20-65 degrees) — hue range for ghost/bleed-through suppression
- If pages look too washed out: lower SAT_THRESHOLD or COLOR_DIST_THRESHOLD
- If bleed-through (裏写り) remains: widen BLEED_HUE range or raise BLEED_VALUE_MIN

### Crop region
- `IQR_FENCE_MULTIPLIER = 1.5` — Tukey fence for outlier page detection
- `--margin 7` (CLI) — percent padding around detected text area
- If text is clipped: increase margin. If too much whitespace: decrease margin

### Paper color padding
- `CORNER_PATCH_PERCENT = 3` — size of corner patches for paper color sampling
- `FEATHER_RANGE_PX = 4` — edge blending width between content and padding
- `PAPER_SAT_MAX = 40` — max saturation for paper-color pixels

### PDF viewer settings
- `--layout single|spread` — SinglePage or TwoPageLeft
- `--binding left|right` — L2R or R2L direction
- Manga/vertical text: `--binding right --layout spread`
- Textbooks/horizontal text: `--binding left` (default)

### Performance
- Full 206-page PDF: ~70 minutes with `--skip-upscale` on Apple Silicon
- Bottlenecks: deskew (ImageMagick subprocess per page), color adjustment (numpy)
- `--max-pages N` for quick testing
- `--save-debug` outputs intermediate images to `debug/` directory

## Testing

- 38 unit tests covering color analysis, color adjustment, text detection, deskew, resize/padding, pipeline
- Tests use small synthetic images (no real PDFs needed)
- External tools are mocked in tests
- Run with: `uv run pytest tests/ -x -q`

## Known Limitations

- AI upscaling requires separate torch/realesrgan install (not in default deps)
- YomiToku OCR is optional and may not be stable
- Edge trimming may be insufficient for poorly cut books (consider increasing EDGE_TRIM_RATIO)
- Processing is sequential per page within each phase (parallelization is set up but not fully utilized yet)
