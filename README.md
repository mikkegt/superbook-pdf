# superbook-pdf

[日本語版 README はこちら](README.ja.md)

Book PDF processor: deskew, color correction, AI upscaling, and OCR.

Python rewrite of [DN_SuperBook_PDF_Converter](https://github.com/mikkegt/DN_SuperBook_PDF_Converter) (C# / .NET 6.0 / Windows-only). Developed and tested on **macOS (Apple Silicon)**. Should also work on **Linux** and **Windows** (untested).

## Features

- **PDF to image extraction** (ImageMagick, 300 dpi)
- **Edge trimming** (0.5% scan artifact removal)
- **AI super-resolution** (RealESRGAN 2x, CUDA/MPS/CPU)
- **Deskew** (ImageMagick angle detection + OpenCV rotation)
- **Color correction** (linear scaling, smoothstep whitening, ghost suppression)
- **Text region detection** (Otsu binarization + contour analysis)
- **IQR-based crop region** (Tukey fence outlier rejection)
- **Gradient paper-color padding** (bilinear corner-sampled background)
- **Page number OCR** (Tesseract)
- **Japanese OCR** (YomiToku, optional)
- **Configurable PDF layout** (single/spread, left/right binding)

## Requirements

### System tools

**macOS** (Homebrew):

```bash
brew install imagemagick ghostscript exiftool qpdf pdfcpu tesseract tesseract-lang
```

**Ubuntu / Debian**:

```bash
sudo apt install imagemagick ghostscript libimage-exiftool-perl qpdf tesseract-ocr tesseract-ocr-jpn
# pdfcpu: download from https://github.com/pdfcpu/pdfcpu/releases
```

**Windows** (Chocolatey):

```powershell
choco install imagemagick ghostscript exiftool qpdf tesseract
# pdfcpu: download from https://github.com/pdfcpu/pdfcpu/releases
```

### Python

Python 3.10+ with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

### AI upscaling (optional)

```bash
pip install torch realesrgan basicsr
```

GPU acceleration: CUDA (NVIDIA, Linux/Windows), MPS (Apple Silicon), or CPU fallback.

### Japanese OCR (optional)

```bash
pip install yomitoku
```

## Usage

```bash
# Basic usage (skipping AI upscale)
superbook input.pdf output.pdf --skip-upscale

# Full pipeline with AI upscale
superbook input.pdf output.pdf

# With debug images and limited pages
superbook input.pdf output.pdf --max-pages 10 --save-debug -v

# With OCR
superbook input.pdf output.pdf --ocr

# Manga (right-to-left, spread view)
superbook input.pdf output.pdf --binding right --layout spread

# Textbook (left-to-right, single page)
superbook input.pdf output.pdf --binding left --layout single
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--margin N` | `7` | Margin percent around text area |
| `--max-pages N` | all | Process only first N pages |
| `--skip-upscale` | off | Skip RealESRGAN upscaling |
| `--ocr` | off | Enable page-number OCR and YomiToku |
| `--save-debug` | off | Save intermediate images |
| `--layout [single\|spread]` | `single` | Page layout: single page or two-page spread |
| `--binding [left\|right]` | `left` | Binding direction: left (L2R) or right (R2L) |
| `-v, --verbose` | off | Debug logging |

### Layout and binding guide

| Use case | `--layout` | `--binding` |
|----------|-----------|-------------|
| Textbook (horizontal writing) | `single` | `left` |
| Textbook (spread view) | `spread` | `left` |
| Manga (vertical, right-to-left) | `spread` | `right` |
| Manga (single page) | `single` | `right` |

## Processing pipeline

```
Step 1: PDF -> images (ImageMagick, 300 dpi)
Step 2: Edge trim (0.5% margin removal)
Step 3: AI upscale (RealESRGAN 2x)
Step 4: Page processing
  4-1: Initialize page list (odd/even groups)
  4-2: Deskew + color statistics
  4-3: Global color parameters (MAD outlier exclusion)
  4-4: Color correction + text detection
  4-5: Page number OCR (Tesseract, optional)
  4-6: Crop region (IQR outlier detection)
  4-7: Final output (resize + gradient padding)
Step 5: PDF reconstruction (ImageMagick -> ExifTool -> pdfcpu)
Step 6: Japanese OCR (YomiToku, optional)
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## License

AGPL-3.0-only (following the original project)

## Credits

Based on [DN_SuperBook_PDF_Converter](https://github.com/mikkegt/DN_SuperBook_PDF_Converter) by mikkegt.
