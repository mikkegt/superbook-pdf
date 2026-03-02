"""Data models for the superbook pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessingOptions:
    """Options controlling the processing pipeline."""

    margin_percent: int = 7
    max_pages: int | None = None
    save_debug_png: bool = False
    skip_upscale: bool = False
    perform_ocr: bool = False
    page_layout: str = "single"  # "single" or "spread"
    binding: str = "left"  # "left" (L2R) or "right" (R2L)


@dataclass
class ColorStats:
    """Per-page color statistics (paper & ink RGB averages)."""

    page_number: int
    paper_r: float
    paper_g: float
    paper_b: float
    ink_r: float
    ink_g: float
    ink_b: float


@dataclass
class GlobalColorParam:
    """Global linear color-correction parameters for an odd/even group."""

    scale_r: float
    scale_g: float
    scale_b: float
    offset_r: float
    offset_g: float
    offset_b: float
    ghost_suppress_lum_threshold: int
    paper_r: int
    paper_g: int
    paper_b: int
    sat_threshold: int
    color_dist_threshold: int
    bleed_hue_min: float
    bleed_hue_max: float
    bleed_value_min: float


@dataclass
class BoundingBox:
    """Axis-aligned bounding box (origin at top-left)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width - 1

    @property
    def bottom(self) -> int:
        return self.y + self.height - 1


@dataclass
class PageInfo:
    """Tracks a single page through the pipeline."""

    file_path: Path
    page_number: int
    is_odd: bool
    color_adj_path: Path | None = None


@dataclass
class PageBoundingBox:
    """A page's detected text bounding box."""

    page_number: int
    bbox: BoundingBox


@dataclass
class PageOcrMetadata:
    """Per-page OCR-derived metadata."""

    physical_page_number: int
    logical_page_number: int
    is_vertical_writing: bool
    shift_x: int
    shift_y: int


@dataclass
class BookOcrMetadata:
    """Book-level OCR metadata."""

    pages: list[PageOcrMetadata] = field(default_factory=list)
    is_vertical_writing: bool = False
