"""YomiToku Japanese OCR CLI wrapper.

Requires: ``pip install yomitoku``
"""

from __future__ import annotations

import logging
from pathlib import Path

from superbook.tools.runner import run_tool_async

logger = logging.getLogger(__name__)


async def run_yomitoku_ocr(
    src_pdf: Path,
    dst_dir: Path,
    *,
    fmt: str = "pdf",
    dpi: int = 300,
    lite_mode: bool = False,
    ignore_line_break: bool = True,
    combine: bool = True,
    ignore_header_footer: bool = True,
    device: str = "mps",
    output_figures: bool = True,
    output_figure_letters: bool = True,
    encoding: str = "utf-8-sig",
    timeout_s: int = 5 * 3600,
) -> Path:
    """Run YomiToku OCR on a PDF and write the result to *dst_dir*.

    Returns the path to the output file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "yomitoku",
        str(src_pdf),
        "-o", str(dst_dir),
        "-f", fmt,
        "--dpi", str(dpi),
        "--device", device,
        "--encoding", encoding,
    ]
    if lite_mode:
        args.append("--lite")
    if ignore_line_break:
        args.append("--ignore_line_break")
    if combine:
        args.append("--combine")
    if ignore_header_footer:
        args.append("--ignore_header_footer")
    if output_figures:
        args.append("--figure")
    if output_figure_letters:
        args.append("--figure_letter")

    await run_tool_async(args, timeout_s=timeout_s)

    # Determine output file
    stem = src_pdf.stem
    ext_map = {"pdf": ".pdf", "md": ".md", "html": ".html", "json": ".json"}
    ext = ext_map.get(fmt, f".{fmt}")
    output_path = dst_dir / f"{stem}{ext}"
    return output_path
