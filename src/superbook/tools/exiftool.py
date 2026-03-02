"""ExifTool CLI wrapper for PDF metadata manipulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from superbook.tools.runner import run_tool_async


async def strip_metadata(exiftool: str, pdf_path: Path) -> None:
    """Remove all metadata from a PDF file."""
    await run_tool_async([exiftool, "-overwrite_original", "-all:all=", str(pdf_path)])


async def set_dates(
    exiftool: str,
    pdf_path: Path,
    *,
    create_date: datetime | None = None,
    modify_date: datetime | None = None,
) -> None:
    """Set CreateDate and/or ModifyDate on a PDF file."""
    args = [exiftool, "-overwrite_original"]
    fmt = "%Y:%m:%d %H:%M:%S"
    if create_date:
        args.extend([f"-CreateDate={create_date.strftime(fmt)}"])
    if modify_date:
        args.extend([f"-ModifyDate={modify_date.strftime(fmt)}"])
    args.append(str(pdf_path))
    await run_tool_async(args)
