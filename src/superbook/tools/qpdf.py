"""qpdf CLI wrapper for PDF page-label manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from superbook.tools.runner import run_tool_async


async def set_page_labels(
    qpdf: str,
    src_pdf: Path,
    dst_pdf: Path,
    labels: Sequence[str],
) -> None:
    """Apply page labels to a PDF.

    *labels* should be qpdf label specs, e.g. ``["1:D/1", "5:D/5"]``.
    """
    args = [qpdf, str(src_pdf), "--set-page-labels", *labels, "--", str(dst_pdf)]
    await run_tool_async(args)
