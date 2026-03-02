"""pdfcpu CLI wrapper for PDF viewer preferences."""

from __future__ import annotations

from pathlib import Path

from superbook.tools.runner import run_tool_async

# Mapping from user-facing option values to pdfcpu parameters
_DIRECTION_MAP = {"left": "L2R", "right": "R2L"}
_LAYOUT_MAP = {"single": "SinglePage", "spread": "TwoPageLeft"}


async def set_viewer_preferences(
    pdfcpu: str,
    pdf_path: Path,
    *,
    binding: str = "left",
    page_layout: str = "single",
) -> None:
    """Set reading direction and page layout on the PDF."""
    direction = _DIRECTION_MAP[binding]
    layout = _LAYOUT_MAP[page_layout]

    await run_tool_async([
        pdfcpu, "viewerpref", "set", str(pdf_path),
        f'{{"Direction":"{direction}"}}',
    ])
    await run_tool_async([
        pdfcpu, "pagelayout", "set", str(pdf_path),
        layout,
    ])
