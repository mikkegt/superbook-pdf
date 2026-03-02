"""CLI entry point using Click."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from superbook.config import detect_tools
from superbook.models import ProcessingOptions


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.command()
@click.argument("src_pdf", type=click.Path(exists=True, path_type=Path))
@click.argument("dst_pdf", type=click.Path(path_type=Path))
@click.option("--margin", default=7, show_default=True, help="Margin percent added around text area.")
@click.option("--max-pages", default=None, type=int, help="Limit processing to first N pages (debug).")
@click.option("--skip-upscale", is_flag=True, help="Skip RealESRGAN upscaling.")
@click.option("--ocr", is_flag=True, help="Run page-number OCR and YomiToku Japanese OCR.")
@click.option("--save-debug", is_flag=True, help="Save intermediate images for debugging.")
@click.option(
    "--layout",
    type=click.Choice(["single", "spread"], case_sensitive=False),
    default="single",
    show_default=True,
    help="Page layout: single page or spread (見開き).",
)
@click.option(
    "--binding",
    type=click.Choice(["left", "right"], case_sensitive=False),
    default="left",
    show_default=True,
    help="Binding direction: left (左綴じ/L2R) or right (右綴じ/R2L).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    src_pdf: Path,
    dst_pdf: Path,
    margin: int,
    max_pages: int | None,
    skip_upscale: bool,
    ocr: bool,
    save_debug: bool,
    layout: str,
    binding: str,
    verbose: bool,
) -> None:
    """Process a scanned book PDF: deskew, color-correct, upscale, and repackage.

    SRC_PDF is the input PDF file.  DST_PDF is the output path.
    """
    _setup_logging(verbose)

    # Verify external tools
    try:
        tools = detect_tools()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    options = ProcessingOptions(
        margin_percent=margin,
        max_pages=max_pages,
        save_debug_png=save_debug,
        skip_upscale=skip_upscale,
        perform_ocr=ocr,
        page_layout=layout,
        binding=binding,
    )

    from superbook.pipeline import run_pipeline

    asyncio.run(
        run_pipeline(
            src_pdf=src_pdf,
            dst_pdf=dst_pdf,
            options=options,
            magick=tools.magick,
            exiftool=tools.exiftool,
            qpdf=tools.qpdf,
            pdfcpu=tools.pdfcpu,
        )
    )
