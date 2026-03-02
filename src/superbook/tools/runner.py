"""Common subprocess execution helper."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 600  # 10 minutes default


class ToolError(RuntimeError):
    """Raised when an external tool exits with a non-zero status."""

    def __init__(self, tool: str, returncode: int, stderr: str) -> None:
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{tool} exited with code {returncode}: {stderr[:500]}")


def run_tool(args: Sequence[str], *, timeout_s: int = TIMEOUT_SECONDS) -> str:
    """Run an external tool synchronously and return its stdout."""
    logger.debug("Running: %s", " ".join(str(a) for a in args))
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise ToolError(str(args[0]), result.returncode, result.stderr)
    return result.stdout


async def run_tool_async(args: Sequence[str], *, timeout_s: int = TIMEOUT_SECONDS) -> str:
    """Run an external tool asynchronously and return its stdout."""
    logger.debug("Running async: %s", " ".join(str(a) for a in args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(str(args[0]), -1, "Process timed out")
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise ToolError(str(args[0]), proc.returncode, stderr)
    return stdout
