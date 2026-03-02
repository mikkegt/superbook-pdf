"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def white_image() -> np.ndarray:
    """A 100x100 white RGB image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def dark_text_on_light_bg() -> np.ndarray:
    """A 200x200 image with dark text (black rectangle) on a light background."""
    img = np.full((200, 200, 3), 230, dtype=np.uint8)
    # Draw a dark rectangle in the centre (simulating text)
    img[50:150, 40:160] = 30
    return img


@pytest.fixture
def scanned_page_like() -> np.ndarray:
    """A 500x700 image mimicking a scanned book page.

    - Background: warm paper color (240, 235, 220)
    - Text block: dark (20, 20, 25) in the center
    - Margins: paper-colored
    """
    img = np.full((700, 500, 3), dtype=np.uint8, fill_value=0)
    img[:, :, 0] = 240  # R
    img[:, :, 1] = 235  # G
    img[:, :, 2] = 220  # B

    # Text block
    img[100:600, 60:440, :] = np.array([20, 20, 25], dtype=np.uint8)

    return img
