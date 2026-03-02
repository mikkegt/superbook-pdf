"""RealESRGAN AI upscaler with MPS/CPU fallback.

Requires optional dependencies: ``pip install torch realesrgan basicsr``
"""

from __future__ import annotations

import logging

import numpy as np

from superbook.config import REALESRGAN_SCALE

logger = logging.getLogger(__name__)

_HAS_REALESRGAN = False
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    _HAS_REALESRGAN = True
except ImportError:
    pass


def _detect_device() -> str:
    """Pick the best available PyTorch device."""
    if not _HAS_REALESRGAN:
        raise RuntimeError(
            "RealESRGAN dependencies not installed. "
            "Install them with: pip install torch realesrgan basicsr"
        )
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Upscaler:
    """Wraps RealESRGAN for 2× super-resolution.

    Automatically selects MPS (Apple Silicon GPU), CUDA, or CPU.
    """

    def __init__(self, scale: float = REALESRGAN_SCALE, model_name: str = "RealESRGAN_x4plus") -> None:
        if not _HAS_REALESRGAN:
            raise RuntimeError(
                "RealESRGAN dependencies not installed. "
                "Install them with: pip install torch realesrgan basicsr"
            )

        device = _detect_device()
        logger.info("RealESRGAN using device: %s", device)

        # RealESRGAN_x4plus architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        self._engine = RealESRGANer(
            scale=4,
            model_path=None,  # auto-download
            model=model,
            device=device,
            half=False,  # MPS doesn't support half precision well
        )
        self._scale = scale

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale an RGB uint8 image.

        *image* is ``(H, W, 3)`` RGB.  Returns the upscaled image.

        Note: RealESRGAN expects BGR input internally; we handle the conversion.
        """
        import cv2

        # RealESRGAN expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output, _ = self._engine.enhance(bgr, outscale=self._scale)
        # Convert back to RGB
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
