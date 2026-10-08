"""Backend interface for prompted multimodal-LLM baselines: (RGB image, text prompt) -> answer text."""

from __future__ import annotations

import abc
import importlib
import io
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def require(module: str, extra: str, purpose: str):
    """Import an optional dependency or raise an ``ImportError`` naming the PyHazards extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{purpose} needs the optional package '{module.split('.')[0]}'. "
            f"Install it with `pip install pyhazards[{extra}]`."
        ) from exc


def check_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            f"expected an RGB uint8 image array of shape (H, W, 3), got shape {array.shape} and dtype {array.dtype}"
        )
    return array


def to_pil(image: np.ndarray):
    """RGB uint8 array -> ``PIL.Image`` (needs Pillow)."""
    pil_image = require("PIL.Image", "prompted", "Converting images for prompted backends")
    return pil_image.fromarray(check_image(image))


def encode_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    """JPEG bytes of an RGB uint8 array (Pillow; quality 95 is also OpenCV's ``imencode`` default)."""
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class PromptedBackend(abc.ABC):
    """A model answering one text prompt about one image.

    Subclasses implement :meth:`generate`. :meth:`generate_batch` (used for the 12 tiles of an image)
    defaults to a loop. :attr:`last_served_model` is set by API backends to the model version the
    provider reports for the latest call.
    """

    name: str = "backend"
    model_id: str = ""

    def __init__(self) -> None:
        self.last_served_model: Optional[str] = None

    @abc.abstractmethod
    def generate(self, image: np.ndarray, prompt: str) -> str:
        """Answer ``prompt`` about an RGB ``(H, W, 3)`` uint8 image."""

    def generate_batch(self, images: Sequence[np.ndarray], prompt: str) -> List[str]:
        return [self.generate(image, prompt) for image in images]

    def describe(self) -> Dict[str, Any]:
        """Settings recorded in evaluation reports."""
        return {"backend": type(self).__name__, "name": self.name, "model_id": self.model_id}


__all__ = ["PromptedBackend", "check_image", "encode_jpeg", "require", "to_pil"]
