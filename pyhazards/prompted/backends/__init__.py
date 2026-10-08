"""Backends for prompted baselines. Optional packages are imported only when a backend is built."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .api import GeminiBackend, OpenAIBackend
from .base import PromptedBackend, check_image, encode_jpeg, to_pil
from .hf import Idefics2Backend, InternVL3Backend, Qwen25VLBackend, TransformersBackend

# Hugging Face revisions of the open checkpoints, pinned on 2026-10-07 (SmokeBench did not pin any);
# pass revision=None to build_backend for the latest revision.
REVISIONS: Dict[str, str] = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "Qwen/Qwen2.5-VL-32B-Instruct": "7cfb30d71a1f4f49a57592323337a4a4727301da",
    "OpenGVLab/InternVL3-14B-hf": "e22931943e5336f85e06f4e2b38f3e5e6ee4de3b",
    "HuggingFaceM4/idefics2-8b": "2c42686c57fe21cf0348c9ce1077d094b72e7698",
}


def _open(cls: Callable[..., PromptedBackend], model_id: str) -> Callable[..., PromptedBackend]:
    def build(**kwargs: Any) -> PromptedBackend:
        kwargs.setdefault("model_id", model_id)
        if kwargs["model_id"] == model_id:
            kwargs.setdefault("revision", REVISIONS[model_id])
        return cls(**kwargs)

    return build


def _api(cls: Callable[..., PromptedBackend], model_id: str) -> Callable[..., PromptedBackend]:
    def build(**kwargs: Any) -> PromptedBackend:
        kwargs.setdefault("model_id", model_id)
        return cls(**kwargs)

    return build


# Named configurations of the models SmokeBench evaluated (one per card in pyhazards/prompted/cards).
MODEL_IDS: Dict[str, str] = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "internvl3_14b": "OpenGVLab/InternVL3-14B-hf",
    "idefics2_8b": "HuggingFaceM4/idefics2-8b",
    "gemini_2_5_pro": "gemini-2.5-pro",
    "gpt_4o": "gpt-4o",
}
_PRESETS: Dict[str, Callable[..., PromptedBackend]] = {
    "qwen2_5_vl_7b": _open(Qwen25VLBackend, MODEL_IDS["qwen2_5_vl_7b"]),
    "qwen2_5_vl_32b": _open(Qwen25VLBackend, MODEL_IDS["qwen2_5_vl_32b"]),
    "internvl3_14b": _open(InternVL3Backend, MODEL_IDS["internvl3_14b"]),
    "idefics2_8b": _open(Idefics2Backend, MODEL_IDS["idefics2_8b"]),
    "gemini_2_5_pro": _api(GeminiBackend, MODEL_IDS["gemini_2_5_pro"]),
    "gpt_4o": _api(OpenAIBackend, MODEL_IDS["gpt_4o"]),
}


def available_backends() -> List[str]:
    return sorted(_PRESETS)


def build_backend(name: str, **kwargs: Any) -> PromptedBackend:
    """Build a SmokeBench model by preset name (see :func:`available_backends`).

    Keyword arguments go to the backend class. The backend keeps the preset name (which links run
    reports to the numbers SmokeBench reports) only while ``model_id`` is the preset's model.
    """
    if name not in _PRESETS:
        raise KeyError(f"unknown prompted backend {name!r}; known: {available_backends()}")
    backend = _PRESETS[name](**kwargs)
    if backend.model_id == MODEL_IDS[name]:
        backend.name = name
    return backend


__all__ = [
    "MODEL_IDS",
    "REVISIONS",
    "GeminiBackend",
    "Idefics2Backend",
    "InternVL3Backend",
    "OpenAIBackend",
    "PromptedBackend",
    "Qwen25VLBackend",
    "TransformersBackend",
    "available_backends",
    "build_backend",
    "check_image",
    "encode_jpeg",
    "to_pil",
]
