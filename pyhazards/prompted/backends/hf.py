"""Open-weight SmokeBench models through Hugging Face transformers (optional extra ``prompted-hf``).

SmokeBench evaluated Qwen2.5-VL-7B/32B-Instruct, InternVL3-14B and Idefics2-8B. The paper states only
the prompts and the decoding temperature (0.5); everything else here follows the authors'
``mllms.py`` (github.com/SoraLink/MLLM, commit 183cdca488e8c7afa5a254f86ac675cb98951c73, no licence;
nothing is copied, the settings are re-implemented and listed in the docs):

* Qwen2.5-VL: the image is resized to 448 x 448 with ``PIL.Image.resize`` before the processor
  (all four tasks), single-turn chat template with the image before the text, 128 new tokens, and a
  Markdown ```json fence is stripped from the answer;
* InternVL3: the transformers-native checkpoint ``OpenGVLab/InternVL3-14B-hf`` through its chat
  template (the image-text-to-text pipeline the authors used), float32 weights, 256 new tokens (the
  pipeline default);
* Idefics2: ``do_image_splitting=False``, bfloat16. Single images (classification, grid, detection)
  are sent as the raw text ``"<image>\\n" + prompt`` without the chat template, 256 new tokens (the
  pipeline call of the authors' ``predict``); the 12 tiles of the tile task go as one left-padded
  batch through the chat template with 16 new tokens (their ``batch_predict``).

Sampling: ``temperature=0.5`` with ``do_sample=True`` as the paper states. The released script never
sets a temperature, so its runs used each checkpoint's ``generation_config`` (greedy for Idefics2 and
InternVL3, temperature 1e-6 for Qwen2.5-VL); pass ``do_sample=False`` to decode greedily instead.
Other sampling parameters come from the checkpoint's ``generation_config``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..parsers import strip_json_fence
from ..prompts import TEMPERATURE
from .base import PromptedBackend, require, to_pil

DTypeLike = Union[str, Any]


def _resolve_dtype(dtype: DTypeLike):
    if isinstance(dtype, str) and dtype != "auto":
        import torch  # noqa: PLC0415

        return getattr(torch, dtype)
    return dtype


def _load(model_id: str, revision: Optional[str], dtype: DTypeLike, device_map: Any):
    transformers = require("transformers", "prompted-hf", f"Loading {model_id}")
    major_minor = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if major_minor >= (4, 56) else "torch_dtype"
    processor = transformers.AutoProcessor.from_pretrained(model_id, revision=revision)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, device_map=device_map, **{dtype_key: _resolve_dtype(dtype)}
    )
    return model, processor


class TransformersBackend(PromptedBackend):
    """Shared loading, generation settings and decoding of the transformers backends."""

    default_model_id = ""
    default_dtype: DTypeLike = "auto"
    default_max_new_tokens = 256

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        revision: Optional[str] = None,
        dtype: Optional[DTypeLike] = None,
        device_map: Any = "auto",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = TEMPERATURE,
        do_sample: bool = True,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id or self.default_model_id
        self.revision = revision
        self.dtype = self.default_dtype if dtype is None else dtype
        self.device_map = device_map
        self.max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        self.temperature = temperature
        self.do_sample = bool(do_sample)
        if model is None or processor is None:
            model, processor = _load(self.model_id, revision, self.dtype, device_map)
        self.model = model
        self.processor = processor
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generation_kwargs(self, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample and self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        return kwargs

    def _to_device(self, inputs):
        dtype = getattr(self.model, "dtype", None)
        device = getattr(self.model, "device", None)
        if dtype is not None:
            return inputs.to(device, dtype=dtype)
        return inputs.to(device)

    def _generate(self, inputs, max_new_tokens: Optional[int] = None) -> List[str]:
        import torch  # noqa: PLC0415

        inputs = self._to_device(inputs)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.generation_kwargs(max_new_tokens))
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return list(
            self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )

    def _left_padding(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        config = getattr(self.model, "config", None)
        info.update(
            revision=self.revision or getattr(config, "_commit_hash", None),
            dtype=str(self.dtype),
            generation=self.generation_kwargs(),
        )
        return info


class Qwen25VLBackend(TransformersBackend):
    """Qwen2.5-VL-Instruct (Bai et al., arXiv:2502.13923) as run by SmokeBench."""

    name = "qwen2_5_vl"
    default_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    default_max_new_tokens = 128

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        image_size: Optional[Tuple[int, int]] = (448, 448),
        strip_fence: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.strip_fence = bool(strip_fence)

    def _prepare(self, image: np.ndarray):
        pil = to_pil(image)
        if self.image_size is not None:
            pil = pil.resize(self.image_size)
        return pil

    def generate(self, image: np.ndarray, prompt: str) -> str:
        return self.generate_batch([image], prompt)[0]

    def generate_batch(self, images: Sequence[np.ndarray], prompt: str) -> List[str]:
        pils = [self._prepare(image) for image in images]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for pil in pils
        ]
        if len(pils) > 1:
            self._left_padding()
        inputs = self.processor(text=texts, images=pils, padding=True, return_tensors="pt")
        answers = self._generate(inputs)
        return [strip_json_fence(text) if self.strip_fence else text for text in answers]

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(image_size=self.image_size, strip_json_fence=self.strip_fence)
        return info


class InternVL3Backend(TransformersBackend):
    """InternVL3 (Zhu et al., arXiv:2504.10479) through its transformers-native ``-hf`` checkpoint."""

    name = "internvl3"
    default_model_id = "OpenGVLab/InternVL3-14B-hf"
    default_dtype = "float32"
    default_max_new_tokens = 256

    def generate(self, image: np.ndarray, prompt: str) -> str:
        return self.generate_batch([image], prompt)[0]

    def generate_batch(self, images: Sequence[np.ndarray], prompt: str) -> List[str]:
        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": to_pil(image)}, {"type": "text", "text": prompt}]}]
            for image in images
        ]
        if len(conversations) > 1:
            self._left_padding()
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        return self._generate(inputs)


class Idefics2Backend(TransformersBackend):
    """Idefics2 (Laurencon et al., NeurIPS 2024) as run by SmokeBench (no image splitting)."""

    name = "idefics2"
    default_model_id = "HuggingFaceM4/idefics2-8b"
    default_dtype = "bfloat16"
    default_max_new_tokens = 256
    batch_max_new_tokens = 16

    def __init__(self, model_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            image_processor.do_image_splitting = False

    def generate(self, image: np.ndarray, prompt: str) -> str:
        text = prompt if "<image>" in prompt else "<image>\n" + prompt
        inputs = self.processor(text=[text], images=[[to_pil(image)]], return_tensors="pt")
        return self._generate(inputs)[0].lstrip()

    def generate_batch(self, images: Sequence[np.ndarray], prompt: str) -> List[str]:
        pils = [to_pil(image) for image in images]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for pil in pils
        ]
        self._left_padding()
        inputs = self.processor(
            text=texts, images=[[pil] for pil in pils], return_tensors="pt", padding=True, truncation=True
        )
        return [text.strip() for text in self._generate(inputs, max_new_tokens=self.batch_max_new_tokens)]

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(do_image_splitting=False, batch_generation=self.generation_kwargs(self.batch_max_new_tokens))
        return info


__all__ = ["Idefics2Backend", "InternVL3Backend", "Qwen25VLBackend", "TransformersBackend"]
