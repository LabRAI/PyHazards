"""Closed SmokeBench models through their official Python clients (optional extras).

SmokeBench ran GPT-4o and Gemini-2.5 Pro on the classification task only (Sec. 4.1), at temperature
0.5. The release contains no code for these runs, so the request layout is a PyHazards choice: the
image is sent as a JPEG (quality 95, OpenCV's ``imencode`` default that the authors' one GPT-4o script
uses) and the prompt as text. GPT-4o gets ``[text, image]`` with ``max_tokens=300`` as in that script
(``mllm_smoke_locate.py`` of github.com/SoraLink/MLLM @ 183cdca); Gemini gets ``[image, text]``, the
order Google's prompting guide recommends for single-image prompts, and no output cap (Gemini 2.5 Pro
spends output tokens on thinking).

API keys are read from the environment only: ``OPENAI_API_KEY`` for OpenAI, ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``) for Gemini. Each call records the model version the provider reports in
``last_served_model``; providers update the models behind an alias, so results drift over time.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import numpy as np

from ..prompts import TEMPERATURE
from .base import PromptedBackend, encode_jpeg, require


class GeminiBackend(PromptedBackend):
    """Google Gemini (Gemini 2.5 report: Comanici et al., arXiv:2507.06261) via ``google-genai``."""

    name = "gemini"

    def __init__(
        self,
        model_id: str = "gemini-2.5-pro",
        *,
        temperature: Optional[float] = TEMPERATURE,
        jpeg_quality: int = 95,
        max_output_tokens: Optional[int] = None,
        client: Any = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.jpeg_quality = int(jpeg_quality)
        self.max_output_tokens = max_output_tokens
        if client is None:
            genai = require("google.genai", "prompted-gemini", "The Gemini backend")
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("The Gemini backend reads its API key from GEMINI_API_KEY (or GOOGLE_API_KEY); set it in the environment.")
            client = genai.Client(api_key=api_key)
        self.client = client

    def _config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.temperature is not None:
            config["temperature"] = float(self.temperature)
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = int(self.max_output_tokens)
        return config

    def generate(self, image: np.ndarray, prompt: str) -> str:
        image_part = {"inline_data": {"mime_type": "image/jpeg", "data": encode_jpeg(image, self.jpeg_quality)}}
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[{"role": "user", "parts": [image_part, {"text": prompt}]}],
            config=self._config(),
        )
        self.last_served_model = getattr(response, "model_version", None)
        return getattr(response, "text", None) or ""

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(generation=self._config(), jpeg_quality=self.jpeg_quality, content_order="image, text")
        return info


class OpenAIBackend(PromptedBackend):
    """OpenAI chat models such as GPT-4o (GPT-4o system card: Hurst et al., arXiv:2410.21276)."""

    name = "openai"

    def __init__(
        self,
        model_id: str = "gpt-4o",
        *,
        temperature: Optional[float] = TEMPERATURE,
        max_tokens: Optional[int] = 300,
        jpeg_quality: int = 95,
        client: Any = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.jpeg_quality = int(jpeg_quality)
        if client is None:
            openai = require("openai", "prompted-openai", "The OpenAI backend")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("The OpenAI backend reads its API key from OPENAI_API_KEY; set it in the environment.")
            client = openai.OpenAI()
        self.client = client

    def _options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.max_tokens is not None:
            options["max_tokens"] = int(self.max_tokens)
        return options

    def generate(self, image: np.ndarray, prompt: str) -> str:
        data = base64.b64encode(encode_jpeg(image, self.jpeg_quality)).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                    ],
                }
            ],
            **self._options(),
        )
        self.last_served_model = getattr(response, "model", None)
        return response.choices[0].message.content or ""

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(generation=self._options(), jpeg_quality=self.jpeg_quality, content_order="text, image")
        return info


__all__ = ["GeminiBackend", "OpenAIBackend"]
