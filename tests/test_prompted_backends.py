"""Request construction of the prompted backends with fake models / clients (needs Pillow; no network)."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PIL = pytest.importorskip("PIL.Image")

from pyhazards.prompted import CLASSIFICATION_PROMPT, GRID_PROMPT, build_backend  # noqa: E402
from pyhazards.prompted.backends import GeminiBackend, Idefics2Backend, InternVL3Backend, OpenAIBackend, Qwen25VLBackend  # noqa: E402

IMAGE = (np.arange(30 * 40 * 3) % 256).reshape(30, 40, 3).astype(np.uint8)


class FakeInputs(dict):
    def to(self, device, dtype=None):
        self.moved = (device, dtype)
        return self


class FakeTokenizer:
    pad_token = None
    eos_token = "</s>"
    padding_side = "right"


class FakeProcessor:
    """Records what the backend sends; 'tokenizes' each text into 5 prompt tokens."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(do_image_splitting=True)
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append(("chat", conversation, kwargs))
        if kwargs.get("tokenize"):
            return FakeInputs(input_ids=torch.zeros(len(conversation), 5, dtype=torch.long))
        return "<chat>"

    def __call__(self, text=None, images=None, **kwargs):
        self.calls.append(("call", text, images, kwargs))
        return FakeInputs(input_ids=torch.zeros(len(text), 5, dtype=torch.long))

    def batch_decode(self, tokens, **kwargs):
        self.calls.append(("decode", tokens.tolist(), kwargs))
        return [self.answer] * tokens.shape[0]


class FakeModel:
    device = "cpu"
    dtype = torch.bfloat16
    config = SimpleNamespace(_commit_hash="abc123")

    def __init__(self):
        self.generate_kwargs = []

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs.append(kwargs)
        new = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


def _backend(cls, answer, **kwargs):
    processor, model = FakeProcessor(), FakeModel()
    processor.answer = answer
    return cls(model=model, processor=processor, **kwargs), model, processor


def test_qwen_backend_resizes_to_448_and_strips_fences():
    backend, model, processor = _backend(Qwen25VLBackend, '```json\n[{"region": 3}]\n```')
    assert backend.generate(IMAGE, GRID_PROMPT) == '[{"region": 3}]'
    kind, conversation, kwargs = processor.calls[0]
    assert kind == "chat" and kwargs == {"tokenize": False, "add_generation_prompt": True}
    content = conversation[0]["content"]
    assert content[0]["type"] == "image" and content[0]["image"].size == (448, 448)
    assert content[1] == {"type": "text", "text": GRID_PROMPT}
    _, texts, images, call_kwargs = processor.calls[1]
    assert texts == ["<chat>"] and images[0].size == (448, 448) and call_kwargs["padding"] is True
    assert model.generate_kwargs == [{"max_new_tokens": 128, "do_sample": True, "temperature": 0.5}]
    _, decoded, decode_kwargs = processor.calls[2]
    assert decoded == [[7, 7]]  # only the new tokens are decoded
    assert decode_kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    assert backend.describe()["revision"] == "abc123" and backend.describe()["image_size"] == (448, 448)


def test_qwen_backend_batches_tiles_with_left_padding_and_greedy_option():
    backend, model, processor = _backend(Qwen25VLBackend, "True", do_sample=False, image_size=None)
    answers = backend.generate_batch([IMAGE, IMAGE[:, :20]], CLASSIFICATION_PROMPT)
    assert answers == ["True", "True"]
    assert processor.tokenizer.padding_side == "left"
    assert processor.calls[2][2][0].size == (40, 30) and processor.calls[2][2][1].size == (20, 30)
    assert model.generate_kwargs == [{"max_new_tokens": 128, "do_sample": False}]


def test_internvl_backend_uses_the_chat_template_with_tokenization():
    backend, model, processor = _backend(InternVL3Backend, "False")
    assert backend.generate(IMAGE, CLASSIFICATION_PROMPT) == "False"
    kind, conversations, kwargs = processor.calls[0]
    assert kind == "chat" and kwargs["tokenize"] and kwargs["return_dict"] and kwargs["add_generation_prompt"]
    assert conversations[0][0]["content"][0]["image"].size == (40, 30)  # native resolution
    assert model.generate_kwargs == [{"max_new_tokens": 256, "do_sample": True, "temperature": 0.5}]
    assert backend.dtype == "float32"


def test_idefics2_backend_raw_prompt_for_images_and_chat_batch_for_tiles():
    backend, model, processor = _backend(Idefics2Backend, " True")
    assert processor.image_processor.do_image_splitting is False
    assert backend.generate(IMAGE, CLASSIFICATION_PROMPT) == "True"
    _, texts, images, _ = processor.calls[0]
    assert texts == ["<image>\n" + CLASSIFICATION_PROMPT] and len(images) == 1 and len(images[0]) == 1
    assert model.generate_kwargs[-1] == {"max_new_tokens": 256, "do_sample": True, "temperature": 0.5}
    processor.calls.clear()
    assert backend.generate_batch([IMAGE] * 3, CLASSIFICATION_PROMPT) == ["True"] * 3
    assert [call[0] for call in processor.calls] == ["chat", "chat", "chat", "call", "decode"]
    assert processor.tokenizer.padding_side == "left" and processor.tokenizer.pad_token == "</s>"
    assert model.generate_kwargs[-1] == {"max_new_tokens": 16, "do_sample": True, "temperature": 0.5}


class FakeGeminiModels:
    def __init__(self):
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text="True", model_version="gemini-2.5-pro-test")


class FakeOpenAICompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(model="gpt-4o-2024-08-06", choices=[SimpleNamespace(message=SimpleNamespace(content="False"))])


def _decode_jpeg(data: bytes) -> np.ndarray:
    return np.asarray(PIL.open(io.BytesIO(data)).convert("RGB"))


def test_gemini_request_layout():
    client = SimpleNamespace(models=FakeGeminiModels())
    backend = build_backend("gemini_2_5_pro", client=client)
    assert backend.generate(IMAGE, CLASSIFICATION_PROMPT) == "True"
    assert backend.last_served_model == "gemini-2.5-pro-test" and backend.name == "gemini_2_5_pro"
    request = client.models.requests[0]
    assert request["model"] == "gemini-2.5-pro" and request["config"] == {"temperature": 0.5}
    image_part, text_part = request["contents"][0]["parts"]
    assert text_part == {"text": CLASSIFICATION_PROMPT}
    assert image_part["inline_data"]["mime_type"] == "image/jpeg"
    assert np.abs(_decode_jpeg(image_part["inline_data"]["data"]).astype(int) - IMAGE).mean() < 8


def test_openai_request_layout():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeOpenAICompletions()))
    backend = OpenAIBackend(client=client)
    assert backend.generate(IMAGE, CLASSIFICATION_PROMPT) == "False"
    assert backend.last_served_model == "gpt-4o-2024-08-06"
    request = client.chat.completions.requests[0]
    assert request["model"] == "gpt-4o" and request["temperature"] == 0.5 and request["max_tokens"] == 300
    text_part, image_part = request["messages"][0]["content"]
    assert text_part == {"type": "text", "text": CLASSIFICATION_PROMPT}
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert _decode_jpeg(base64.b64decode(url.split(",", 1)[1])).shape == IMAGE.shape


def test_backends_reject_non_rgb_arrays():
    backend = GeminiBackend(client=SimpleNamespace(models=FakeGeminiModels()))
    with pytest.raises(ValueError, match="RGB uint8"):
        backend.generate(IMAGE.astype(np.float32), CLASSIFICATION_PROMPT)


def test_presets_pin_revisions_and_keep_the_card_name_only_for_the_card_model():
    from pyhazards.prompted.backends import REVISIONS

    client = SimpleNamespace(models=FakeGeminiModels())
    assert build_backend("gemini_2_5_pro", client=client).name == "gemini_2_5_pro"
    assert build_backend("gemini_2_5_pro", client=client, model_id="gemini-3.1-pro-preview").name == "gemini"
    processor, model = FakeProcessor(), FakeModel()
    qwen = build_backend("qwen2_5_vl_7b", model=model, processor=processor)
    assert qwen.name == "qwen2_5_vl_7b" and qwen.revision == REVISIONS["Qwen/Qwen2.5-VL-7B-Instruct"]
    other = build_backend("qwen2_5_vl_7b", model=model, processor=processor, model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    assert other.name == "qwen2_5_vl" and other.revision is None
