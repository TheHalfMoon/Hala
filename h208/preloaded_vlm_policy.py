"""H208: bounded, offline preloaded Qwen3-VL-style HF processor/model policy.

Connects H207 screenshot-only predictor signature to an *already loaded*
multimodal model. Requires actual decoded PNG pixels in processor messages,
a visual tensor in its input batch, continuation-only model decoding and
strict H199/H200 JSON. No checkpoint loading, downloads, GPU scheduling,
media persistence or model attestation happens in this module.

Caller MUST separately qualify pinned model/processor provenance, CPU/GPU
allocation, source-disjoint data, human supervision and fresh OOD before
claiming a Hala model result. Synthetic mock test ≠ model inference.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any

from h199.native_action_sft import parse_actions
from h201.original_native_action_executor import _bounded_preaction_png_digest

MAX_NEW_TOKENS = 192


def _has_visual_values(value: Any) -> bool:
    """Reject missing or zero-element visual tensors, never cast Tensor to bool."""
    if value is None:
        return False
    try:
        if hasattr(value, "numel"):
            return int(value.numel()) > 0
        return len(value) > 0
    except (AttributeError, ValueError, TypeError, OverflowError):
        return False


def _tokens(seq: Any, *, why: str) -> list[int]:
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if not isinstance(seq, (list, tuple)):
        raise ValueError("H208 invalid " + why + " token sequence")
    result = list(seq)
    if not result or any(type(x) is not int or x < 0 for x in result):
        raise ValueError("H208 invalid " + why + " token sequence")
    return result


class PreloadedVisionPolicy:
    """Use as H207 predict callback with a verified, preloaded HF VLM.

    Uses the documented Qwen3-VL apply_chat_template image/text workflow.
    No from_pretrained call; model identity is NOT attested by this class.
    """

    def __init__(
        self, processor: Any, model: Any, *,
        max_new_tokens: int = 160,
    ) -> None:
        if (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= MAX_NEW_TOKENS
            or not callable(getattr(processor, "apply_chat_template", None))
            or not callable(getattr(processor, "batch_decode", None))
            or not callable(getattr(model, "generate", None))
            or not callable(getattr(model, "eval", None))
            or getattr(model, "device", None) is None
        ):
            raise ValueError("H208 requires preloaded processor/model and bounded tokens")
        self.processor = processor
        self.model = model
        self.max_new_tokens = max_new_tokens

    def __call__(self, png: bytes, prompt: str) -> str:
        # The PNG image is passed all the way into HF multimodal processing.
        _bounded_preaction_png_digest(png)
        if type(prompt) is not str or not 0 < len(prompt) <= 2000:
            raise ValueError("H208 invalid screenshot-only task prompt")
        try:
            from PIL import Image
            with Image.open(BytesIO(png)) as source:
                if source.format != "PNG" or source.size != (960, 720):
                    raise ValueError("unexpected original GUI PNG")
                rgb = source.convert("RGB")
                rgb.load()  # force full real pixel decoding, not PNG header only
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError("H208 malformed preaction PNG pixels") from None

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": prompt},
            ],
        }]
        # The real Qwen3-VL AutoProcessor supports this exact API. Never
        # allow a text-only fallback on missing pixel_values.
        try:
            batch = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if (
                batch is None
                or "input_ids" not in batch
                or "pixel_values" not in batch
                or not _has_visual_values(batch["pixel_values"])
            ):
                raise ValueError("missing model-visible image tensor")
            prefix = _tokens(batch["input_ids"][0], why="prompt")
            if not hasattr(batch, "to"):
                raise ValueError("multimodal tensor batch cannot move to model")
            batch = batch.to(self.model.device)
            if (
                "pixel_values" not in batch
                or not _has_visual_values(batch["pixel_values"])
            ):
                raise ValueError("image lost during device transfer")
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            raise ValueError("H208 multimodal processor rejected screenshot") from None

        try:
            import torch
            self.model.eval()
            with torch.inference_mode():
                output = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            if len(output) != 1:
                raise ValueError("wrong generated batch size")
            sequence = _tokens(output[0], why="generated")
            if (
                len(sequence) <= len(prefix)
                or len(sequence) - len(prefix) > self.max_new_tokens
                or sequence[:len(prefix)] != prefix
            ):
                raise ValueError("model generated invalid prompt prefix")
            texts = self.processor.batch_decode(
                [sequence[len(prefix):]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if (
                type(texts) is not list
                or len(texts) != 1
                or type(texts[0]) is not str
            ):
                raise ValueError("bad continuation decoding")
            # The only accepted output is a single exact H199 native action.
            # No markdown, extra fields, free-form explanation or guesses.
            parse_actions(texts[0])
            return texts[0]
        except Exception:
            # Never echo raw model output, media, instruction or paths.
            raise ValueError("H208 generated action was not qualified") from None
