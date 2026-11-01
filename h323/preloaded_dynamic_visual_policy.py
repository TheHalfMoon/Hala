"""H323 real preloaded image-conditioned two-stage policy for BeaconBoard.

MODEL/SOURCE ID and owner-private GPU authorization must be independently
attested by the caller. This class alone is NOT performance evidence.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from typing import Any

from h199.native_action_sft import parse_actions, source_task_prompt
from h273.relative_native_bridge import NORMALIZED_REGEX
from h323.dynamic_grammar import (
    STAGE_CLICK, STAGE_TYPE, STAGE_DONE, DynamicPolicyState,
    model_time_filter, model_visible_prompt,
)


class DynamicVisualPolicyRefused(ValueError):
    """Terminal categorical failure only, no model output or prompt."""


def need(ok: bool) -> None:
    if not ok:
        raise DynamicVisualPolicyRefused("H323 DYNAMIC_VISUAL_POLICY_REFUSED")


def rgb_digest(png: bytes) -> bytes:
    from PIL import Image
    try:
        need(type(png) is bytes and 100 <= len(png) < 4 * 1024 * 1024)
        with Image.open(BytesIO(png)) as image:
            need(image.format == "PNG" and image.size == (960, 720))
            rgb = image.convert("RGB")
            rgb.load()
            return sha256(rgb.tobytes()).digest()
    except Exception:
        raise DynamicVisualPolicyRefused("H323 INVALID_MODEL_VISIBLE_IMAGE") from None


class PreloadedDynamicVisualPolicy:
    def __init__(self, processor: Any, model: Any, task: str):
        need(callable(getattr(processor, "apply_chat_template", None))
             and callable(getattr(processor, "batch_decode", None))
             and hasattr(processor, "tokenizer"))
        need(callable(getattr(model, "generate", None))
             and callable(getattr(model, "eval", None))
             and getattr(model, "device", None) is not None)
        need(type(task) is str and 10 <= len(task) <= 500)
        self.processor, self.model, self.task = processor, model, task
        self.state = DynamicPolicyState()
        self._seen: set[bytes] = set()
        self._model_calls = 0
        self._grammar_calls = 0
        self._failed = False

    def __call__(self, png: bytes, prompt: str) -> str:
        need(not self._failed)
        try:
            return self._turn(png, prompt)
        except Exception:
            self._failed = True
            raise DynamicVisualPolicyRefused(
                "H323 REAL_VISUAL_GRAMMAR_TURN_FAILED"
            ) from None

    def _turn(self, png: bytes, prompt: str) -> str:
        need(prompt == source_task_prompt(self.task)
             and self.state.stage in (STAGE_CLICK, STAGE_TYPE))
        digest = rgb_digest(png)
        need(digest not in self._seen)
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from h268.free_t4_constrained_native import NativeGrammarModel
        from h279.private_relative_native_probe import (
            RelativePrevalidationProcessor,
        )
        stage = self.state.stage
        gate = model_time_filter(self.processor.tokenizer, stage)
        wrapper = NativeGrammarModel(self.model, gate)
        if stage == STAGE_CLICK:
            decoder = RelativePrevalidationProcessor(
                self.processor, self.state.consume_model_text,
                NORMALIZED_REGEX)
        else:
            decoder = self.processor
        policy = PreloadedVisionPolicy(decoder, wrapper, max_new_tokens=160)
        native = policy(png, model_visible_prompt(self.task, stage))
        need(wrapper.grammar_installed_calls == 1
             and wrapper.completed_generations == 1)
        parsed = parse_actions(native)
        if stage == STAGE_CLICK:
            need(decoder.converted_calls == 1
                 and self.state.generations == 1
                 and self.state.stage in (STAGE_TYPE, STAGE_DONE))
        else:
            need(self.state.stage == STAGE_TYPE
                 and self.state.generations == 1)
            need(self.state.consume_model_text(native) == native
                 and self.state.stage == STAGE_DONE
                 and self.state.generations == 2)
        kind = parsed["actions"][0]["kind"]
        need(kind in (("click", "abstain")
                      if stage == STAGE_CLICK else ("type_text", "abstain")))
        self._seen.add(digest)
        self._model_calls += 1
        self._grammar_calls += 1
        return native

    def seal_source_focus_refusal(self) -> None:
        need(not self._failed and self.state.stage == STAGE_TYPE
             and self.state.generations == self._model_calls == 1
             and self.state.native_clicks == 1
             and self.state.native_types == 0)
        self._failed = True

    def seal_source_write_refusal(self) -> None:
        need(not self._failed and self.state.stage == STAGE_DONE
             and self.state.generations == self._model_calls == 2
             and self.state.native_clicks == 1
             and self.state.native_types == 1)
        self._failed = True

    def safe_contract_summary(self) -> dict:
        safe = self.state.categorical_state()
        return {
            "model_generations_via_wrapped_generate": self._model_calls,
            "model_time_token_filters_installed": self._grammar_calls,
            "distinct_original_pngs_observed": len(self._seen),
            "focus_clicks": safe["focus_clicks"],
            "type_calls": safe["type_calls"],
            "abstained": safe["abstained"],
            "protocol_terminal": safe["terminal"],
            "protocol_failed_terminal": self._failed,
            "pretrained_model_identity_attested": False,
            "real_owner_private_gpu_run_attested": False,
            "native_source_task_success_attested": False,
            "official_train_dev_admitted": False,
            "blind_ood_proven": False,
            "generalizable_hala_gain_proven": False,
        }
