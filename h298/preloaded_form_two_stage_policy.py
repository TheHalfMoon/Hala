"""H298 — preloaded REAL VLM wiring for the original two-action form task.

Connects H297 model-time focus/click and type token grammars to the
unchanged H208 visual processor and H268 wrapped real model.generate.
This is source-only wiring, NOT trained-model inference or model skill
evidence. Never loads a checkpoint or reads DOM/gold from a policy.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any

from h199.native_action_sft import parse_actions, source_task_prompt
from h273.relative_native_bridge import NORMALIZED_REGEX
from h297.two_stage_form_grammar import (
    STAGE_FOCUS, STAGE_TYPE, STAGE_DONE, TwoStepFormPolicyState,
    model_time_filter, model_visible_prompt,
)


class ModelContractRefused(ValueError):
    """Categorical reason only; never append raw model text or screenshots."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ModelContractRefused("H298 original two-stage research refused: " + reason)


class PreloadedTwoStageVisualPolicy:
    """Actual H208 multimodal processing + two real token filters per scene.

    The caller, not this class, must separately attest authentic frozen
    pretrained model/revision, noncommercial free hardware, PRIVATE source
    runtime, fresh screenshot/source and post-action task state. Neither
    a fake model nor a successful click proves Qwen competence.
    """

    def __init__(self, processor: Any, model: Any, user_task: str) -> None:
        require(callable(getattr(processor, "apply_chat_template", None))
                and callable(getattr(processor, "batch_decode", None))
                and hasattr(processor, "tokenizer"),
                "preloaded multimodal processor contract required")
        require(callable(getattr(model, "generate", None))
                and callable(getattr(model, "eval", None))
                and getattr(model, "device", None) is not None,
                "preloaded visual model contract required")
        require(type(user_task) is str
                and 10 <= len(user_task) <= 500,
                "fixed original source user task required")
        self.processor = processor
        self.model = model
        self.task = user_task
        self.state = TwoStepFormPolicyState()
        self._png_digests: list[str] = []
        self._model_calls = 0
        self._installed_grammar_calls = 0
        self._terminal_failure = False

    def __call__(self, png: bytes, native_source_prompt: str) -> str:
        """Fail closed PERMANENTLY after any malformed model/source turn."""
        require(not self._terminal_failure,
                "failed policy must not resume on another screenshot")
        try:
            return self._qualified_turn(png, native_source_prompt)
        except Exception:
            # Any unexpected tokenizer/processor/model exception is terminal.
            # Do not leak exception messages or allow a second model action.
            self._terminal_failure = True
            raise ModelContractRefused(
                "H298 model-time visual grammar or source policy terminal failure"
            ) from None

    def _qualified_turn(self, png: bytes, native_source_prompt: str) -> str:
        """Use a FRESH rendered screenshot on EACH source callback."""
        require(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n")
                and 100 <= len(png) <= 4 * 1024 * 1024,
                "one real bounded PNG screenshot required")
        require(type(native_source_prompt) is str
                and native_source_prompt == source_task_prompt(self.task),
                "only the exact original user-visible source task")
        stage = self.state.stage
        require(stage in (STAGE_FOCUS, STAGE_TYPE),
                "no model call after abstention or two native steps")
        digest = sha256(png).hexdigest()
        require(digest not in self._png_digests,
                "fresh source screenshot required after first native action")

        # These imports are the genuine existing preloaded real visual
        # processor, native grammar wrapper and H279 prevalidation bridge.
        # They are never replaced with a text-only synthetic fallback.
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from h268.free_t4_constrained_native import NativeGrammarModel
        from h279.private_relative_native_probe import (
            RelativePrevalidationProcessor,
        )
        token_gate = model_time_filter(self.processor.tokenizer, stage)
        real_wrapper = NativeGrammarModel(self.model, token_gate)

        if stage == STAGE_FOCUS:
            # RelativePrevalidationProcessor converts DURING decoding.
            # This is the only safe point to let H297 advance the
            # state with the ORIGINAL strict prospective model text.
            # Never store/print the original raw model string.
            def convert_original_model_text(raw: str) -> str:
                return self.state.consume_model_text(raw)

            decoder = RelativePrevalidationProcessor(
                self.processor, convert_original_model_text, NORMALIZED_REGEX)
        else:
            decoder = self.processor
        policy = PreloadedVisionPolicy(
            decoder, real_wrapper, max_new_tokens=160)

        try:
            native = policy(png, model_visible_prompt(self.task, stage))
        except (ValueError, TypeError, RuntimeError, OSError):
            # Do not propagate model output, prompt, private token IDs
            # or image data in an error. No success evidence on error.
            raise ModelContractRefused(
                "H298 model-time visual grammar or processor failed"
            ) from None

        require(real_wrapper.grammar_installed_calls == 1
                and real_wrapper.completed_generations == 1,
                "one complete REAL wrapped model-time generation per stage")
        parsed = parse_actions(native)
        if stage == STAGE_FOCUS:
            require(decoder.converted_calls == 1
                    and self.state.generations == 1
                    and self.state.stage in (STAGE_TYPE, STAGE_DONE),
                    "strict prospective relative focus decoding incomplete")
            # H297 independently converted exact returned relative JSON.
        else:
            require(self.state.stage == STAGE_TYPE
                    and self.state.generations == 1,
                    "second stage must follow only the real first click")
            validated = self.state.consume_model_text(native)
            require(native == validated,
                    "type-stage raw response was rewritten")
            require(self.state.stage == STAGE_DONE
                    and self.state.generations == 2,
                    "strict second-stage transition incomplete")

        kind = parsed["actions"][0]["kind"]
        expected_kind = (
            ("click", "abstain") if stage == STAGE_FOCUS
            else ("type_text", "abstain"))
        require(kind in expected_kind, "unexpected stage action")
        self._png_digests.append(digest)
        self._model_calls += 1
        self._installed_grammar_calls += 1
        return native

    def safe_contract_summary(self) -> dict:
        """Only categorical counters, NO model outputs or image hashes."""
        safe = self.state.categorical_state()
        return {
            "source": "H298_FIELDPORT_PUBLIC_EXPOSED_RESEARCH",
            "model_generations_via_wrapped_generate": self._model_calls,
            "model_time_token_filters_installed": self._installed_grammar_calls,
            "distinct_original_pngs_observed": len(self._png_digests),
            "focus_clicks": safe["focus_clicks"],
            "type_calls": safe["type_calls"],
            "abstained": safe["abstained"],
            "protocol_terminal": safe["terminal"],
            "protocol_failed_terminal": self._terminal_failure,
            "pretrained_model_identity_attested": False,
            "real_owner_private_gpu_run_attested": False,
            "native_source_task_success_attested": False,
            "official_train_dev_admitted": False,
            "blind_ood_proven": False,
            "generalizable_hala_gain_proven": False,
        }


def run_fixed_original_source_research(
    browser: Any, source: Any, processor: Any,
    model: Any, scene: tuple,
) -> dict:
    """Integration entrypoint for caller-provided PRELOADED model only.

    The source-owned H296 evaluator must still independently judge
    post-action ALL-FOUR-field task state. This returns NO per-case raw
    task, code, screen or predicted coordinate and does not attest that
    a caller's model is authentic Qwen.
    """
    from h296.fieldport_sequence_research import (
        ROLE, SCENES, exact_task, run_original_form_sequence,
    )
    require(type(scene) is tuple and scene in SCENES,
            "registered public original form task only")
    policy = PreloadedTwoStageVisualPolicy(
        processor, model, exact_task(scene))
    outcome = run_original_form_sequence(
        browser, source, scene, policy, role=ROLE)
    counters = policy.safe_contract_summary()
    require(counters["protocol_terminal"]
            and counters["model_generations_via_wrapped_generate"]
                == outcome["native_actions"]
            and counters["model_time_token_filters_installed"]
                == outcome["native_actions"]
            and counters["distinct_original_pngs_observed"]
                == outcome["native_actions"]
            and counters["focus_clicks"]
                == int(outcome["focus_click_dispatched"])
            and counters["type_calls"]
                == int(outcome["type_dispatched"]),
            "model-time and source-owned native dispatch counts diverge")
    # Explicitly do NOT claim genuine model identity without separate
    # owner-private pinned model/revision/GPU verification.
    return {
        "source_original_gui_state_success":
            bool(outcome["source_task_success"]),
        "native_actions": outcome["native_actions"],
        "preloaded_model_contract": counters,
        "source_public_exposed_not_blind": True,
        "independent_model_identity_proven": False,
        "independently_attested_model_skill": False,
    }
