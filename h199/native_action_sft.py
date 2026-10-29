"""H199: unified Hala original-GUI action and multimodal SFT contract.

Copy-first H161 native click/drag JSON + H157 typed actions, extending to
real original slider keyboard input. This is a CPU-only, purely *synthetic*
contract smoke; it never opens H188/H190/H191/H194/H197 images or private
gold, and MUST NOT approve training or a new model.

No user media, Kaggle, Colab, model weights, paid compute, or OOD.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

WIDTH, HEIGHT = 960, 720
ALLOWED_KEYS = frozenset({
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "Enter", "Escape", "Tab", "Space",
})
KINDS = frozenset({"click", "drag", "key_press", "abstain"})
MAX_JSON_BYTES = 1200
# Intentional: no original GUI source manifests have been admitted.
# Future independent release qualification MUST modify this on a
# reviewed Draft PR only AFTER proven role, source and history gates.
ADMITTED_TRAIN_MANIFEST_SHAS: frozenset[str] = frozenset()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H199 native action SFT contract: " + why)


def _point(value: object) -> list[int | float]:
    require(type(value) is list and len(value) == 2,
            "native absolute screenshot point must have two coordinates")
    for coordinate, extent in zip(value, (WIDTH, HEIGHT)):
        require(type(coordinate) in (int, float)
                and math.isfinite(coordinate)
                and 0 <= coordinate < extent,
                "native screenshot point is outside 960x720")
    return value


def _one_action(item: object, *, training_target: bool) -> dict:
    require(type(item) is dict and "kind" in item,
            "exact typed native action required")
    kind = item["kind"]
    require(type(kind) is str and kind in KINDS, "unknown native action")
    if kind == "abstain":
        require(set(item) == {"kind"} and not training_target,
                "abstain only as standalone model fallback")
    elif kind == "click":
        require(set(item) == {"kind", "point"}, "click must target visible coordinates")
        _point(item["point"])
    elif kind == "drag":
        require(set(item) == {"kind", "start", "end"}, "drag needs native start and end")
        _point(item["start"])
        _point(item["end"])
        require(item["start"] != item["end"], "zero-distance original drag forbidden")
    elif kind == "key_press":
        require(set(item) == {"kind", "point", "key"},
                "keyboard action requires a visually grounded focus point and key")
        _point(item["point"])
        require(type(item["key"]) is str and item["key"] in ALLOWED_KEYS,
                "unsafe/unregistered keyboard key")
    # Return only bounded normal JSON-compatible native action keys.
    return item


def parse_actions(value: object, *, training_target: bool = False) -> dict:
    """Strict one-step action parser, not a dataset/model qualification."""
    require(type(value) is str, "model must emit bounded JSON text")
    require(len(value.encode("utf-8")) <= MAX_JSON_BYTES,
            "excessive native JSON response")
    try:
        doc = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("H199 native action SFT contract: malformed native JSON") from exc
    require(type(doc) is dict and set(doc) == {"actions"},
            "only exact actions array permitted")
    items = doc["actions"]
    require(type(items) is list and len(items) == 1,
            "exactly one native action is supported")
    action = _one_action(items[0], training_target=training_target)
    require(action["kind"] != "abstain" or not training_target,
            "cannot train on abstain reference")
    return {"actions": [dict(action)]}


def source_task_prompt(task: str) -> str:
    """Only model-visible instruction, NEVER owner source gold/reference."""
    require(type(task) is str and 0 < len(task) <= 500,
            "single bounded user-visible original GUI task required")
    require("\x00" not in task and "\r" not in task,
            "unsafe instruction control characters")
    return (
        "You are Hala, a screenshot-only computer-use agent. "
        "Use only the 960x720 screenshot and the user's task, never DOM, "
        "hidden HTML or reference labels. Reply ONLY in JSON with one "
        "{\"actions\":[...]} action. A click is "
        "{\"kind\":\"click\",\"point\":[x,y]}; a drag is "
        "{\"kind\":\"drag\",\"start\":[x,y],\"end\":[x,y]}; "
        "to focus a visible control and press a key use "
        "{\"kind\":\"key_press\",\"point\":[x,y],\"key\":\"ArrowRight\"}, "
        "with key in ArrowLeft/ArrowRight/ArrowUp/ArrowDown/Enter/"
        "Escape/Tab/Space. All coordinates are screenshot pixels. "
        "If unsafe or uncertain, output {\"actions\":[{\"kind\":\"abstain\"}]}. "
        "Task: " + task
    )


def _hex_sha(value: object, n: int) -> bool:
    return type(value) is str and len(value) == n and all(
        c in "0123456789abcdef" for c in value
    )


def verify_real_training_release(release: dict) -> None:
    """Fail-closed until an actual independently approved new release exists.

    No conditional/test bypass, no implicit promotion of 20 quarantined cases.
    """
    require(type(release) is dict and set(release) == {
        "schema", "manifest_sha256", "split", "official_admitted",
        "complete_source_history_proven", "complete_old_rgba_proven",
        "sealed_dev_gold", "consumed_ood_reused",
    }, "unrecognized or incomplete frozen training release")
    require(release["schema"] == "hala.original-native-train-release.v1"
            and release["split"] == "TRAIN"
            and release["official_admitted"] is True
            and release["complete_source_history_proven"] is True
            and release["complete_old_rgba_proven"] is True
            and release["sealed_dev_gold"] is True
            and release["consumed_ood_reused"] is False,
            "unadmitted original GUI data, incomplete history or OOD leakage")
    manifest = release["manifest_sha256"]
    require(_hex_sha(manifest, 64) and manifest in ADMITTED_TRAIN_MANIFEST_SHAS,
            "no independently pinned and qualified TRAIN release in H199")


def training_messages_for_synthetic_contract(
    instruction: str, one_native_action_json: str,
) -> tuple[list[dict], list[dict]]:
    """Synthetic CPU contract shape only. NOT an admitted dataset batch."""
    native = parse_actions(one_native_action_json, training_target=True)
    content = [
        {"type": "image"},
        {"type": "text", "text": source_task_prompt(instruction)},
    ]
    question = [{"role": "user", "content": content}]
    answer = question + [{
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(
            native, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )}],
    }]
    return question, answer


def assistant_completion_mask(
    prompt_token_ids: list[int], full_token_ids: list[int],
) -> list[int]:
    """Copy-first H161 completion-only masking, pure CPU transferable primitive.

    Real HF processor/tokenizer must independently prove identical prefix
    tokenization; the synthetic smoke cannot establish actual model parity.
    """
    require(type(prompt_token_ids) is list
            and type(full_token_ids) is list
            and all(type(v) is int and v >= 0 for v in
                    prompt_token_ids + full_token_ids),
            "tokenizer token arrays invalid")
    count = len(prompt_token_ids)
    require(count >= 1 and len(full_token_ids) >= count + 5,
            "missing multimodal user prefix or supervised completion")
    require(full_token_ids[:count] == prompt_token_ids,
            "multimodal assistant tokens differ from prompt prefix")
    result = [-100]*count + full_token_ids[count:]
    require(result[:count] == [-100]*count
            and all(v >= 0 for v in result[count:]),
            "assistant-only loss mask not safely aligned")
    return result


def synthetic_cpu_sft_dry_run() -> dict:
    """No source files or model: synthetic one-example-per-action CPU proof."""
    samples = (
        ("Click the visible Save button.",
         {"kind": "click", "point": [300, 250]}),
        ("Drag the visible card to the destination.",
         {"kind": "drag", "start": [225, 280], "end": [440, 300]}),
        ("Increase the selected slider one step.",
         {"kind": "key_press", "point": [512, 260], "key": "ArrowRight"}),
    )
    actions = set()
    for i, (task, action) in enumerate(samples):
        payload = json.dumps({"actions": [action]})
        q, a = training_messages_for_synthetic_contract(task, payload)
        require(q[0]["content"][0] == {"type": "image"}
                and a[-1]["role"] == "assistant",
                "synthetic multimodal conversation shape wrong")
        require("reference" not in q[0]["content"][-1]["text"].lower()
                or "reference labels" in q[0]["content"][-1]["text"].lower(),
                "synthetic source goal leaked into prompt")
        # Fake stable IDs: exactly tests prefix-mask semantics, NOT HF inference.
        p = [i+1, 10, 20, 30, 40]
        full = p + [31, 32, 33, 34, 35, 36]
        masked = assistant_completion_mask(p, full)
        require(masked[:len(p)] == [-100]*len(p)
                and masked[len(p):] == full[len(p):],
                "synthetic completion-only loss check failed")
        actions.add(action["kind"])
    require(actions == {"click", "drag", "key_press"},
            "required three native source action kinds missing")
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "H199 synthetic smoke may not enable any real training")
    return {
        "status": "H199_SYNTHETIC_CPU_NATIVE_SFT_CONTRACT_PASS",
        "synthetic_action_examples": len(samples),
        "supported_kinds": sorted(KINDS),
        "real_original_train_cases_admitted": 0,
        "real_private_images_opened": 0,
        "real_private_gold_opened": 0,
        "gpu_optimizer_steps": 0,
        "model_calls": 0,
        "old_ood_accessed": False,
    }


if __name__ == "__main__":
    result = synthetic_cpu_sft_dry_run()
    print(
        f"{result['status']} "
        f"synthetic={result['synthetic_action_examples']} "
        "CLICK_DRAG_KEYPRESS ABSTAIN_FALLBACK_ONLY "
        "NO_REAL_TRAIN_NO_MEDIA_NO_OOD_NO_GPU"
    )
