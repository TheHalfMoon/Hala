"""H359: inspect original browser candidate rows as H199-native SFT messages.

NOT a training loader, adapter export, admission mechanism or model call.
This bridge explicitly refuses translation to the unrelated legacy
training.artifacts logical-action schema, which expects 'type' objects.
Input is original ephemeral owner-run QA data, never private OOD.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, parse_actions, source_task_prompt,
    training_messages_for_synthetic_contract,
)
from training.artifacts import (
    normalized_relative_path, strict_json, validate_action_completion,
)
from h357.source_candidate import SOURCE as CONTROL_SOURCE
from h358.source_candidate import SOURCE as NOTE_SOURCE

ROLE357 = "TRAIN_DEV_REVIEW_CANDIDATE_ONLY_NOT_ADMITTED"
ROLE358 = "ORIGINAL_MULTISTEP_TRAIN_DEV_REVIEW_CANDIDATE_NOT_ADMITTED"
SCHEMA = "hala.h359.native-source-review-bridge.v1"
SOURCES = {
    "controlgarden-single-app": {
        "role": ROLE357,
        "split": "TRAIN_DEV_CANDIDATE_NOT_ADMITTED",
        "schema": "hala.h357.controlgarden.original-candidate.v1",
        "source": CONTROL_SOURCE,
        "source_sha": "bf111dc50fe9083611a846016b35982b4e29bbd72eb1b556c989b00356600e93",
        "records": 64,
        "families": 1,
        "languages": {"en": 32, "ar": 32},
        "action_counts": {"click": 64},
    },
    "noterelay-original-multistep": {
        "role": ROLE358,
        "split": ROLE358,
        "schema": "hala.h358.noterelay.original-candidate.v1",
        "source": NOTE_SOURCE,
        "source_sha": "f40d285e3841d294f58bd410f1150f7aeaaea0a71d0e7ea4784b9786f2e860be",
        "records": 128,
        "families": 1,
        "languages": {"en": 64, "ar": 64},
        "action_counts": {"click": 96, "type_text": 32},
    },
}
SOURCE_ROW_COMMON = frozenset({
    "id", "family", "split", "language", "prompt", "completion",
    "image_path", "image_sha256", "original_source_sha256",
})
SOURCE_ROW358 = SOURCE_ROW_COMMON | {"phase"}
PREFIX_SENTINEL = "H359-ORIGINAL-TASK-SENTINEL"
PREFIX = source_task_prompt(PREFIX_SENTINEL).removesuffix(PREFIX_SENTINEL)
REPORT_KEYS = ("schema", "role", "family", "records", "en", "ar",
               "action_counts", "distinct_images", "legacy_schema_compatible",
               "h199_messages_checked", "official_train_admitted",
               "independent_rights_verified", "can_train",
               "blind_ood", "can_release")


def need(ok: bool):
    if not ok:
        raise ValueError("H359_UNADMITTED_NATIVE_SFT_REVIEW_REFUSED")


def safe_regular(root: Path, file: str) -> Path:
    relative = normalized_relative_path(file)
    target = root.joinpath(*relative.parts)
    need(target.resolve().is_relative_to(root.resolve()))
    need(all(not item.is_symlink()
             for item in (target, *target.parents)
             if item != root.parent))
    need(target.is_file())
    return target


def audit_review_bundle(root: Path, family: str) -> dict:
    """Validate exact native action and media WITHOUT admitting or exporting it.

    Creates no persistent file and returns only public-safe aggregate stats.
    The approved H199 TRAIN digest remains empty regardless of source quality.
    """
    need(isinstance(root, Path) and root.is_dir() and not root.is_symlink()
         and family in SOURCES)
    need(not ADMITTED_TRAIN_MANIFEST_SHAS)
    expected = SOURCES[family]
    actual_source = expected["source"].read_bytes()
    need(sha256(actual_source).hexdigest() == expected["source_sha"])
    mf = safe_regular(root, "candidate-manifest.json")
    need(mf.stat().st_size <= 4096)
    report = strict_json(mf.read_text(encoding="utf-8"))
    need(type(report) is dict
         and report.get("schema") == expected["schema"]
         and report.get("role") == expected["role"]
         and report.get("original_source_sha256") == expected["source_sha"])
    for key in ("official_train_admitted", "fresh_blind_ood",
                "can_train", "can_release"):
        need(report.get(key) is False)
    if family == "controlgarden-single-app":
        need(report.get("independent_rights_reviewed") is False
             and report.get("case_count") == 64
             and report.get("language_counts") == expected["languages"]
             and report.get("original_native_browser_successes") == 64
             and report.get("exact_pixel_double_captures") == 64
             and report.get("families") == [family])
    else:
        need(report.get("independent_rights_approved") is False
             and report.get("has_source_disjoint_dev") is False
             and report.get("rows") == 128
             and report.get("case_count") == 32
             and report.get("steps_per_case") == 4
             and report.get("language_cases") == {"en": 16, "ar": 16}
             and report.get("original_native_state_successes") == 128
             and report.get("stable_initial_screenshots") == 32
             and report.get("family") == family)
    examples = safe_regular(root, "candidate-rows.jsonl")
    need(examples.stat().st_size <= 500_000)
    entries = examples.read_bytes().splitlines()
    need(len(entries) == expected["records"])
    ids, digests = set(), set()
    actions, langs = Counter(), Counter()
    # Import Pillow lazily only after metadata, family/source and rights flags
    # have been validated, and never initialize a HF/torch model.
    from io import BytesIO
    from PIL import Image

    for raw in entries:
        need(10 <= len(raw) <= 8192)
        row = strict_json(raw.decode("utf-8"))
        keys = SOURCE_ROW358 if family == "noterelay-original-multistep" else SOURCE_ROW_COMMON
        need(type(row) is dict and set(row) == keys
             and row["family"] == family
             and row["split"] == expected["split"]
             and row["language"] in ("en", "ar")
             and type(row["id"]) is str and 8 <= len(row["id"]) <= 90
             and row["id"] not in ids
             and row["original_source_sha256"] == expected["source_sha"]
             and type(row["image_sha256"]) is str
             and len(row["image_sha256"]) == 64
             and row["image_sha256"] not in digests
             and type(row["prompt"]) is str
             and row["prompt"].startswith(PREFIX)
             and len(row["prompt"]) <= 8192)
        task = row["prompt"][len(PREFIX):]
        need(task and source_task_prompt(task) == row["prompt"])
        native = parse_actions(row["completion"], training_target=True)
        kind = native["actions"][0]["kind"]
        need(kind in expected["action_counts"])
        if family == "noterelay-original-multistep":
            need(row["phase"] in ("open", "focus", "type", "save")
                 and kind == ("type_text" if row["phase"] == "type" else "click"))
        # This is the key divergence: legacy logical-action SFT must REFUSE
        # original H199 screenshot-only action JSON, never silently translate.
        try:
            validate_action_completion(row["completion"])
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise ValueError("H359_LEGACY_ACTION_SCHEMA_UNEXPECTEDLY_ACCEPTED")
        question, answer = training_messages_for_synthetic_contract(
            task, row["completion"])
        need(question[0]["content"] == [
            {"type": "image"}, {"type": "text", "text": row["prompt"]}]
            and answer[:-1] == question
            and answer[-1]["role"] == "assistant"
            and parse_actions(answer[-1]["content"][0]["text"],
                              training_target=True) == native)
        image_file = safe_regular(root, row["image_path"])
        need(image_file.parent.name == "images"
             and image_file.stat().st_size <= 4_000_000)
        raw_image = image_file.read_bytes()
        need(sha256(raw_image).hexdigest() == row["image_sha256"])
        with Image.open(BytesIO(raw_image)) as img:
            need(img.format == "PNG" and img.size == (960, 720))
            img.verify()
        ids.add(row["id"])
        digests.add(row["image_sha256"])
        actions[kind] += 1
        langs[row["language"]] += 1
    need(dict(actions) == expected["action_counts"]
         and dict(langs) == expected["languages"])
    need(len(ids) == len(digests) == expected["records"])
    return {
        "schema": SCHEMA,
        "role": "SOURCE_ONLY_NATIVE_MESSAGES_AUDIT_NO_TRAIN",
        "family": family,
        "records": len(ids), "en": langs["en"], "ar": langs["ar"],
        "action_counts": dict(sorted(actions.items())),
        "distinct_images": len(digests),
        "legacy_schema_compatible": False,
        "h199_messages_checked": len(ids),
        "official_train_admitted": False,
        "independent_rights_verified": False,
        "can_train": False, "blind_ood": False, "can_release": False,
    }
