"""Independent H173 PRIVATE FREE-T4 genuine unseen OOD GPU proposal auditor.

No model execution and no owner gold. Model proposals are NOT native
browser task successes. Only compare owner-native reward once separately.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h132.grounder_contract import MODEL_ID, MODEL_SHA
from h164.native_action_contract import parse_action
from h166.vision_probe import (
    EXPECTED_PEFT,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    MAX_PEAK_GIB,
)
from h167.combined_probe import LANGUAGE_WEIGHTS_SHA
from h172.infer_gpu import VISION_WEIGHTS_SHA
from h173.infer_gpu import (
    MODEL_PROPOSALS_SCHEMA, OOD_ZIP_SHA
)
from h173.source_preflight import OOD_COUNT, preregister_provisional_cases
from h173.source_pin import pinned_source_commit, validate_source_commit

# Owner-provided expected SHA is external to the immutable Git source.
# SOURCE_SHA retained for existing isolated negative unit tests only.
SOURCE_SHA = "UNQUALIFIED"
JOB_SCHEMA = "hala.h173.owner-private-free-t4-unseen-ood-job.v1"


def audit(output, expected_source_sha=None):
    # Independent local audit needs the owner's immutable Git SHA as input.
    # A record cannot corroborate its own untrusted SHA.
    source_sha = validate_source_commit(
        expected_source_sha
        if expected_source_sha is not None
        else SOURCE_SHA if SOURCE_SHA != "UNQUALIFIED" else pinned_source_commit()
    )
    root = Path(output)
    job = json.loads((root / "h173-paired-ood-job.json").read_text("utf8"))
    result = json.loads(
        (root / "hala-h173-private-paired-ood"
         / "h173-paired-ood-predictions.json").read_text("utf8")
    )
    if (
        len(OOD_ZIP_SHA) != 64
        or job.get("schema") != JOB_SCHEMA
        or job.get("status") != "H173_PRIVATE_64_FROZEN_UNSEEN_OOD_T4_PROPOSALS_PASS"
        or job.get("source_commit") != source_sha
        or job.get("optimizer_steps") != 0
        or job.get("real_gpu_calls") != 64
        or job.get("owner_gold_uploaded") is not False
        or job.get("old_dev_ood_used") is not False
        or job.get("paid_gpu") is not False
        or job.get("weights_published") is not False
        or job.get("flagship_35b_unchanged") is not True
    ):
        raise ValueError("H173 private pinned original OOD GPU job unqualified")
    if (
        result.get("schema") != MODEL_PROPOSALS_SCHEMA
        or result.get("status")
        != "H173_PRIVATE_64_REAL_GPU_UNSEEN_OOD_PROPOSALS_COMPLETE"
        or result.get("source_ood_zip_sha256") != OOD_ZIP_SHA
        or result.get("frozen_language_sha256") != LANGUAGE_WEIGHTS_SHA
        or result.get("trained_vision_sha256") != VISION_WEIGHTS_SHA
        or result.get("base_model") != MODEL_ID
        or result.get("base_revision") != MODEL_SHA
        or result.get("paired_ood_cases") != OOD_COUNT
        or result.get("real_gpu_calls") != OOD_COUNT * 2
        or result.get("counts") != {"before": OOD_COUNT, "after": OOD_COUNT}
        or result.get("optimizer_steps") != 0
        or result.get("owner_gold_on_gpu") is not False
        or result.get("previous_consumed_dev_ood_used") is not False
        or result.get("real_browser_reward_measured") is not False
        or result.get("paid_gpu") is not False
        or result.get("weights_published") is not False
        or result.get("flagship_35b_unchanged") is not True
        or result.get("versions") != {
            "torch": EXPECTED_TORCH,
            "transformers": EXPECTED_TRANSFORMERS,
            "peft": EXPECTED_PEFT,
            "Pillow": "11.3.0",
        }
        or "T4" not in result.get("gpu", "")
    ):
        raise ValueError("H173 model and no-gold original OOD source integrity failed")
    peak = result.get("peak_allocated_gib")
    if (
        type(peak) not in (float, int)
        or not math.isfinite(peak)
        or not 0 < peak <= MAX_PEAK_GIB
        or peak != job.get("peak_allocated_gib")
        or result.get("memory_cap_gib") != MAX_PEAK_GIB
    ):
        raise ValueError("H173 FREE Nvidia T4 allocation evidence invalid")
    proposals = result.get("predictions")
    if not isinstance(proposals, list) or len(proposals) != OOD_COUNT * 2:
        raise ValueError("H173 64 paired genuine OOD predictions required")
    specs = {item["index"]: item for item in preregister_provisional_cases()}
    seen, stats = set(), Counter()
    source_image_by_index = {}
    index_by_source_image = {}
    for item in proposals:
        role, ix = item.get("role"), item.get("case_index")
        spec = specs.get(ix)
        key = (ix, role)
        if (
            spec is None
            or role not in ("before", "after")
            or key in seen
            or item.get("app") != spec["app"]
            or item.get("language") != spec["language"]
            or item.get("action_type") != "click"
            or type(item.get("image_sha256")) is not str
            or len(item["image_sha256"]) != 64
            or item.get("parsed") != parse_action(item.get("raw"))
        ):
            raise ValueError("H173 GPU proposals malformed/duplicated/tainted")
        image_sha = item["image_sha256"]
        if ix in source_image_by_index:
            if source_image_by_index[ix] != image_sha:
                raise ValueError("H173 paired baseline and candidate did not see identical pixels")
        elif image_sha in index_by_source_image:
            raise ValueError("H173 GPU OOD accepted duplicated cross-case image identity")
        else:
            source_image_by_index[ix] = image_sha
            index_by_source_image[image_sha] = ix
        seen.add(key)
        stats[role + "/valid"] += int(item["parsed"]["valid"])
    if (
        seen != {
            (scene["index"], role)
            for scene in preregister_provisional_cases()
            for role in ("before", "after")
        }
        or len(source_image_by_index) != OOD_COUNT
        or len(index_by_source_image) != OOD_COUNT
    ):
        raise ValueError("H173 frozen paired original app OOD GPU coverage incomplete")
    return {
        "status": "H173_INDEPENDENT_64_PRIVATE_FREE_T4_UNSEEN_OOD_PROPOSALS_VERIFIED",
        "model_calls": OOD_COUNT * 2,
        "unique_original_unseen_ood_cases": OOD_COUNT,
        "before_schema_valid": stats["before/valid"],
        "after_schema_valid": stats["after/valid"],
        "peak_cuda_allocated_gib": peak,
        "optimizer_steps": 0,
        "native_original_browser_reward_measured": False,
        "previous_consumed_ood_used": False,
        "paid_gpu": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(
        audit(args.output, expected_source_sha=args.expected_source_sha),
        indent=2, sort_keys=True
    ))
