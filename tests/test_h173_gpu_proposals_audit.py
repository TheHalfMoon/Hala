"""H173 GPU auditor proves only frozen 64 private T4 proposals, never success."""
import hashlib
import json
from pathlib import Path

import pytest

from h173 import audit_gpu
from h173.infer_gpu import MODEL_PROPOSALS_SCHEMA
from h173.source_preflight import preregister_provisional_cases


def private_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_gpu, "SOURCE_SHA", "b" * 40)
    monkeypatch.setattr(audit_gpu, "OOD_ZIP_SHA", "a" * 64)
    folder = tmp_path / "hala-h173-private-paired-ood"
    folder.mkdir()
    cases = preregister_provisional_cases()
    rows = []
    for role in ("before", "after"):
        for case in cases:
            raw = '{"actions":[{"kind":"click","point":[300,250]}]}'
            rows.append({
                "role": role, "case_index": case["index"],
                "image_sha256": hashlib.sha256(
                    str(case["index"]).encode()
                ).hexdigest(),
                "app": case["app"], "language": case["language"],
                "action_type": "click",
                "raw": raw,
                "parsed": {
                    "valid": True,
                    "kind": "click",
                    "points": [[300, 250]],
                },
            })
    job = {
        "schema": audit_gpu.JOB_SCHEMA,
        "status": "H173_PRIVATE_64_FROZEN_UNSEEN_OOD_T4_PROPOSALS_PASS",
        "source_commit": "b" * 40,
        "optimizer_steps": 0,
        "real_gpu_calls": 64,
        "owner_gold_uploaded": False,
        "old_dev_ood_used": False,
        "paid_gpu": False,
        "weights_published": False,
        "flagship_35b_unchanged": True,
        "peak_allocated_gib": 4.5,
    }
    result = {
        "schema": MODEL_PROPOSALS_SCHEMA,
        "status": "H173_PRIVATE_64_REAL_GPU_UNSEEN_OOD_PROPOSALS_COMPLETE",
        "source_ood_zip_sha256": "a" * 64,
        "frozen_language_sha256": audit_gpu.LANGUAGE_WEIGHTS_SHA,
        "trained_vision_sha256": audit_gpu.VISION_WEIGHTS_SHA,
        "base_model": audit_gpu.MODEL_ID,
        "base_revision": audit_gpu.MODEL_SHA,
        "paired_ood_cases": 32,
        "real_gpu_calls": 64,
        "counts": {"before": 32, "after": 32},
        "optimizer_steps": 0,
        "owner_gold_on_gpu": False,
        "previous_consumed_dev_ood_used": False,
        "real_browser_reward_measured": False,
        "paid_gpu": False,
        "weights_published": False,
        "flagship_35b_unchanged": True,
        "peak_allocated_gib": 4.5,
        "memory_cap_gib": audit_gpu.MAX_PEAK_GIB,
        "gpu": "Tesla T4",
        "versions": {
            "torch": audit_gpu.EXPECTED_TORCH,
            "transformers": audit_gpu.EXPECTED_TRANSFORMERS,
            "peft": audit_gpu.EXPECTED_PEFT,
            "Pillow": "11.3.0",
        },
        "predictions": rows,
    }
    return tmp_path, job, result


def write_receipts(root, job, result):
    (root / "h173-paired-ood-job.json").write_text(
        json.dumps(job), encoding="utf8"
    )
    (root / "hala-h173-private-paired-ood" /
     "h173-paired-ood-predictions.json").write_text(
        json.dumps(result), encoding="utf8"
    )


def test_exact_all_64_frozen_gpu_proposals_audited(tmp_path, monkeypatch):
    root, job, result = private_receipts(tmp_path, monkeypatch)
    write_receipts(root, job, result)
    audited = audit_gpu.audit(root)
    assert audited["status"].endswith("PROPOSALS_VERIFIED")
    assert audited["model_calls"] == 64
    assert audited["native_original_browser_reward_measured"] is False


@pytest.mark.parametrize("mutation", [
    lambda j, r: j.update(source_commit="unqualified"),
    lambda j, r: j.update(real_gpu_calls=63),
    lambda j, r: j.update(owner_gold_uploaded=True),
    lambda j, r: r.update(real_browser_reward_measured=True),
    lambda j, r: r.update(paid_gpu=True),
    lambda j, r: r.update(optimizer_steps=1),
    lambda j, r: r.update(source_ood_zip_sha256="0"*64),
    lambda j, r: r.update(peak_allocated_gib=14.0),
    lambda j, r: r["predictions"].pop(),
    lambda j, r: r["predictions"][1].update(role="after"),
    lambda j, r: r["predictions"][0].update(raw="not json"),
    lambda j, r: r["predictions"][0].update(app="Pell"),
    lambda j, r: r["predictions"][0].update(language="xx"),
    lambda j, r: r["predictions"][32].update(image_sha256="0"*64),
    lambda j, r: r["predictions"][1].update(
        image_sha256=r["predictions"][0]["image_sha256"]
    ),
])
def test_incomplete_or_contaminated_free_gpu_receipt_fails(
    tmp_path, monkeypatch, mutation
):
    root, job, result = private_receipts(tmp_path, monkeypatch)
    mutation(job, result)
    write_receipts(root, job, result)
    with pytest.raises(ValueError):
        audit_gpu.audit(root)


def test_model_ood_inference_source_sha_must_be_frozen(tmp_path, monkeypatch):
    root, job, result = private_receipts(tmp_path, monkeypatch)
    write_receipts(root, job, result)
    monkeypatch.setattr(audit_gpu, "SOURCE_SHA", "UNQUALIFIED")
    with pytest.raises(ValueError):
        audit_gpu.audit(root)
