"""Independent CPU replay of H25 DEV-only model run, hash-verified raw records.

Verifies identity and scores. Does not attest physical GPU, commercial cost,
true held-out performance, real Orcel actions, training, or model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

from h25 import baseline
from h24.mind_dev import MODEL_ID, MODEL_REVISION, strict_json
from h25 import projection_v2


def inspect(path, expected_sha256, source_commit):
    if (type(expected_sha256) is not str or len(expected_sha256) != 64 or
            any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise ValueError("Observed raw SHA256 lowercase hex required")
    if not baseline.SOURCE_SHA.fullmatch(source_commit):
        raise ValueError("Immutable SHA40 source identity required")
    file = Path(path)
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Regular bounded downloaded receipt required")
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Raw downloaded byte identity mismatch")
    doc = strict_json(raw.decode("utf-8"))
    if type(doc) is not dict:
        raise ValueError("Malformed result record")
    pin = {
        "schema": baseline.SCHEMA, "protocol": baseline.PROTOCOL,
        "status": "measured_unqualified", "source_commit": source_commit,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "source_cases_sha256": projection_v2.H25_PIN,
        "projection_sha256": projection_v2.validate(projection_v2.candidate())["projection_sha256"],
        "provider": "kaggle", "gpu_name": "Tesla T4",
        "precision": "fp16", "split": "dev", "training_runs": 0,
        "device_actions_executed": False, "model_adopted": False,
        "task_cases": 42,
    }
    for key, wanted in pin.items():
        if type(doc.get(key)) is not type(wanted) or doc.get(key) != wanted:
            raise ValueError("Unreviewed experimental identity/claim: " + key)
    for key in ("load_seconds", "cuda_allocated_after_load_bytes", "cuda_peak_allocated_bytes"):
        value = doc.get(key)
        if key == "load_seconds":
            if type(value) not in (float, int) or not 0 <= value < 5400:
                raise ValueError("Model loading duration not bounded")
        elif type(value) is not int or not 0 <= value < 16 * 1024**3:
            raise ValueError("Reported CUDA allocation invalid")
    if doc["cuda_peak_allocated_bytes"] < doc["cuda_allocated_after_load_bytes"]:
        raise ValueError("Reported allocator values contradict")
    rows = doc.get("outcomes")
    if type(rows) is not list or len(rows) != 42:
        raise ValueError("Missing/partial or repeated H25 results")
    paired = baseline.expected_rows()
    if [row.get("id") for row in rows] != [row["source_ref"]["id"] for row, _, _ in paired]:
        raise ValueError("DEV source case order/identity changed")
    for record, (ref, oracle, state) in zip(rows, paired):
        source = ref["source_ref"]
        for key, expected in (("id", source["id"]), ("language", source["language"]),
                              ("family", source["family"]), ("state", state)):
            if record.get(key) != expected:
                raise ValueError("Result/source family mismatch: " + key)
        seconds = record.get("wall_seconds")
        if type(seconds) not in (float, int) or not 0 <= seconds < 5400:
            raise ValueError("Missing/bad case wall time")
        if record.get("raw") is None:
            if (record.get("error") != "model_inference_error" or
                    record.get("valid") is not False or
                    record.get("tool_correct") is not False or
                    record.get("exact_correct") is not False or
                    record.get("predicted_tool") is not None or
                    type(record.get("error_type")) is not str or
                    not 0 < len(record["error_type"]) <= 100):
                raise ValueError("Raw failure mislabeled as success")
        else:
            if type(record["raw"]) is not str or len(record["raw"]) > 16384:
                raise ValueError("Unbounded model output")
            calculated = baseline.evaluate_raw(oracle, record["raw"])
            for key, expected in calculated.items():
                if record.get(key) != expected:
                    raise ValueError("Model output falsely scored: " + key)
    recomputed = baseline.summarize(rows)
    if doc.get("summary") != recomputed:
        raise ValueError("Summary must match independent per-case score")
    return {
        "schema": "hala.h25.dev_raw_independent_cpu_audit.v1",
        "exact_raw_sha256": expected_sha256, "source_commit": source_commit,
        "source_cases_sha256": projection_v2.H25_PIN,
        "protocol": baseline.PROTOCOL,
        "all_42_dev_recomputed": True,
        "bilingual_metrics": recomputed,
        "training_performed": False, "real_device_execution": False,
        "model_approved": False, "independently_attested_physical_gpu": False,
        "limitations": "42 public synthetic correlated DEV examples; replay is not a real-world agent quality test",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_result")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    proof = inspect(a.raw_result, a.expected_sha256, a.source_commit)
    with Path(a.output).open("x", encoding="utf-8") as f:
        json.dump(proof, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(proof, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
