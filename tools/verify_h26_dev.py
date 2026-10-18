"""H26 exact-raw-file CPU audit; no model load or physical GPU attestation."""
import argparse
import hashlib
import json
from pathlib import Path

from h24.mind_dev import MODEL_ID,MODEL_REVISION,strict_json
from h26 import runner,choice


def inspect(path,expected_sha256,source_commit):
    if (type(expected_sha256) is not str or len(expected_sha256)!=64 or
            any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise ValueError("Externally observed raw SHA256 lowercase hex required")
    if (type(source_commit) is not str or len(source_commit)!=40 or
            any(ch not in "0123456789abcdef" for ch in source_commit)):
        raise ValueError("Exact source commit SHA40 required")
    file=Path(path)
    if file.is_symlink() or not file.is_file() or file.stat().st_size>4*1024*1024:
        raise ValueError("H26 regular bounded raw result required")
    contents=file.read_bytes()
    if hashlib.sha256(contents).hexdigest()!=expected_sha256:
        raise ValueError("H26 raw bytes/hash mismatch")
    record=strict_json(contents.decode("utf-8"))
    if type(record) is not dict:
        raise ValueError("H26 raw JSON object required")
    expected={
        "schema":runner.SCHEMA,"status":"measured_unqualified",
        "protocol":runner.RUN_PROTOCOL,"source_commit":source_commit,
        "data_sha256":runner.EXACT_DATA_SHA,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "gpu_name":"Tesla T4","provider":"kaggle","precision":"fp16",
        "partition":"dev","training_runs":0,
        "native_device_actions_executed":False,"model_adopted":False,
    }
    for field,value in expected.items():
        if type(record.get(field)) is not type(value) or record.get(field)!=value:
            raise ValueError("H26 raw experimental identity/claim mismatch: "+field)
    for key in ("load_seconds","torch_allocated_after_load_bytes","torch_peak_allocated_bytes"):
        value=record.get(key)
        if key=="load_seconds":
            if type(value) not in (int,float) or not 0<=value<5400:
                raise ValueError("Invalid H26 remote load_seconds")
        elif type(value) is not int or not 0<=value<16*1024**3:
            raise ValueError("Reported torch allocator invalid")
    if record["torch_peak_allocated_bytes"]<record["torch_allocated_after_load_bytes"]:
        raise ValueError("Inconsistent allocator counts")
    original=runner.validated_rows()
    outcomes=record.get("outcomes")
    if type(outcomes) is not list or len(outcomes)!=42 or [x.get("id") for x in outcomes]!=[r["source_ref"]["id"] for r in original]:
        raise ValueError("H26 expected exact 42 ordered source cases")
    for current,case in zip(outcomes,original):
        reference=case["source_ref"]
        for key,value in (("family",reference["family"]),("state",reference["state"]),
                          ("language",reference["language"])):
            if current.get(key)!=value:
                raise ValueError("H26 per-case provenance mismatch")
        wall=current.get("wall_seconds")
        if type(wall) not in (int,float) or not 0<=wall<5400:
            raise ValueError("Bad per-case remote duration")
        if current.get("raw") is None:
            if (current.get("error")!="model_inference_error" or
                current.get("format_valid") is not False or
                current.get("withheld") is not False or
                current.get("semantic_correct") is not False or
                current.get("suggestion") is not None or
                type(current.get("error_type")) is not str or
                not 0<len(current["error_type"])<=100):
                raise ValueError("H26 model exception incorrectly accepted")
        else:
            if type(current["raw"]) is not str or len(current["raw"])>200:
                raise ValueError("H26 unbounded reply")
            verdict=choice.score(current["raw"],case["private_reference"]["typed"])
            if (current.get("error") is not None or current.get("error_type") is not None or
                    any(current.get(k)!=v for k,v in verdict.items())):
                raise ValueError("H26 forged per-case score/metadata")
    replay=runner.aggregate(outcomes)
    if record.get("summary")!=replay:
        raise ValueError("H26 unsupported aggregate claims")
    return {
        "schema":"hala.h26.cpu_exact_raw_audit.v1",
        "source_commit":source_commit,
        "raw_sha256":expected_sha256,
        "fixed_source_sha256":runner.EXACT_DATA_SHA,
        "all_42_recomputed":True,
        "language_state_metrics":replay,
        "format_validity_is_not_semantic_success":True,
        "hardware_attested_independently":False,
        "actual_gui_tasks_executed":False,
        "production_model_approved":False,
        "independent_holdout_test":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("raw_result")
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--source-commit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    receipt=inspect(args.raw_result,args.expected_sha256,args.source_commit)
    with Path(args.output).open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,ensure_ascii=False,sort_keys=True)
        f.write("\n")
    print(json.dumps(receipt,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
