"""Independent CPU-only audit of downloaded H24 Kaggle Mind DEV result bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from h24.mind_dev import (H24_CASES_SHA256, MODEL_ID, MODEL_REVISION,
                          dev_cases, score, strict_json, summary)


def verify(path, expected_sha256, source_commit):
    if not re.fullmatch(r"[a-f0-9]{64}",expected_sha256):
        raise ValueError("External raw result SHA256 required")
    if not re.fullmatch(r"[a-f0-9]{40}",source_commit):
        raise ValueError("External reviewed source commit required")
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4*1024*1024:
        raise ValueError("Bounded, regular downloaded result file required")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError("Downloaded result byte identity mismatch")
    value=strict_json(raw.decode("utf-8"))
    identity={
        "schema":"hala.h24.mind_dev_baseline.v1","status":"measured_unqualified",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "source_commit":source_commit,"frozen_h24_sha256":H24_CASES_SHA256,
        "provider":"kaggle","precision":"fp16","split":"dev",
        "training_runs":0,"model_adopted":False,"native_qualified":False,
        "task_cases":20,"gpu_name":"Tesla T4",
    }
    for k,wanted in identity.items():
        if type(value.get(k)) is not type(wanted) or value.get(k)!=wanted:
            raise ValueError("Unexpected H24 result provenance: "+k)
    for key in ("load_seconds", "cuda_allocated_after_load_bytes", "cuda_peak_allocated_bytes"):
        v=value.get(key)
        if key=="load_seconds":
            if type(v) not in (int,float) or not 0<=v<5400:
                raise ValueError("Missing/invalid GPU load duration")
        elif type(v) is not int or not 0<=v<16*1024**3:
            raise ValueError("Missing/invalid CUDA allocator memory estimate")
    if value["cuda_peak_allocated_bytes"]<value["cuda_allocated_after_load_bytes"]:
        raise ValueError("CUDA allocation evidence inconsistent")
    actual=value.get("outcomes")
    prospective=dev_cases()
    if type(actual) is not list or len(actual)!=20:
        raise ValueError("Incomplete DEV results")
    rows=[]
    for reference,record in zip(prospective,actual):
        if type(record) is not dict or any(record.get(k)!=reference[k] for k in
            ("id","family","language","variant")):
            raise ValueError("DEV row order/family/source mismatch")
        if type(record.get("wall_seconds")) not in (int,float) or not (
                0 <= record["wall_seconds"] < 3600):
            raise ValueError("Invalid measured elapsed time")
        if record.get("raw") is None:
            if (record.get("error")!="model_inference_error" or
                    type(record.get("error_type")) is not str or
                    len(record["error_type"])>100 or
                    record.get("valid") is not False or record.get("tool_correct") is not False or
                    record.get("exact_correct") is not False):
                raise ValueError("Failed inference mislabeled as success")
        else:
            if type(record["raw"]) is not str or len(record["raw"])>16384:
                raise ValueError("Unbounded or non-text model output")
            expected=score(reference,record["raw"])
            if any(record.get(k)!=v for k,v in expected.items()):
                raise ValueError("Raw response, independent label and scoring disagree")
        rows.append(record)
    recomputed=summary(rows)
    if value.get("summary")!=recomputed:
        raise ValueError("Final summary differs from independently scored raw answers")
    return {
        "schema":"hala.h24.independent_dev_audit.v1",
        "source_commit":source_commit,"result_sha256":expected_sha256,
        "complete_dev_rows":20,"strata":recomputed,
        "raw_reply_scoring_verified":True,"external_hash_verified":True,
        "physical_gpu_independently_attested":False,
        "model_qualified":False,"native_orcel_qualified":False,
        "training_performed":False,
        "limitations":"20 synthetic dependent DEV cases, not application-level quality or Orcel execution",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("downloaded_result")
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--source-commit",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report=verify(a.downloaded_result,a.expected_sha256,a.source_commit)
    output=Path(a.output)
    with output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
