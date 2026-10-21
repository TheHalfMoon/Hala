"""H56 receipt consistency verifier; does NOT attest external GPU independently.

Input: unmodified actual notebook run-result.json + raw-attempts.json.
This cannot certify model weights, only detect altered/mismatched claims before
actual Windows Edge task replay. No model, network or trainer runs here.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from h56.real_drag_baseline import (MODEL_SCHEMA,MODEL_ID,MODEL_REVISION,DEV_SOURCE_SHA,
                                     ATTEMPTS_SCHEMA,PROTOCOL,source_inputs)

def sha(b):return hashlib.sha256(b).hexdigest()
def unique_pairs(pairs):
    obj={}
    for k,v in pairs:
        if k in obj:raise ValueError("Duplicate JSON keys")
        obj[k]=v
    return obj
def load(path):
    path=Path(path)
    if path.stat().st_size>1024*1024:raise ValueError("Unbounded original raw result")
    return json.loads(path.read_text("utf8"),object_pairs_hook=unique_pairs,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))

def verify(public_dev,notebook_run,raw_attempts):
    scenes=source_inputs(public_dev)
    a=load(notebook_run)
    r=load(raw_attempts)
    if (a.get("schema")!=MODEL_SCHEMA or a.get("model_id")!=MODEL_ID or
        a.get("revision")!=MODEL_REVISION or a.get("status")!="completed" or
        a.get("source_sha256")!=DEV_SOURCE_SHA or a.get("coordinate_protocol")!=PROTOCOL or
        a.get("split")!="public_dev" or a.get("blind_test") is not False or
        a.get("model_training_executed") is not False or a.get("adapters_loaded") is not False or
        a.get("provider")!="interactive_approved_free_colab" or
        a.get("evidence")!="real_inference" or
        a.get("independent_model_performance_qualified") is not False or
        type(a.get("max_cases")) is not int or a["max_cases"]!=24 or
        type(a.get("vram_bytes")) is not int or a["vram_bytes"]<=0 or
        not isinstance(a.get("gpu"),str) or not a["gpu"] or
        type(a.get("peak_vram_bytes")) is not int or a["peak_vram_bytes"]<=0 or
        a.get("package_versions",{}).get("transformers")!="5.16.1" or
        not isinstance(a.get("real_generated_cases"),list) or len(a["real_generated_cases"])!=24 or
        a.get("raw_attempts_sha256")!=sha(Path(raw_attempts).read_bytes()) or
        r.get("schema")!=ATTEMPTS_SCHEMA or r.get("source_sha256")!=DEV_SOURCE_SHA or
        r.get("origin")!="provider_raw_model_outputs" or
        not isinstance(r.get("cases"),list) or len(r["cases"])!=24):
        raise ValueError("Unqualified H56 original model-evidence manifest")
    for src,case,raw in zip(scenes,a["real_generated_cases"],r["cases"]):
        prompt_sha=sha(src["prompt"].encode("utf8"))
        value=case.get("output_raw")
        if (not isinstance(value,str) or len(value)>8192 or
            case.get("id")!=src["id"] or case.get("image_sha256")!=src["image_sha256"] or
            case.get("language")!=src["language"] or case.get("input_prompt_sha256")!=prompt_sha or
            case.get("output_sha256")!=sha(value.encode("utf8")) or
            case.get("status")!="real_neural_raw" or
            type(case.get("elapsed_seconds")) not in (int,float) or
            not math.isfinite(case["elapsed_seconds"]) or case["elapsed_seconds"]<=0 or
            type(case.get("output_tokens")) is not int or not 0<=case["output_tokens"]<=192 or
            raw!=dict(id=src["id"],image_sha256=src["image_sha256"],raw_output=value)):
            raise ValueError("Real output changed or detached from source/frozen prompt")
    return dict(schema="hala.h56.raw-inference-receipt-consistency.v1",
                source_sha256=DEV_SOURCE_SHA,exact_model_revision=MODEL_REVISION,
                source_bound_real_raw_outputs=24,arabic=12,english=12,
                run_sha256=sha(Path(notebook_run).read_bytes()),
                attempts_sha256=sha(Path(raw_attempts).read_bytes()),
                consistency_verified=True,
                actual_remote_gpu_independently_attested=False,
                edge_task_success_verified=False,
                blind_test_qualified=False,
                model_training_executed=False)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="h55/assets")
    p.add_argument("--run",required=True)
    p.add_argument("--attempts",required=True)
    opts=p.parse_args()
    print(json.dumps(verify(opts.source,opts.run,opts.attempts),sort_keys=True))
