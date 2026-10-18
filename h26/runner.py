"""Hala H26 prospective constrained-choice DEV GPU research only.

No model training, native interaction or adoption. CPU tests prove only
machinery correctness, not model capability. Requires private free Kaggle.
"""
from collections import Counter
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

from h24.mind_dev import MODEL_ID, MODEL_REVISION
from h25 import baseline
from h26 import choice,corpus

SCHEMA="hala.h26.constrained_dev_measured.v1"
RUN_PROTOCOL="h26-code8-tokengate-dev-v1"
EXACT_DATA_SHA="3ba8f40d8c0579bed1e5ea6bfd3f9043e18c5d19f68f3f078d636308ea927a66"
REQUIRED_PACKAGES={"torch":"2.11.0+cu128","transformers":"5.16.1",
                   "accelerate":"1.12.0","safetensors":"0.8.0","Pillow":"11.3.0"}
CACHE="/tmp/hala-h26-dev-cache"


def validated_rows():
    source=corpus.dataset()
    manifest=corpus.validate(source)
    if manifest["sha256"]!=EXACT_DATA_SHA or len(source["records"])!=42:
        raise ValueError("H26 source data SHA changed")
    for row in source["records"]:
        if choice.reference_code(row["private_reference"]["typed"])!=row["private_reference"]["choice"]:
            raise ValueError("H26 source/oracle mismatch")
    return source["records"]


def remote_admit(source_sha, output):
    if (os.name!="posix" or not Path("/kaggle/working").is_dir() or
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")!="Batch" or
        os.environ.get("HALA_REMOTE_DEVELOPMENT_ALLOWED")!="1"):
        raise RuntimeError("H26 requires explicit private free Kaggle Batch")
    if type(source_sha) is not str or not re.fullmatch("[a-f0-9]{40}",source_sha):
        raise ValueError("Full exact SHA40 source required")
    root=Path(__file__).resolve().parents[1]
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    modified=subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=root,text=True).strip()
    if actual!=source_sha or modified:
        raise ValueError("Unreviewed H26 checkout or modified source")
    target=Path(output).resolve()
    if not target.is_relative_to(Path("/kaggle/working")) or target==Path("/kaggle/working") or target.exists():
        raise ValueError("H26 output must be a NEW child directory of Kaggle working")
    for name in ("HF_HOME","HF_HUB_CACHE","TORCH_HOME","XDG_CACHE_HOME"):
        if os.environ.get(name) not in (None,CACHE):
            raise ValueError("Unreviewed remote cache override: "+name)
        os.environ[name]=CACHE
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["DO_NOT_TRACK"]="1"
    for package,pinned in REQUIRED_PACKAGES.items():
        if importlib.metadata.version(package)!=pinned:
            raise RuntimeError("Wrong remote dependency version: "+package)
    rows=validated_rows()
    target.mkdir(parents=True,exist_ok=False)
    return rows,target


def aggregate(rows):
    original=validated_rows()
    if len(rows)!=42 or [x.get("id") for x in rows]!=[x["source_ref"]["id"] for x in original]:
        raise ValueError("Missing/duplicate/misordered H26 cases")
    for row,case in zip(rows,original):
        track=case["source_ref"]
        if row.get("language")!=track["language"] or row.get("family")!=track["family"] or row.get("state")!=track["state"]:
            raise ValueError("Unexpected H26 case reference")
        if row.get("raw") is None:
            valid={"format_valid":False,"withheld":False,"semantic_correct":False,"suggestion":None}
            if (row.get("error")!="model_inference_error" or
                type(row.get("error_type")) is not str or
                not 0<len(row["error_type"])<=100):
                raise ValueError("Missing genuine failure type")
        else:
            if type(row["raw"]) is not str or len(row["raw"])>200:
                raise ValueError("Unbounded raw H26 choice")
            if row.get("error") is not None or row.get("error_type") is not None:
                raise ValueError("Contradictory H26 model error")
            valid=choice.score(row["raw"],case["private_reference"]["typed"])
        if any(row.get(key)!=value for key,value in valid.items()):
            raise ValueError("False H26 source-based score")
    scores={}
    for language in ("en","ar","all"):
        subset=rows if language=="all" else [x for x in rows if x["language"]==language]
        if len(subset)!=(42 if language=="all" else 21):
            raise ValueError("Missing H26 language subgroup")
        scores[language]={
            "cases":len(subset),
            "complete_codes":sum(x["format_valid"] is True for x in subset),
            "withheld":sum(x["withheld"] is True for x in subset),
            "semantic_correct":sum(x["semantic_correct"] is True for x in subset),
            "inference_errors":sum(x.get("error")=="model_inference_error" for x in subset),
            "unsafe_user_misrouted":sum(x["state"]=="unsafe_user" and
                                        x["suggestion"] is not None and
                                        x["suggestion"]["tool"]!="abstain" for x in subset),
            "by_state":{
                state:{"cases":sum(x["state"]==state for x in subset),
                       "correct":sum(x["state"]==state and x["semantic_correct"] is True for x in subset)}
                for state in corpus.STATE_TO_CODE
            },
        }
    return scores


def run(source_sha, output):
    rows, folder=remote_admit(source_sha,output)
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!="Tesla T4":
        raise RuntimeError("Only qualified free Tesla T4 for H26")
    torch.manual_seed(42)
    receipt={
        "schema":SCHEMA,"status":"loading","protocol":RUN_PROTOCOL,
        "source_commit":source_sha,"data_sha256":EXACT_DATA_SHA,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "gpu_name":torch.cuda.get_device_name(0),"provider":"kaggle",
        "precision":"fp16","partition":"dev","training_runs":0,
        "native_device_actions_executed":False,"model_adopted":False,
        "outcomes":[],"summary":None,
    }
    output_file=folder/"result.json"
    baseline.save_atomic(output_file,receipt)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,dtype=torch.float16,
        device_map="cuda",trust_remote_code=False)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False)
    model.eval()
    receipt["load_seconds"]=round(time.perf_counter()-started,6)
    receipt["torch_allocated_after_load_bytes"]=int(torch.cuda.memory_allocated())
    receipt["status"]="running"
    baseline.save_atomic(output_file,receipt)
    for row in rows:
        ref=row["source_ref"]
        entry={
            "id":ref["id"],"language":ref["language"],
            "family":ref["family"],"state":ref["state"],
            "raw":None,"error":None,
        }
        receipt["outcomes"].append(entry)
        baseline.save_atomic(output_file,receipt)
        start=time.perf_counter()
        try:
            messages=[
                {"role":"system","content":[{"type":"text","text":choice.INSTRUCTION}]},
                {"role":"user","content":[{"type":"text","text":json.dumps(row["model_input"],sort_keys=True,ensure_ascii=False)}]},
            ]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            tensors=processor(text=[prompt],return_tensors="pt")
            tensors={k:v.to("cuda") for k,v in tensors.items()}
            prompt_ids=tensors["input_ids"][0].tolist()
            gate=choice.TokenGate.build(processor.tokenizer,prompt_ids)
            with torch.inference_mode():
                generated=model.generate(
                    **tensors,do_sample=False,max_new_tokens=gate.max_new_tokens,
                    eos_token_id=gate.eos_id,pad_token_id=gate.eos_id,
                    prefix_allowed_tokens_fn=gate.allowed)
            suffix=generated[:,tensors["input_ids"].shape[-1]:]
            raw=processor.batch_decode(suffix,skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
            entry["raw"]=raw
            entry.update(choice.score(raw,row["private_reference"]["typed"]))
        except Exception as ex:
            entry.update(format_valid=False,withheld=False,semantic_correct=False,
                         suggestion=None,error="model_inference_error",
                         error_type=type(ex).__name__)
        entry["wall_seconds"]=round(time.perf_counter()-start,6)
        baseline.save_atomic(output_file,receipt)
    receipt["summary"]=aggregate(receipt["outcomes"])
    receipt["torch_peak_allocated_bytes"]=int(torch.cuda.max_memory_allocated())
    receipt["status"]="measured_unqualified"
    baseline.save_atomic(output_file,receipt)
    print(json.dumps({"status":receipt["status"],"summary":receipt["summary"]},
                     sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-commit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    run(args.source_commit,args.output)


if __name__=="__main__":
    main()
