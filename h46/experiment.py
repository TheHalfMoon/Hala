"""H46: actual Qwen3-VL 2B, first-frame pixel-only 8-case public DEV pilot.

Noncommercial free Kaggle only. Inference never reads evaluator_only.json.
Not OSWorld, not an independent blind holdout and not a competitor win.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from collections import Counter

from h34.runner import MODEL,REVISION,PINS,PROMPT,_load_model,configure_environment,owned_process,save_json
from h34.protocol import score_attempt
from h34.audit import bounded_read,strict_json
from h37.freeze import checked_png

PROTOCOL="hala.h46.qwen3vl_2b_firstframe_public_dev.v1"
SOURCE_BRANCH="ps/h46-qwen-firstframe-dev"
PROMPT_SHA=hashlib.sha256(PROMPT.encode()).hexdigest()
ROOT=Path(__file__).resolve().parent.parent
ASSETS=ROOT/"h44"/"assets"
INPUT_SHA="f432747051c5fc2b6741db1a8c9b4b58f04d0dd7248c77546baa191bcdc29193"
ORACLE_SHA="974b6ce6a51d272437e67471c10ee07f03b9303427bac9ec66e0391fe254ee22"
CAPTURE_SHA="ba87f75f412ba235903ec72e119fc8fb021455a646f8a0c9becb5adc455cd14c"
TASK_COUNT=8

def sha(v:bytes)->str:return hashlib.sha256(v).hexdigest()

def cases()->list[dict]:
    """Read ONLY goal+image manifest and original real-browser PNG bytes."""
    path=ASSETS/"model_inputs.json"
    if sha(path.read_bytes())!=INPUT_SHA:raise ValueError("Frozen model-only manifest changed")
    envelope=strict_json(bounded_read(path,50000))
    if (envelope.get("schema")!="hala.h44.model-only-projection.v1"
        or envelope.get("model_qualified") is not False
        or envelope.get("independent_blind_test") is not False
        or type(envelope.get("cases")) is not list or len(envelope["cases"])!=24):
        raise ValueError("Not original H44 model-only source")
    rows=[x for x in envelope["cases"] if x["split"]=="dev"]
    if (len(rows)!=TASK_COUNT or {r["family"] for r in rows}!={"harbor_inventory","summit_scheduling"}
        or sum(r["language"]=="en" for r in rows)!=4 or sum(r["language"]=="ar" for r in rows)!=4):
        raise ValueError("Frozen 8-case Arabic/English DEV not reproduced")
    for row in rows:
        if (set(row)!={"id","language","split","family","goal","screenshot",
                       "image_sha256","width","height","model_trained"} or
            row["model_trained"] is not False or
            (row["width"],row["height"])!=(960,540) or
            row["screenshot"]!="screenshots/"+row["id"]+".png" or
            re.fullmatch(r"h43-(harbor_inventory|summit_scheduling)-0[0-3]-(ar|en)",row["id"]) is None):
            raise ValueError("Model input row includes privileged answers or changed fields")
        png=(ASSETS/row["screenshot"]).read_bytes()
        checked_png(png)
        if sha(png)!=row["image_sha256"]:raise ValueError("Input PNG changed")
    return rows

def source_gate(commit:str)->dict:
    if re.fullmatch("[0-9a-f]{40}",commit) is None:raise ValueError("Need exact source SHA")
    def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,timeout=35)
    if (git("rev-parse","HEAD").decode().strip()!=commit or
        git("status","--porcelain","--untracked-files=all").strip()):
        raise ValueError("Only run on clean source at exact immutable commit")
    paths=["h46/__init__.py","h46/experiment.py","tools/build_h46_kaggle.py",
           "h34/runner.py","h34/protocol.py","h44/assets/model_inputs.json"]
    paths+=["h44/assets/"+r["screenshot"] for r in cases()]
    for f in paths:
        if (ROOT/f).read_bytes()!=git("show",f"{commit}:{f}"):
            raise ValueError("Source blob changed: "+f)
    return {f:sha((ROOT/f).read_bytes()) for f in paths}

def admit():
    if (platform.system()!="Linux" or not Path("/kaggle/working").is_dir()
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")!="Batch"
        or os.environ.get("HALA_H46_NONCOMMERCIAL_FREE_RESEARCH")!="1"):
        raise RuntimeError("H46 requires explicitly approved free Kaggle Batch research")

def worker(out:Path,commit:str):
    configure_environment()
    mp=out/"run_metadata.json"
    m=json.loads(mp.read_text("utf-8"))
    try:
        m["source_inventory"]=source_gate(commit)
        rows=cases()
        versions={n:importlib.metadata.version(n) for n in PINS}
        m["runtime_versions"]=versions
        if versions!=PINS:raise RuntimeError("H37 approved exact inference package versions absent")
        import torch
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Actual free Kaggle T4 required, no paid fallback")
        m["gpu"]={"name":torch.cuda.get_device_name(0),
                  "vram":torch.cuda.get_device_properties(0).total_memory}
        m["state"]="loading_model";save_json(mp,m)
        processor,model=_load_model()
        from PIL import Image
        m["state"]="inference";save_json(mp,m)
        with (out/"raw_attempts.jsonl").open("ab") as log:
            for n,row in enumerate(rows):
                result={"case_id":row["id"],"language":row["language"],"split":"dev",
                        "image_sha256":row["image_sha256"],"prompt_sha256":PROMPT_SHA,
                        "raw":None,"error":"interrupted"}
                save_json(out/"pending_attempt.json",result)
                try:
                    img_bytes=(ASSETS/row["screenshot"]).read_bytes()
                    checked_png(img_bytes)
                    if sha(img_bytes)!=row["image_sha256"]:raise ValueError("Input changed after manifest")
                    with Image.open(io.BytesIO(img_bytes)) as source:
                        image=source.convert("RGB")
                    content=[{"type":"image","image":image},
                             {"type":"text","text":row["goal"]+"\n"+PROMPT}]
                    text=processor.apply_chat_template(
                        [{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
                    tensors=processor(text=[text],images=[image],return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        generated=model.generate(**tensors,do_sample=False,num_beams=1,max_new_tokens=96)
                    answer=processor.batch_decode(generated[:,tensors.input_ids.shape[1]:],
                                                   skip_special_tokens=True)[0]
                    if not isinstance(answer,str) or len(answer.encode())>2048:
                        raise ValueError("Unbounded model response")
                    result.update(raw=answer,error=None)
                except Exception as exc:
                    result.update(raw=None,error=type(exc).__name__)
                log.write((json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n").encode())
                log.flush();os.fsync(log.fileno())
                (out/"pending_attempt.json").unlink()
                print("H46_RECORDED_ATTEMPT",n+1,row["id"],flush=True)
        m["state"]="completed"
    except Exception as exc:
        m.update(state="incomplete",failure_type=type(exc).__name__)
    save_json(mp,m)

def run(out:Path,commit:str,timeout:int=850)->dict:
    admit();configure_environment()
    if type(timeout) is not int or not 60<=timeout<=850:
        raise ValueError("Bounded H46 timeout")
    out=Path(out).resolve()
    if not out.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Private Kaggle output only")
    out.mkdir(parents=True,exist_ok=False)
    m={"protocol":PROTOCOL,"model":MODEL,"revision":REVISION,"source_commit":commit,
       "cases":TASK_COUNT,"prompt_sha256":PROMPT_SHA,"input_sha256":INPUT_SHA,
       "oracle_sha256_not_read":ORACLE_SHA,"capture_sha256":CAPTURE_SHA,
       "native_actions":False,"model_qualified":False,"external_benchmark":False,
       "independent_blind_test":False,"state":"starting","retries":0}
    save_json(out/"run_metadata.json",m)
    (out/"raw_attempts.jsonl").touch(exist_ok=False)
    t0=time.monotonic()
    rc,timed_out,error=None,False,None
    try:
        rc,timed_out=owned_process([sys.executable,"-u","-m","h46.experiment",
            "--worker","--output",str(out),"--source-commit",commit],timeout,cwd=ROOT)
    except Exception as exc:error=type(exc).__name__
    pending=out/"pending_attempt.json"
    if pending.exists():
        last=json.loads(pending.read_text("utf-8"))
        lines=(out/"raw_attempts.jsonl").read_bytes().splitlines()
        if len(lines)<TASK_COUNT and (not lines or json.loads(lines[-1])["case_id"]!=last["case_id"]):
            with (out/"raw_attempts.jsonl").open("ab") as f:
                f.write((json.dumps(last,sort_keys=True)+"\n").encode());f.flush();os.fsync(f.fileno())
    raw=(out/"raw_attempts.jsonl").read_bytes()
    m=json.loads((out/"run_metadata.json").read_text("utf-8"))
    if rc!=0 or timed_out or m.get("state")!="completed" or len(raw.splitlines())!=TASK_COUNT:
        m["state"]="incomplete"
    m.update(exit_code=rc,timeout=timed_out,supervisor_error=error,
             attempted=len(raw.splitlines()),raw_sha256=sha(raw),
             elapsed_s=time.monotonic()-t0)
    save_json(out/"run_metadata.json",m)
    print("H46_RAW_SHA256",m["raw_sha256"],"STATE",m["state"],flush=True)
    return m

def replay(out:Path,raw_sha:str)->dict:
    """Independent evaluator, never invoked from Kaggle notebook runner."""
    if re.fullmatch("[0-9a-f]{64}",raw_sha) is None:raise ValueError("External raw digest required")
    raw=bounded_read(Path(out)/"raw_attempts.jsonl",75000)
    if sha(raw)!=raw_sha or raw and not raw.endswith(b"\n"):
        raise ValueError("Original model raw SHA changed")
    m=strict_json(bounded_read(Path(out)/"run_metadata.json",90000))
    if (m.get("protocol")!=PROTOCOL or m.get("model")!=MODEL or m.get("revision")!=REVISION
        or m.get("input_sha256")!=INPUT_SHA or m.get("oracle_sha256_not_read")!=ORACLE_SHA
        or m.get("prompt_sha256")!=PROMPT_SHA or m.get("capture_sha256")!=CAPTURE_SHA
        or m.get("native_actions") is not False or m.get("model_qualified") is not False):
        raise ValueError("Unqualified model metadata has changed")
    inputs=cases()
    if (sha((ASSETS/"evaluator_only.json").read_bytes())!=ORACLE_SHA
        or sha((ASSETS/"capture.json").read_bytes())!=CAPTURE_SHA):
        raise ValueError("Original source/browser evaluator changed")
    oracles=strict_json(bounded_read(ASSETS/"evaluator_only.json",100000))["cases"]
    labels=[x for x in oracles if x["split"]=="dev"]
    if [x["id"] for x in inputs]!=[x["id"] for x in labels]:
        raise ValueError("Model/evaluator identifiers differ")
    attempts=[strict_json(x) for x in raw.splitlines()]
    if len(attempts)>TASK_COUNT:raise ValueError("Extra attempts")
    totals=Counter()
    language={"en":Counter(),"ar":Counter()}
    records=[]
    for i,(inp,label) in enumerate(zip(inputs,labels)):
        sample=attempts[i] if i<len(attempts) else None
        if sample is None:answer,error=None,"missing_attempt"
        else:
            binding={"case_id":inp["id"],"language":inp["language"],
                     "split":"dev","image_sha256":inp["image_sha256"],
                     "prompt_sha256":PROMPT_SHA}
            if (type(sample) is not dict or
                set(sample)!={"case_id","language","split","image_sha256","prompt_sha256","raw","error"} or
                any(sample.get(k)!=v for k,v in binding.items())):
                raise ValueError("Unbound, reordered or duplicated model attempt")
            answer,error=sample["raw"],sample["error"]
            if error is not None and (type(error) is not str or
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}",error) is None or answer is not None):
                raise ValueError("Invalid model error receipt")
        observed=[{"bounds":c["bounds"],"label":c["ref"]} for c in label["controls"]]
        case={"width":960,"height":540,"controls":observed,"expected_label":label["expected_ref"]}
        outcome=score_attempt(case,answer,error)
        status={k:int(bool(outcome.get(k))) for k in
                ("parse_valid","target_hit","wrong_target","abstain","error","gate_valid_unique")}
        totals.update(status);language[inp["language"]].update(status)
        records.append({"id":inp["id"],"language":inp["language"],"status":status,"error":error})
    return {"schema":"hala.h46.public_dev_external_raw_replay.v1",
            "raw_sha256":raw_sha,"denominator":TASK_COUNT,"attempted":len(attempts),
            "complete":len(attempts)==TASK_COUNT,"totals":dict(totals),
            "languages":{v:dict(c) for v,c in language.items()},
            "per_language_denominator":{"en":4,"ar":4},"rows":records,
            "model_qualified":False,"independent_blind_test":False,
            "external_benchmark":False,"competitor_win":False,"native_actions":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args()
    if a.worker:worker(a.output,a.source_commit)
    else:run(a.output,a.source_commit)
