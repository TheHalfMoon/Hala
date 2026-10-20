"""H47 source-locked, real paired Qwen3-VL 2B prompt experiment on H46 public DEV.

No training, model result until genuine private Kaggle Batch and offline replay.
Compare H36's original numeric-example prompt vs its no-example version.
"""
from __future__ import annotations
import argparse
from collections import Counter
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

from h34.runner import MODEL,REVISION,PINS,_load_model,configure_environment,owned_process,save_json
from h34.protocol import score_attempt
from h34.audit import bounded_read,strict_json
from h36.protocol import NUMERIC_EXAMPLE, NO_NUMERIC_EXAMPLE
from h37.freeze import checked_png
from h46.experiment import cases,INPUT_SHA,ORACLE_SHA,CAPTURE_SHA,ASSETS,sha

PROTOCOL="hala.h47.same_gpu_paired_prompt_8_public_dev.v1"
SOURCE_BRANCH="ps/h47-paired-actual-dev-8"
ARMS=("numeric_example","no_numeric_example")
PROMPTS={"numeric_example":NUMERIC_EXAMPLE,"no_numeric_example":NO_NUMERIC_EXAMPLE}
PROMPT_SHA={name:sha(prompt.encode("utf-8")) for name,prompt in PROMPTS.items()}
N=8
ROOT=Path(__file__).resolve().parent.parent
if (NUMERIC_EXAMPLE==NO_NUMERIC_EXAMPLE or
    '[0.25,0.75]' not in NUMERIC_EXAMPLE or
    '[0.25,0.75]' in NO_NUMERIC_EXAMPLE):
    raise RuntimeError("H47 prompt treatment not actually isolated")

def plan()->list[dict]:
    data=cases()
    if len(data)!=N:raise ValueError("Fixed H46 DEV task count changed")
    out=[]
    for i,row in enumerate(data):
        order=ARMS if i%2==0 else ARMS[::-1]
        for arm in order:
            out.append({"case_id":row["id"],"language":row["language"],
                        "image_sha256":row["image_sha256"],"arm":arm,
                        "prompt_sha256":PROMPT_SHA[arm]})
    if len(out)!=16 or set(x["arm"] for x in out)!=set(ARMS):
        raise ValueError("H47 pre-registered arm allocation differs")
    return out

def plan_sha()->str:
    return sha((json.dumps(plan(),sort_keys=True,separators=(",",":"))+"\n").encode())

def source_inventory(commit:str)->dict:
    if re.fullmatch("[0-9a-f]{40}",commit) is None:raise ValueError("Exact signed source SHA needed")
    def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,timeout=40)
    if (git("rev-parse","HEAD").decode().strip()!=commit or
        git("status","--porcelain","--untracked-files=all").strip()):
        raise ValueError("H47 source checkout must be exact clean signed commit")
    paths=["h47/__init__.py","h47/experiment.py","tools/build_h47_kaggle.py",
           "h46/experiment.py","h36/protocol.py","h34/runner.py",
           "h34/protocol.py","h44/assets/model_inputs.json"]
    paths+=["h44/assets/"+x["screenshot"] for x in cases()]
    for f in paths:
        if (ROOT/f).read_bytes()!=git("show",f"{commit}:{f}"):
            raise ValueError("H47 source blob tampering: "+f)
    return {f:sha((ROOT/f).read_bytes()) for f in paths}

def admit()->None:
    if (platform.system()!="Linux" or not Path("/kaggle/working").is_dir()
        or os.getenv("KAGGLE_KERNEL_RUN_TYPE")!="Batch"
        or os.getenv("HALA_H47_NONCOMMERCIAL_FREE_RESEARCH")!="1"):
        raise RuntimeError("Require explicit private free Kaggle Batch research opt-in")

def worker(out:Path,commit:str)->None:
    configure_environment()
    meta_path=out/"run_metadata.json"
    meta=json.loads(meta_path.read_text("utf-8"))
    try:
        meta["source_inventory"]=source_inventory(commit)
        trials=plan()
        versions={n:importlib.metadata.version(n) for n in PINS}
        meta["runtime_versions"]=versions
        if versions!=PINS:raise RuntimeError("Original H37 pinned GPU runtime unavailable")
        import torch
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("No allocated free Kaggle T4; refuse substitution")
        meta["hardware"]={"gpu":torch.cuda.get_device_name(0),
                          "vram_bytes":torch.cuda.get_device_properties(0).total_memory}
        meta["state"]="loading_model";save_json(meta_path,meta)
        processor,model=_load_model()
        from PIL import Image
        by_id={r["id"]:r for r in cases()}
        meta["state"]="paired_inference";save_json(meta_path,meta)
        with (out/"raw_attempts.jsonl").open("ab") as logfile:
            for i,request in enumerate(trials):
                row=by_id[request["case_id"]]
                response={"case_id":request["case_id"],"language":row["language"],
                          "arm":request["arm"],"image_sha256":request["image_sha256"],
                          "prompt_sha256":request["prompt_sha256"],
                          "raw":None,"error":"interrupted"}
                save_json(out/"pending_attempt.json",response)
                try:
                    image_bytes=(ASSETS/row["screenshot"]).read_bytes()
                    checked_png(image_bytes)
                    if sha(image_bytes)!=request["image_sha256"]:
                        raise ValueError("Image bytes changed")
                    with Image.open(io.BytesIO(image_bytes)) as reader:
                        im=reader.convert("RGB")
                    content=[{"type":"image","image":im},
                             {"type":"text","text":row["goal"]+"\n"+PROMPTS[request["arm"]]}]
                    prompt=processor.apply_chat_template(
                        [{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
                    encoded=processor(text=[prompt],images=[im],return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        ids=model.generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=96)
                    raw=processor.batch_decode(ids[:,encoded.input_ids.shape[1]:],
                                               skip_special_tokens=True)[0]
                    if type(raw) is not str or len(raw.encode())>2048:
                        raise ValueError("Invalid generated answer")
                    response.update(raw=raw,error=None)
                except Exception as exc:
                    response.update(raw=None,error=type(exc).__name__)
                logfile.write((json.dumps(response,sort_keys=True,ensure_ascii=False)+"\n").encode())
                logfile.flush();os.fsync(logfile.fileno())
                (out/"pending_attempt.json").unlink()
                print("H47_RAW_ATTEMPT",i+1,"of",len(trials),request["case_id"],request["arm"],flush=True)
        meta["state"]="completed"
    except Exception as exc:
        meta.update(state="incomplete",failure_type=type(exc).__name__)
    save_json(meta_path,meta)

def run(out:Path,commit:str,timeout:int=1050)->dict:
    admit();configure_environment()
    if type(timeout) is not int or not 60<=timeout<=1050:
        raise ValueError("H47 owned process timeout must be bounded")
    out=Path(out).resolve()
    if not out.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Private Kaggle output directory required")
    out.mkdir(parents=True,exist_ok=False)
    meta={"schema":"hala.h47.metadata.v1","protocol":PROTOCOL,"source_commit":commit,
          "model":MODEL,"revision":REVISION,"runtime_pins":PINS,
          "prompt_hashes":PROMPT_SHA,"input_sha256":INPUT_SHA,
          "evaluator_sha256_not_read":ORACLE_SHA,"capture_sha256":CAPTURE_SHA,
          "plan_sha256":plan_sha(),"expected_attempts":16,
          "model_qualified":False,"blind_test":False,"benchmark":False,
          "native_actions":False,"retries":0,"state":"starting"}
    save_json(out/"run_metadata.json",meta)
    (out/"raw_attempts.jsonl").touch(exist_ok=False)
    t0=time.monotonic();exit_code,timed_out,worker_error=None,False,None
    try:
        exit_code,timed_out=owned_process(
            [sys.executable,"-u","-m","h47.experiment","--worker",
             "--output",str(out),"--source-commit",commit],
            timeout,cwd=ROOT)
    except Exception as exc:worker_error=type(exc).__name__
    pending=out/"pending_attempt.json"
    if pending.exists():
        last=json.loads(pending.read_text("utf-8"))
        records=(out/"raw_attempts.jsonl").read_bytes().splitlines()
        if len(records)<16 and (not records or
            (json.loads(records[-1])["case_id"],json.loads(records[-1])["arm"])!=(last["case_id"],last["arm"])):
            with (out/"raw_attempts.jsonl").open("ab") as dest:
                dest.write((json.dumps(last,sort_keys=True)+"\n").encode());dest.flush();os.fsync(dest.fileno())
    blob=(out/"raw_attempts.jsonl").read_bytes()
    meta=json.loads((out/"run_metadata.json").read_text("utf-8"))
    if timed_out or exit_code!=0 or meta.get("state")!="completed" or len(blob.splitlines())!=16:
        meta["state"]="incomplete"
    meta.update(exit_code=exit_code,timeout=timed_out,worker_error=worker_error,
                attempted=len(blob.splitlines()),raw_sha256=sha(blob),
                elapsed_seconds=time.monotonic()-t0)
    save_json(out/"run_metadata.json",meta)
    print("H47_RETAINED_UNQUALIFIED",meta["state"],meta["attempted"],meta["raw_sha256"],flush=True)
    return meta

def replay(out:Path,raw_sha:str)->dict:
    """External-only scoring, worker doesn't import/call this function."""
    if re.fullmatch("[0-9a-f]{64}",raw_sha) is None:raise ValueError("External raw SHA256 needed")
    out=Path(out)
    blob=bounded_read(out/"raw_attempts.jsonl",100000)
    if sha(blob)!=raw_sha or blob and not blob.endswith(b"\n"):
        raise ValueError("Original raw journal SHA or line boundary changed")
    meta=strict_json(bounded_read(out/"run_metadata.json",100000))
    if (meta.get("protocol")!=PROTOCOL or meta.get("plan_sha256")!=plan_sha()
        or meta.get("model")!=MODEL or meta.get("revision")!=REVISION
        or meta.get("prompt_hashes")!=PROMPT_SHA or meta.get("input_sha256")!=INPUT_SHA
        or meta.get("evaluator_sha256_not_read")!=ORACLE_SHA
        or meta.get("capture_sha256")!=CAPTURE_SHA
        or meta.get("model_qualified") is not False
        or meta.get("native_actions") is not False
        or meta.get("blind_test") is not False):
        raise ValueError("H47 metadata claim/source changed")
    if (sha((ASSETS/"evaluator_only.json").read_bytes())!=ORACLE_SHA or
        sha((ASSETS/"capture.json").read_bytes())!=CAPTURE_SHA):
        raise ValueError("Offline evaluator original H44 bytes changed")
    models=cases()
    labels=[r for r in strict_json(bounded_read(ASSETS/"evaluator_only.json",150000))["cases"] if r["split"]=="dev"]
    if [r["id"] for r in models]!=[r["id"] for r in labels]:
        raise ValueError("Source and evaluator case order mismatch")
    raw_rows=[strict_json(line) for line in blob.splitlines()]
    if len(raw_rows)>16:raise ValueError("Extra paired model attempts")
    outcomes={}
    actually_attempted=set()
    metrics={arm:Counter() for arm in ARMS}
    strata={arm:{"en":Counter(),"ar":Counter()} for arm in ARMS}
    attempts=[]
    by_id={r["id"]:(r,lab) for r,lab in zip(models,labels)}
    for i,req in enumerate(plan()):
        attempt=raw_rows[i] if i<len(raw_rows) else None
        if attempt is None:response,error=None,"missing_attempt"
        else:
            expect={**req}
            if (type(attempt) is not dict or set(attempt)!=set(expect)|{"raw","error"} or
                any(attempt.get(k)!=v for k,v in expect.items())):
                raise ValueError("Altered, reordered or wrong-arm result")
            response,error=attempt["raw"],attempt["error"]
            actually_attempted.add((req["case_id"],req["arm"]))
            if error is not None and (type(error) is not str or
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}",error) is None or response is not None):
                raise ValueError("Malformed model error")
        row,lab=by_id[req["case_id"]]
        observed=[{"bounds":ctrl["bounds"],"label":ctrl["ref"]} for ctrl in lab["controls"]]
        case={"width":960,"height":540,"controls":observed,"expected_label":lab["expected_ref"]}
        result=score_attempt(case,response,error)
        observed_flags={k:int(bool(result.get(k))) for k in
                        ("parse_valid","target_hit","wrong_target","gate_valid_unique","abstain","error")}
        metrics[req["arm"]].update(observed_flags)
        strata[req["arm"]][req["language"]].update(observed_flags)
        outcomes[(req["case_id"],req["arm"])]=observed_flags
        attempts.append({"case_id":req["case_id"],"arm":req["arm"],
                         "language":req["language"],"error":error,
                         "outcome":observed_flags})
    pair={"no_numeric_better":0,"numeric_better":0,"tied_both_correct":0,
          "tied_both_wrong":0,"incomplete_pairs":0}
    for row in models:
        k=row["id"];a=outcomes.get((k,"numeric_example"));b=outcomes.get((k,"no_numeric_example"))
        if ((k,"numeric_example") not in actually_attempted or
            (k,"no_numeric_example") not in actually_attempted):
            pair["incomplete_pairs"]+=1
        elif a["target_hit"] and b["target_hit"]:pair["tied_both_correct"]+=1
        elif a["target_hit"]:pair["numeric_better"]+=1
        elif b["target_hit"]:pair["no_numeric_better"]+=1
        else:pair["tied_both_wrong"]+=1
    return {"schema":"hala.h47.paired_dev_external_replay.v1",
        "denominator_per_arm":8,"total_expected_attempts":16,
        "attempted":len(raw_rows),"complete":len(raw_rows)==16,
        "arm_outcomes":{a:dict(c) for a,c in metrics.items()},
        "language_outcomes":{a:{lang:dict(count) for lang,count in v.items()}
                             for a,v in strata.items()},
        "pair_result":pair,"attempt_rows":attempts,
        "raw_sha256":raw_sha,"not_independent":True,
        "model_qualified":False,"benchmark":False,"competitor_win":False}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--worker",action="store_true")
    ap.add_argument("--source-commit",required=True)
    ap.add_argument("--output",type=Path,required=True)
    z=ap.parse_args()
    if z.worker:worker(z.output,z.source_commit)
    else:run(z.output,z.source_commit)
