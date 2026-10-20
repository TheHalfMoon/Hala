"""H37: private zero-cost Kaggle FP16 baseline on SHA-frozen local-browser raster.

Only screenshot + user goal are model inputs. The evaluator-only oracle is never
loaded by this worker. No native actions, side effects, or training.
"""
from __future__ import annotations

import argparse
import csv
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

from h34.runner import (MODEL, REVISION, PINS, PROMPT, _load_model,
                        configure_environment, owned_process, save_json)
from h37.freeze import checked_png

PROTOCOL = "hala.h37.browser_grounding_baseline.v1"
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
SOURCE_BRANCH = "ps/h37-browser-rendered-dev"
OUTPUT_SLUG = "hala-h37-realbrowser-visual-20261020"


def admit() -> None:
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("H37 model inference requires Kaggle Linux")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "Batch":
        raise RuntimeError("H37 requires retained Kaggle Batch mode")
    if os.environ.get("HALA_H37_RESEARCH_ALLOWED") != "1":
        raise RuntimeError("explicit noncommercial H37 GPU research opt-in required")


def source_identity(source_commit: str) -> dict:
    if not re.fullmatch("[a-f0-9]{40}", source_commit):
        raise ValueError("exact H37 source SHA required")
    root = Path(__file__).resolve().parent.parent
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root, timeout=40)
    if (git("rev-parse","HEAD").decode().strip() != source_commit or
            git("status","--porcelain","--untracked-files=all").strip()):
        raise ValueError("H37 model source must be clean exact commit")
    from tools.build_h37_kaggle import FILES
    hashes={}
    for name in FILES:
        payload=(root/name).read_bytes()
        if payload!=git("show",f"{source_commit}:{name}"):
            raise ValueError("H37 frozen source blob changed: "+name)
        hashes[name]=hashlib.sha256(payload).hexdigest()
    return hashes


def inputs() -> tuple[list[dict],dict,Path]:
    root=Path(__file__).resolve().parent/"assets"
    manifest=json.loads((root/"manifest.json").read_text("utf-8"))
    payload=(root/"model_inputs.json").read_bytes()
    if (manifest.get("schema")!="hala.h37.browser_rendered_dev_test.v1" or
            manifest.get("count")!=64 or manifest.get("dev")!=32 or
            manifest.get("test")!=32 or manifest.get("model_evaluated") is not False or
            hashlib.sha256(payload).hexdigest()!=manifest["model_inputs_sha256"]):
        raise ValueError("H37 frozen browser input/manifest integrity invalid")
    rows=json.loads(payload)
    if type(rows) is not list or len(rows)!=64:
        raise ValueError("H37 complete source image input rows required")
    known_images={r["id"]:r for r in manifest["images"]}
    for row in rows:
        if (set(row)!={"id","split","language","image","goal","width","height"}
                or row["id"] not in known_images or row["image"] != row["id"]+".png"
                or row["split"] not in ("dev","test") or row["language"] not in ("en","ar")
                or (row["width"],row["height"])!=(960,540)):
            raise ValueError("H37 invalid model-only row")
        png=(root/"screenshots"/row["image"]).read_bytes()
        checked_png(png)
        if hashlib.sha256(png).hexdigest()!=known_images[row["id"]]["image_sha256"]:
            raise ValueError("H37 screenshot did not match frozen source")
    return rows,manifest,root


def _worker(output:Path,source_commit:str) -> None:
    configure_environment()
    meta_path=output/"run_metadata.json"
    meta=json.loads(meta_path.read_text("utf-8"))
    raw_path=output/"raw_attempts.jsonl"
    try:
        meta["source_files"]=source_identity(source_commit)
        rows,manifest,assets=inputs()
        meta["source_browser"]=manifest["browser_version_observed"]
        meta["input_sha256"]=manifest["model_inputs_sha256"]
        meta["evaluator_sha256_not_read"]=manifest["evaluator_sha256"]
        meta["packages"]={}
        for name in PINS:
            try:
                meta["packages"][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                meta["packages"][name]=None
        if meta["packages"]!=PINS:
            raise RuntimeError("H37 requires exactly historical H34 runtime package pins")
        try:
            gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,memory.total",
                                         "--format=csv,noheader,nounits"],text=True,timeout=10)
            meta["gpu_observations"]=[list(map(str.strip,line)) for line in csv.reader(gpu.splitlines())]
        except (OSError,subprocess.SubprocessError) as exc:
            meta["gpu_observation_failure"]=type(exc).__name__
        meta["state"]="loading_model"
        save_json(meta_path,meta)
        processor,model=_load_model()
        import torch
        from PIL import Image
        meta["state"]="inference"
        meta["gpus"]=[{"index":i,"name":torch.cuda.get_device_name(i),
                       "vram_bytes":torch.cuda.get_device_properties(i).total_memory}
                      for i in range(torch.cuda.device_count())]
        save_json(meta_path,meta)
        with raw_path.open("a",encoding="utf-8",newline="\n") as journal:
            for index,row in enumerate(rows):
                png=(assets/"screenshots"/row["image"]).read_bytes()
                digest=hashlib.sha256(png).hexdigest()
                record={"case_id":row["id"],"split":row["split"],
                        "language":row["language"],"image_sha256":digest,
                        "prompt_sha256":PROMPT_SHA,
                        "raw":None,"error":"inference_interrupted"}
                save_json(output/"pending_attempt.json",record)
                try:
                    checked_png(png)
                    with Image.open(io.BytesIO(png)) as image_file:
                        image=image_file.convert("RGB")
                    content=[{"type":"image","image":image},
                             {"type":"text","text":row["goal"]+"\n"+PROMPT}]
                    text=processor.apply_chat_template(
                        [{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
                    features=processor(text=[text],images=[image],return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        tokens=model.generate(**features,do_sample=False,num_beams=1,max_new_tokens=96)
                    raw=processor.batch_decode(tokens[:,features.input_ids.shape[1]:],
                                               skip_special_tokens=True)[0]
                    if type(raw) is not str or len(raw.encode("utf-8"))>2048:
                        raise ValueError("model output exceeds H37 limits")
                    record.update(raw=raw,error=None)
                except Exception as exc:
                    record.update(raw=None,error=type(exc).__name__)
                journal.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
                journal.flush()
                os.fsync(journal.fileno())
                (output/"pending_attempt.json").unlink()
                print("H37_BROWSER",index+1,row["id"],"retained",flush=True)
        meta["state"]="completed"
    except Exception as exc:
        meta.update(state="incomplete",failure_type=type(exc).__name__)
    save_json(meta_path,meta)


def run(output:Path,source_commit:str,timeout_seconds:int=1100)->dict:
    admit()
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=1100:
        raise ValueError("bounded H37 time limit required")
    configure_environment()
    output=Path(output).resolve()
    if not output.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Kaggle output path must remain private inside /kaggle/working")
    output.mkdir(parents=True,exist_ok=False)
    meta={"protocol":PROTOCOL,"source_commit":source_commit,"model":MODEL,"revision":REVISION,
          "split":"first-party-browser-rendered-dev-plus-disjoint-domain-test",
          "expected_trials":64,"prompt_sha256":PROMPT_SHA,"actions_executed":False,
          "model_qualified":False,"independent_blind_test":False,"state":"starting",
          "deadline_seconds":timeout_seconds,"resume":False}
    path=output/"run_metadata.json"
    save_json(path,meta)
    raw_path=output/"raw_attempts.jsonl"
    raw_path.touch(exist_ok=False)
    start=time.monotonic()
    exit_code,timed_out,supervisor_error=None,False,None
    try:
        exit_code,timed_out=owned_process(
            [sys.executable,"-u","-m","h37.runner","--worker",
             "--output",str(output),"--source-commit",source_commit],timeout_seconds)
    except (Exception,SystemExit,KeyboardInterrupt) as exc:
        supervisor_error=type(exc).__name__
    meta=json.loads(path.read_text("utf-8"))
    pending=output/"pending_attempt.json"
    if pending.exists():
        item=json.loads(pending.read_text("utf-8"))
        previous=raw_path.read_bytes().splitlines()
        if len(previous)<64:
            last=json.loads(previous[-1]) if previous else None
            if last is None or last["case_id"]!=item["case_id"]:
                with raw_path.open("ab") as journal:
                    journal.write((json.dumps(item,ensure_ascii=False,sort_keys=True)+"\n").encode())
                    journal.flush()
                    os.fsync(journal.fileno())
    attempted=len(raw_path.read_bytes().splitlines())
    if timed_out or exit_code!=0 or meta.get("state")!="completed" or attempted!=64:
        meta["state"]="incomplete"
    meta.update(worker_exit_code=exit_code,supervisor_error=supervisor_error,
                deadline_exceeded=timed_out,attempted_count=attempted,
                elapsed_seconds=time.monotonic()-start,
                raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    save_json(path,meta)
    print("H37_EXTERNAL_RAW_SHA256",meta["raw_sha256"],flush=True)
    print("H37_INDEPENDENT_REPLAY_REQUIRED; MODEL_NOT_QUALIFIED",flush=True)
    return meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:_worker(Path(args.output),args.source_commit)
    else:run(Path(args.output),args.source_commit)


if __name__=="__main__":
    main()
