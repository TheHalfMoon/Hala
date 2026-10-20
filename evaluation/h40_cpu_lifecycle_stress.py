"""Bounded reproducible CPU-child lifecycle stress receipt (not CUDA evidence).

Run only a purpose-built fake subprocess owned by WorkerEngine. Each injected
failure must be observed exactly; no automatic retries or ignored exceptions.
Create-only JSONL receipts, with a final summary produced on completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

from PIL import Image
from hala.inference import RequestContext, ServiceError
from hala.serving.lifecycle import WorkerEngine
from tests.test_serving_lifecycle import fake_child

CASES=(("ok",None),("wait","deadline_exceeded"),("crash","internal_error"),
       ("oversize","invalid_backend_output"),("resource","busy"))


def run(output:Path, *, rounds:int=8, call_timeout:float=2.0) -> dict:
    output=Path(output)
    if type(rounds) is not int or not 1<=rounds<=24:
        raise ValueError("Stress rounds outside 1..24")
    if type(call_timeout) not in (float,int) or not 0.5<=call_timeout<=5.0:
        raise ValueError("Stress deadline outside bounded range")
    output.parent.mkdir(parents=True,exist_ok=True)
    journal=output.with_suffix(".jsonl")
    summary=output.with_suffix(".json")
    if journal.exists() or summary.exists():
        raise FileExistsError("H40 preserves existing receipts; choose new output")
    fixture=Path("tests/lifecycle_child.py")
    fixture_sha=hashlib.sha256(fixture.read_bytes()).hexdigest()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True,timeout=12).strip()
    metadata={"schema":"hala.h40.cpu_child_stress.v1","python":platform.python_version(),
              "platform":platform.platform(),"rounds":rounds,"timeout_s":call_timeout,
              "source_head":head,"fake_child_sha256":fixture_sha,
              "actual_model":False,"cuda":False,"native_actions":False,
              "retries":False,"model_qualified":False}
    records=[]
    started=time.monotonic()
    try:
        with journal.open("x",encoding="utf-8",newline="\n") as handle:
            for iteration in range(rounds):
                engine=WorkerEngine()
                try:
                    for index,(goal,expected) in enumerate(CASES):
                        t0=time.monotonic()
                        with patch.object(engine,"_spawn",side_effect=fake_child):
                            engine.start(timeout=5)
                        spawn_ms=round((time.monotonic()-t0)*1000,3)
                        got=None
                        error_kind=None
                        begin=time.monotonic()
                        try:
                            answer=engine(Image.new("RGB",(16,16)),
                                {"goal":goal},RequestContext(time.monotonic()+(
                                    0.1 if goal=="wait" else call_timeout)))
                            if goal=="ok" and answer!='{"point":[500,500]}':
                                error_kind="WrongSuccessfulOutput"
                        except ServiceError as e:
                            got=e.code
                        except Exception as e:
                            error_kind=type(e).__name__
                        duration=round((time.monotonic()-begin)*1000,3)
                        child_reaped=(engine.process is None)
                        record={
                            "round":iteration,"step":index,"goal":goal,
                            "expected":expected,"observed":got,"error_type":error_kind,
                            "spawn_ms":spawn_ms,"response_ms":duration,
                            "reaped":child_reaped,
                            "ok":got==expected and error_kind is None and
                                 child_reaped==(goal!="ok"),
                            "last_fault_code":(engine.last_fault or {}).get("code"),
                        }
                        handle.write(json.dumps(record,sort_keys=True)+"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(record)
                        if not record["ok"]:
                            raise RuntimeError("H40 first mismatch recorded; abort without retry")
                        if goal=="ok":
                            engine.close()
                        if engine.process is not None:
                            raise RuntimeError("H40 owned fake process not reaped after response")
                finally:
                    if engine.process is not None:
                        engine.close()
        metadata.update(completed=True,attempts=len(records),passed=sum(x["ok"] for x in records),
                        elapsed_ms=round((time.monotonic()-started)*1000,3),
                        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    except Exception:
        metadata.update(completed=False,attempts=len(records),
                        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest()
                         if journal.exists() else None,
                        elapsed_ms=round((time.monotonic()-started)*1000,3))
        summary.write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        raise
    summary.write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return metadata


def verify_receipts(output:Path) -> dict:
    """Independently reject incomplete, mutated, reordered or inflated trials."""
    output=Path(output)
    metadata=json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    data=output.with_suffix(".jsonl").read_bytes()
    if (metadata.get("schema")!="hala.h40.cpu_child_stress.v1" or
            metadata.get("completed") is not True or
            metadata.get("actual_model") is not False or
            metadata.get("cuda") is not False or
            metadata.get("native_actions") is not False or
            metadata.get("retries") is not False or
            type(metadata.get("rounds")) is not int or
            not 1<=metadata["rounds"]<=24 or
            metadata.get("attempts")!=metadata["rounds"]*len(CASES) or
            metadata.get("passed")!=metadata["attempts"] or
            metadata.get("journal_sha256")!=hashlib.sha256(data).hexdigest() or
            not data.endswith(b"\n")):
        raise ValueError("H40 envelope is not a complete authentic CPU stress trial")
    rows=[]
    def unique_pairs(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError("Duplicate H40 JSON key")
            value[key]=item
        return value
    for line in data.splitlines():
        rows.append(json.loads(line,object_pairs_hook=unique_pairs))
    if len(rows)!=metadata["attempts"]:
        raise ValueError("H40 raw rows missing")
    for index,record in enumerate(rows):
        exp_goal,exp_code=CASES[index % len(CASES)]
        if (set(record)!={"round","step","goal","expected","observed",
                           "error_type","spawn_ms","response_ms","reaped",
                           "ok","last_fault_code"} or
                record["round"]!=index//len(CASES) or
                record["step"]!=index%len(CASES) or
                record["goal"]!=exp_goal or
                record["expected"]!=exp_code or record["observed"]!=exp_code or
                record["error_type"] is not None or
                record["last_fault_code"] != exp_code or
                record["ok"] is not True or
                record["reaped"] is not (exp_goal!="ok") or
                type(record["spawn_ms"]) not in (int,float) or
                type(record["response_ms"]) not in (int,float) or
                not 0<=record["spawn_ms"]<=600000 or
                not 0<=record["response_ms"]<=600000):
            # The successful call retains its child until the explicit close;
            # failed calls must have reaped the exact owned child beforehand.
            raise ValueError("H40 expected ordered CPU child outcome differs")
    return metadata


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--rounds",type=int,default=8)
    ap.add_argument("--timeout-s",type=float,default=2)
    ns=ap.parse_args()
    print(json.dumps(run(ns.output,rounds=ns.rounds,call_timeout=ns.timeout_s),
                     sort_keys=True))
