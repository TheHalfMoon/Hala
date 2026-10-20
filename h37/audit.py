"""Independent hash-bound H37 real-Chromium authoring vs model-output replay.

All 64 attempts count unconditionally; disjoint authored TEST is NOT a blind
external website benchmark. Native actions always unqualified.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

from h34.audit import bounded_read,strict_json
from h34.protocol import score_attempt
from h37.freeze import checked_png
from h37.runner import MODEL,REVISION,PROMPT_SHA,PROTOCOL


def audit(output:Path,raw_sha256:str)->dict:
    if type(raw_sha256) is not str or not re.fullmatch("[a-f0-9]{64}",raw_sha256):
        raise ValueError("external SHA256 of real H37 raw journal required")
    output=Path(output)
    raw=bounded_read(output/"raw_attempts.jsonl",250_000)
    if hashlib.sha256(raw).hexdigest()!=raw_sha256 or raw and not raw.endswith(b"\n"):
        raise ValueError("H37 raw SHA or final line boundary incorrect")
    meta=strict_json(bounded_read(output/"run_metadata.json",65000))
    if (type(meta) is not dict or meta.get("protocol")!=PROTOCOL or
            meta.get("model")!=MODEL or meta.get("revision")!=REVISION or
            meta.get("prompt_sha256")!=PROMPT_SHA or
            not re.fullmatch("[a-f0-9]{40}",str(meta.get("source_commit",""))) or
            meta.get("actions_executed") is not False):
        raise ValueError("H37 metadata identity/source incorrect")
    root=Path(__file__).resolve().parent/"assets"
    manifest=json.loads((root/"manifest.json").read_text("utf-8"))
    modelbytes=(root/"model_inputs.json").read_bytes()
    evalbytes=(root/"evaluator_only.json").read_bytes()
    if (hashlib.sha256(modelbytes).hexdigest()!=manifest["model_inputs_sha256"] or
            hashlib.sha256(evalbytes).hexdigest()!=manifest["evaluator_sha256"] or
            hashlib.sha256((root/"capture.json").read_bytes()).hexdigest()!=manifest["capture_sha256"]):
        raise ValueError("H37 frozen source/browser screenshot sidecars tampered")
    tasks=json.loads(modelbytes)
    labels=json.loads(evalbytes)
    image_by_id={x["id"]:x for x in manifest["images"]}
    if len(tasks)!=64 or len(labels)!=64 or len(image_by_id)!=64:
        raise ValueError("H37 incomplete source data")
    for task,case in zip(tasks,labels):
        if task["id"]!=case["id"] or task["goal"]!=case["goal"]:
            raise ValueError("model and evaluator rows not bound")
        image=(root/"screenshots"/task["image"]).read_bytes()
        checked_png(image)
        if (hashlib.sha256(image).hexdigest()!=case["image_sha256"] or
                image_by_id[task["id"]]["image_sha256"]!=case["image_sha256"]):
            raise ValueError("H37 original browser screenshot digest mismatch")
    lines=[strict_json(line) for line in raw.decode("utf-8").splitlines()]
    if len(lines)>64:
        raise ValueError("unexpected additional H37 model records")
    totals=Counter()
    split={s:Counter() for s in ("dev","test")}
    language={s:Counter() for s in ("en","ar")}
    domains=defaultdict(Counter)
    layouts=defaultdict(Counter)
    outcomes=[]
    for idx,case in enumerate(labels):
        attempt=lines[idx] if idx<len(lines) else None
        if attempt is None:
            outcome={"error":True,"parse_valid":False,"target_hit":False,
                     "gate_valid_unique":False,"wrong_target":False,"abstain":False}
            error_type="not_attempted"
        else:
            task=tasks[idx]
            expected={"case_id":task["id"],"split":task["split"],
                      "language":task["language"],
                      "image_sha256":case["image_sha256"],
                      "prompt_sha256":PROMPT_SHA}
            if (type(attempt) is not dict or
                    set(attempt)!={"case_id","split","language","image_sha256",
                                  "prompt_sha256","raw","error"} or
                    any(type(attempt[k]) is not str or attempt[k]!=v for k,v in expected.items())):
                raise ValueError("H37 model row order, image or prompt binding differs")
            raw_value,error=attempt["raw"],attempt["error"]
            if error is None:
                if type(raw_value) is not str or len(raw_value.encode("utf-8"))>2048:
                    raise ValueError("oversized/nonstring raw answer")
            elif (type(error) is not str or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}",error)
                  or raw_value is not None):
                raise ValueError("H37 invalid runtime failure code")
            outcome=score_attempt(case,raw_value,error)
            error_type=error
        scored={k:int(bool(outcome.get(k))) for k in
                ("parse_valid","target_hit","gate_valid_unique","wrong_target","abstain","error")}
        totals.update(scored)
        split[case["split"]].update(scored)
        language[case["language"]].update(scored)
        domains[case["family"]].update(scored)
        layouts[case["layout"]].update(scored)
        outcomes.append({"id":case["id"],"split":case["split"],"lang":case["language"],
                         "family":case["family"],"layout":case["layout"],
                         "error_type":error_type,"scored":scored})
    return {"schema":"hala.h37.browser_grounding_model_replay.v1",
            "source_commit_observed":meta["source_commit"],
            "raw_sha256":raw_sha256,"complete":len(lines)==64,
            "attempts":len(lines),"unconditional_denominator":64,
            "totals":dict(totals),
            "split":{s:{"denominator":32,**dict(c)} for s,c in split.items()},
            "languages":{s:{"denominator":32,**dict(c)} for s,c in language.items()},
            "domains":{s:{"denominator":16,**dict(c)} for s,c in domains.items()},
            "layouts":{s:{"denominator":16,**dict(c)} for s,c in layouts.items()},
            "rows":outcomes,"browser_rendered":True,
            "external_app_captured":False,"blind_test_qualified":False,
            "model_qualified":False,"native_actions_executed":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--expected-raw-sha256",required=True)
    args=parser.parse_args()
    print(json.dumps(audit(Path(args.output),args.expected_raw_sha256),
                     ensure_ascii=False,sort_keys=True,indent=2))


if __name__=="__main__":main()
