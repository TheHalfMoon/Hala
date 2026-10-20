"""H45 exact replay of real Edge state transitions vs Python H43 final state.

No model, no evaluation leaderboard, no external app. An embedded app may
fabricate its own state; independent real-app oracles remain to be developed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from h37.freeze import checked_png
from h43.factory import author_cases,canonical,digest
from h43.engine import simulate,score_state
from h45.render import dynamic_page


def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def audit(root:Path, *, override:dict|None=None)->tuple[dict,dict,dict]:
    root=Path(root)
    source=author_cases()
    plan_bytes=(root/"scenes.json").read_bytes()
    plan=json.loads(plan_bytes)
    data=(json.loads((root/"capture.json").read_text("utf-8"))
          if override is None else override)
    cases=source["cases"]
    if (plan.get("schema")!="hala.h45.original-browser-workflows.v1"
        or plan.get("h43_manifest_sha256")!=digest(source)
        or len(plan.get("scenes",[]))!=24):
        raise ValueError("Original H45 source plan differs")
    if (data.get("schema")!="hala.h45.actual-edge-interactions.v1"
        or not data.get("browser_version","").startswith("Edg/155.")
        or data.get("puppeteer_core")!="25.12.0"
        or data.get("viewport")!={"width":960,"height":540,"device_scale_factor":1}
        or data.get("h45_scene_sha256")!=sha(plan_bytes)
        or data.get("original_h43_manifest_sha256")!=sha(
            Path("h43/data/pilot_manifest.json").read_bytes())
        or data.get("screenshot_count")!=144
        or data.get("actual_browser_ui_interactions") is not True
        or data.get("only_first_party_synthetic_pages") is not True
        or data.get("external_network_requests") is not False
        or data.get("actual_model_inference") is not False
        or data.get("external_benchmark") is not False
        or data.get("native_desktop_actions") is not False
        or len(data.get("records",[]))!=24):
        raise ValueError("Real source-bound Edge browser receipt differs")
    model_inputs=[]
    oracle=[]
    seen=set()
    for task,scene,entry in zip(cases,plan["scenes"],data["records"]):
        if not all(scene.get(k)==task[k] and entry.get(k)==task[k]
                   for k in ("id","family","split","language")):
            raise ValueError("Scene/task/browser identity drift")
        expected_html=dynamic_page(task).encode("utf-8")
        html=(root/"pages"/(task["id"]+".html")).read_bytes()
        if (html!=expected_html or sha(html)!=entry["page_sha256"]
            or sha(html)!=scene["original_html_sha256"]):
            raise ValueError("HTML differs from original first-party dynamic source")
        golden=task["golden_actions"]
        wrong_actions=[{"kind":"press","ref":"row_decoy","value":None},
                       golden[1],golden[2]]
        unrelated_actions=[*golden,{"kind":"press","ref":"unrelated_button","value":None}]
        shots=entry["states"]+entry["negatives"]
        if len(entry["states"])!=4 or len(entry["negatives"])!=2:
            raise ValueError("Missing required per-action screenshots")
        for index,snapshot in enumerate(shots):
            if (type(snapshot) is not dict or
                set(snapshot)!={"label","action","filename","image_sha256",
                                "image_bytes","selected","amount","confirmed",
                                "unrelated_changed","controls","display","lang","rtl"}):
                raise ValueError("Unrecognized snapshot schema")
            if index<4:
                actions=golden[:index]
                expected_label="step"+str(index)
                expected_action=golden[index-1] if index else None
            else:
                actions=wrong_actions if index==4 else unrelated_actions
                expected_label="negative_wrong_target" if index==4 else "negative_unrelated_mutation"
                expected_action=actions
            if snapshot["label"]!=expected_label or snapshot["action"]!=expected_action:
                raise ValueError("Observed action/source action order differs")
            filename=task["id"]+"-"+expected_label+".png"
            if snapshot["filename"]!=filename or filename in seen:
                raise ValueError("Duplicate/unexpected screenshot receipt")
            seen.add(filename)
            png=(root/"screenshots"/filename).read_bytes()
            checked_png(png)
            if sha(png)!=snapshot["image_sha256"] or len(png)!=snapshot["image_bytes"]:
                raise ValueError("Authentic browser screenshot tampered or missing")
            actual={k:snapshot[k] for k in ("selected","amount","confirmed","unrelated_changed")}
            reference=simulate(task,actions)
            if actual != reference["state"]:
                raise ValueError("Browser DOM state differs from independent reference")
            expected_success=bool(index==3)
            if reference["success"]!=expected_success:
                raise ValueError("Synthetic screenshot milestone success incorrectly represented")
            if (snapshot["lang"]!=task["language"] or snapshot["rtl"]!=task["rtl"]
                or type(snapshot["display"]) is not str
                or ("Confirmed: "+("yes" if actual["confirmed"] else "no")) not in snapshot["display"]
                or ("Other changed: "+("yes" if actual["unrelated_changed"] else "no")) not in snapshot["display"]):
                raise ValueError("UI status presentation or locale differs from DOM state")
            controls=snapshot["controls"]
            if type(controls) is not list or len(controls)!=5:
                raise ValueError("Missing browser UI controls")
            bounds=[]
            for a,b in zip(controls,task["controls"]):
                if (type(a) is not dict or set(a)!={"ref","name","role","visible","bounds"}
                    or not all(a.get(k)==b[k] for k in ("ref","name","role"))
                    or a["visible"] is not True or len(a["bounds"])!=4):
                    raise ValueError("Observed browser control/ARIA semantics differ")
                v=a["bounds"]
                if (not all(type(x) in (int,float) and math.isfinite(x) for x in v)
                    or v[0]<0 or v[1]<0 or v[2]<35 or v[3]<25
                    or v[0]+v[2]>960 or v[1]+v[3]>540):
                    raise ValueError("Browser control outside viewport or too small")
                bounds.append(v)
            for i,(x,y,w,h) in enumerate(bounds):
                for q,(a,b,c,e) in enumerate(bounds):
                    if i<q and max(x,a)<min(x+w,a+c) and max(y,b)<min(y+h,b+e):
                        raise ValueError("Browser DOM controls overlap")
            if index<4:
                model_inputs.append({"id":task["id"],"phase":expected_label,
                  "goal":task["goal"],"language":task["language"],"split":task["split"],
                  "family":task["family"],"screenshot":"screenshots/"+filename,
                  "image_sha256":snapshot["image_sha256"],
                  "model_inference":False})
            oracle.append({"id":task["id"],"phase":expected_label,
                "screenshot_sha256":snapshot["image_sha256"],
                "expected_state":reference["state"],
                "expected_success":expected_success,
                "controls":controls,"model_does_not_receive_these_fields":True})
    return (
        {"schema":"hala.h45.model-only-steps.v1",
         "independent_blind_test":False,"model_qualified":False,
         "input_examples":model_inputs},
        {"schema":"hala.h45.evaluator-only-steps.v1",
         "independent_blind_test":False,"model_qualified":False,
         "reference_results":oracle},
        {"schema":"hala.h45.actual-browser-audit.v1",
         "original_h43_manifest_sha256":digest(source),
         "source_html_sha256":sha(plan_bytes),
         "capture_sha256":sha((root/"capture.json").read_bytes()),
         "source_tasks":24,"real_edge_screen_pngs":len(seen),
         "correct_golden_trajectories":24,"wrong_target_refused":24,
         "unrelated_mutation_refused":24,
         "model_only_steps":len(model_inputs),
         "native_desktop_actions":False,"independent_benchmark":False,
         "model_inference":False}
    )
