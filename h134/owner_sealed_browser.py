"""H134 owner Windows independent once-only sealed original GUI execution.

Inputs: immutable real neural outputs on SEALED_TEST16 and frozen adapter hash.
Never use DOM/true boxes for model action choice, only for scoring AFTER output.
"""
import json
import re
import statistics
from pathlib import Path

from h134.source import validate,sha
from h134.sealed_gpu import SEALED_MANIFEST_SHA,TRAINED_ADAPTER_SHA
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h133.paired_stats import exact_discordant_p
from h131.owner_diagnostics import distance_to_box

def verify_native(source,model_rows):
    if not isinstance(model_rows,list) or len(model_rows)!=32:
        raise ValueError("H134 once-only sealed16 requires exactly 32 real model roles")
    cases=source["cases"]
    predictions={}
    for row,source_record in zip(model_rows,
       ((case,role) for case in cases for role in ("start","end"))):
        case,role=source_record
        if (row.get("source_id")!=case["source_id"] or
            row.get("image_sha256")!=case["image_sha256"] or
            row.get("role")!=role):
            raise ValueError("H134 sealed native neural case/role/image mismatch")
        raw= row.get("raw")
        inferred=native_point(raw,WIDTH,HEIGHT)
        if (inferred is None and row.get("prediction") is not None or
            inferred is not None and row.get("prediction")!=inferred):
            raise ValueError("H134 sealed raw native response/point tampered")
        bbox=case[role+"_bbox"]
        hit=inferred is not None and (
            bbox["x"]<=inferred[0]<=bbox["x"]+bbox["width"] and
            bbox["y"]<=inferred[1]<=bbox["y"]+bbox["height"])
        if row.get("hit") is not bool(hit):
            raise ValueError("H134 sealed reported native bbox hit cannot be reproduced")
        predictions.setdefault(case["source_id"],{})[role]=inferred
    return predictions

def actual_browser_trajectories(cases,source_predictions):
    from playwright.sync_api import sync_playwright
    fixture=Path(__file__).with_name("fixture.html").resolve()
    results=[]
    with sync_playwright() as p:
        kwargs={"headless":True}
        binary=_chromium_binary()
        if binary:kwargs["executable_path"]=binary
        browser=p.chromium.launch(**kwargs)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route(re.compile(r"^https?://"),lambda route:route.abort())
            page=ctx.new_page()
            for case in cases:
                seed=case["source_id"].rsplit("-",1)[1]
                page.goto(fixture.as_uri()+"?seed="+seed,wait_until="load")
                page.locator('[data-role="handle"]').wait_for()
                image=page.screenshot(type="png")
                if sha(image)!=case["image_sha256"]:
                    raise ValueError("H134 original sealed screenshot drift on actual owner Chromium")
                points=source_predictions[case["source_id"]]
                start=points["start"];end=points["end"]
                if start is None or end is None:
                    succeeded=False;changed=False
                else:
                    page.mouse.move(*start)
                    page.mouse.down()
                    page.mouse.move(*end,steps=12)
                    page.mouse.up()
                    succeeded=page.locator("#stage").get_attribute("data-success")=="true"
                    changed=sha(page.screenshot(type="png"))!=sha(image)
                results.append({
                    "source_id":case["source_id"],
                    "browser_success":succeeded,
                    "browser_changed":changed,
                    "native_start_present":start is not None,
                    "native_end_present":end is not None,
                })
            ctx.close()
        finally:browser.close()
    return results

def replay(source_dir,output_root, private_result):
    source_dir,output_root,private_result=map(Path,(source_dir,output_root,private_result))
    source=validate(source_dir,expect="sealed_test")
    if sha((source_dir/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H134 one-time source manifest no longer frozen")
    result=json.loads((output_root/"private-sealed16-scores.json").read_text("utf8"))
    if (result.get("status")!="H134_REAL_PRIVATE_ONE_TIME_SEALED16_SCORED" or
        result.get("adapter_sha256")!=TRAINED_ADAPTER_SHA or
        result.get("sealed_source_sha256")!=SEALED_MANIFEST_SHA or
        result.get("actual_optimizer_steps")!=0 or
        result.get("real_neural_calls")!=64 or
        result.get("original_train_dev_scenes_in_eval")!=0):
        raise ValueError("H134 frozen GPU sealed evidence lacks strict provenance")
    outcomes={}
    for arm,name in (("pretrained","private-sealed16-before.json"),
                     ("adapter","private-sealed16-after.json")):
        data=json.loads((output_root/name).read_text("utf8"))
        points=verify_native(source,data)
        outcomes[arm]=actual_browser_trajectories(source["cases"],points)
        for role in ("start","end"):
            count=sum(bool(row["hit"]) for row in data if row["role"]==role)
            if count!=result[arm][role]["hit"]:
                raise ValueError("H134 sealed raw source bbox score mismatch")
        if result[arm]["both_targets_hit"]!=sum(
           all(bool(x["hit"]) for x in data[i:i+2])
           for i in range(0,len(data),2)):
            raise ValueError("H134 raw both-points score mismatch")
    before={x["source_id"]:x["browser_success"] for x in outcomes["pretrained"]}
    after={x["source_id"]:x["browser_success"] for x in outcomes["adapter"]}
    improved_only=sum(not before[k] and after[k] for k in before)
    lost_only=sum(before[k] and not after[k] for k in before)
    private={
        "schema":"hala.h134.owner-independent-sealed16-real-GUI-replay.v1",
        "status":"H134_PRIVATE_ONCE_ONLY_SEALED16_NEURAL_AND_BROWSER_VERIFIED",
        "source_sha256":SEALED_MANIFEST_SHA,
        "adapter_sha256":TRAINED_ADAPTER_SHA,
        "actual_sealed_screenshots":16,
        "real_neural_inferences":64,
        "actual_browser_drag_attempts":sum(
            sum(x["native_start_present"] and x["native_end_present"]
                for x in outcomes[arm]) for arm in outcomes),
        "before_full_browser_successes":sum(before.values()),
        "after_full_browser_successes":sum(after.values()),
        "paired_adapter_only":improved_only,
        "paired_pretrained_only":lost_only,
        "exact_p_two_sided":exact_discordant_p(improved_only,lost_only),
        "dev_or_train_data_reused_for_sealed_eval":False,
        "main_Hala_flagship_improvement_proven":False,
        "sample_size_caution":"16 synthetic tasks only",
    }
    if private_result.exists():
        raise ValueError("H134 owner sealed audit output create-only")
    private_result.parent.mkdir(parents=True,exist_ok=True)
    private_result.write_text(json.dumps(private,indent=2,sort_keys=True)+"\n","utf8")
    return private
