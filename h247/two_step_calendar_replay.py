"""H247: real source-native TWO-step calendar state verification.

H199 model trains/outputs exactly ONE native action per screenshot.
H246's task requires TWO distinct clicks: choose day, then click OK.
H247 separately captures the intermediate UI state and two bounded
click-targets. ALL owner-private and RESEARCH ONLY, NOT TRAIN/DEV/OOD.
Never reuses the consumed H173 scored OOD or opens old media/gold.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from h199.native_action_sft import parse_actions, source_task_prompt
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import require
from h243.browser_offline import require_browser_offline
from h244.private_research_overlap import private_json, sha_ok
from h246.original_calendar_native import (
    CASES, COMMIT, COUNT, FILES, REPO, preflight,
    self_authored_html,
)

STATUS="H247_TWO_STEP_NATIVE_CALENDAR_REPLAY_RESEARCH_NOT_ADMITTED"
OLD_STATUS="H246_THIRD_ORIGINAL_CALENDAR_NATIVE_RESEARCH_NOT_ADMITTED"
STAGES=("before_select","after_select_before_confirm","after_confirm")


def one_click_action(point:object)->dict:
    require(type(point) is list and len(point)==2
            and all(type(p) is int for p in point)
            and 0<=point[0]<960 and 0<=point[1]<720,
            "native screenshot click outside coordinates")
    action={"actions":[{"kind":"click","point":point}]}
    return parse_actions(json.dumps(action),training_target=True)


def verify_original_h246_research(record:dict)->None:
    """No H246 gold/images opened; only allowed source-only private receipt."""
    require(type(record) is dict
            and record.get("status")==OLD_STATUS
            and record.get("source_repository")==REPO
            and record.get("source_git_sha")==COMMIT
            and record.get("original_source_sha256")==FILES
            and record.get("original_mit_javascript_unmodified") is True
            and record.get("self_authored_html_fixture") is True
            and record.get("historical_source_visual_history_complete") is False
            and record.get("transitive_assets_rights_complete") is False
            and record.get("model_calls")==0
            and record.get("optimizer_steps")==0,
            "source family or original private calendar research was changed")
    for field in ("train_eligible","dev_eligible","ood_eligible"):
        require(record.get(field) is False, "old research split promoted")
    cases=record.get("cases")
    require(type(cases) is list and len(cases)==COUNT,
            "missing old calendar receipt cases")
    existing_png:set[str]=set()
    existing_rgba:set[str]=set()
    for i,row in enumerate(cases):
        expected=CASES[i]
        require(type(row) is dict and
                row.get("case_id")==expected["id"]
                and row.get("language")==expected["lang"]
                and row.get("instruction")==expected["instruction"]
                and row.get("calendar_day")==expected["day"]
                and row.get("real_native_calendar_selection_verified") is True
                and row.get("network_requests")==0
                and row.get("train_eligible") is False
                and row.get("dev_eligible") is False
                and row.get("ood_eligible") is False
                and type(row.get("confirmed_input_value")) is str
                and row["confirmed_input_value"].endswith(
                    f"-{expected['day']:02}"),
                "old native two-click case has changed or been promoted")
        one_click_action(row.get("calendar_click_point"))
        for stage in ("preaction","postaction"):
            p=row.get(stage+"_png_sha256")
            v=row.get(stage+"_rgba_sha256")
            require(sha_ok(p) and sha_ok(v)
                    and p not in existing_png and v not in existing_rgba,
                    "original receipt visual hash absent/duplicated")
            existing_png.add(p)
            existing_rgba.add(v)


def validate_two_step_replay(row:dict, case:dict)->None:
    """Validate per-screenshot step horizon, not an official TRAIN label."""
    require(type(row) is dict and case in CASES
            and row.get("case_id")==case["id"]
            and row.get("language")==case["lang"]
            and row.get("source_repository")==REPO
            and row.get("original_native_calendar_verified") is True
            and row.get("network_requests")==0
            and row.get("selected_day")==case["day"]
            and type(row.get("confirmed_date")) is str
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}",row["confirmed_date"])
                is not None
            and int(row["confirmed_date"][-2:])==case["day"],
            "native two-step source continuity missing")
    steps=row.get("steps")
    require(type(steps) is list and len(steps)==2,
            "H199 one-action horizon requires EXACTLY two separate observations")
    expected_prompts=(
        f"Select day {case['day']} from the open calendar."
        if case["lang"]=="en" else
        f"اختر اليوم {case['day']} من التقويم المفتوح.",
        "Confirm the chosen date with the OK button."
        if case["lang"]=="en" else
        "أكد التاريخ المحدد بالضغط على زر OK.",
    )
    for i,step in enumerate(steps):
        require(type(step) is dict and
                step.get("step_index")==i+1
                and step.get("instruction")==expected_prompts[i]
                and step.get("source_frame")==STAGES[i]
                and step.get("result_frame")==STAGES[i+1]
                and step.get("research_only") is True
                and step.get("train_eligible") is False
                and step.get("dev_eligible") is False
                and step.get("ood_eligible") is False,
                "invalid one-click step transition or role promotion")
        expected=one_click_action(step.get("click_point"))
        require(step.get("native_action")==expected
                and source_task_prompt(step["instruction"]),
                "one-step action parser/prompt roundtrip mismatch")
    require(steps[0]["click_point"]!=steps[1]["click_point"],
            "selection click and confirmation click cannot be identical")
    images=row.get("frames")
    require(type(images) is dict and set(images)==set(STAGES),
            "missing intermediate screenshot between native clicks")
    png:set[str]=set()
    rgba:set[str]=set()
    for stage in STAGES:
        img=images[stage]
        require(type(img) is dict
                and sha_ok(img.get("png_sha256"))
                and sha_ok(img.get("rgba_sha256"))
                and img["png_sha256"] not in png
                and img["rgba_sha256"] not in rgba,
                "missing or duplicate two-step calendar observation")
        png.add(img["png_sha256"])
        rgba.add(img["rgba_sha256"])


def visible_native_pixel(locator:object)->list[int]:
    """A recorded pixel must hit the original visible DOM target, not a locator shortcut."""
    box=locator.bounding_box()
    require(type(box) is dict
            and all(type(box.get(k)) in (int,float) for k in
                    ("x","y","width","height"))
            and all(math.isfinite(box[k]) for k in
                    ("x","y","width","height"))
            and box["width"]>0 and box["height"]>0,
            "original native click target lacks a finite visible bounding box")
    point=[int(box["x"]+box["width"]/2),
           int(box["y"]+box["height"]/2)]
    one_click_action(point)
    hit=locator.evaluate(
        """(element, pixel) => {
            const at = document.elementFromPoint(pixel[0], pixel[1]);
            return at !== null && (at === element || element.contains(at));
        }""",point)
    require(hit is True,
            "recorded screenshot pixel is occluded or does not hit original target")
    return point


def native_pixel_click(page:object,point:object)->None:
    """Execute exactly the screen pixel stored as the H199 one-action label."""
    one_click_action(point)
    page.mouse.click(point[0],point[1])


def capture(source:Path,output:Path,historical_parent:Path,owner_root:Path,
            h236:Path,h242:Path,h245:Path,h246:Path)->dict:
    require(output.name.startswith("mcdatepicker-private-research-h247-"),
            "use distinct H247 private replay output prefix")
    info=preflight(source,output,historical_parent,owner_root,h236,h242,h245)
    old=private_json(h246/"PRIVATE_RESEARCH_NOT_ADMITTED.json")
    verify_original_h246_research(old)
    from playwright.sync_api import sync_playwright
    output=info["output"]
    output.mkdir(parents=True,exist_ok=False)
    report={
        "status":STATUS,
        "source_repository":REPO,
        "source_git_sha":COMMIT,
        "original_source_sha256":dict(FILES),
        "prior_h246_research_verified":True,
        "source_history_complete":False,
        "visual_history_complete":False,
        "transitive_media_rights_complete":False,
        "train_approved":False,"dev_approved":False,"ood_approved":False,
        "model_calls":0,"optimizer_steps":0,"cases":[],
    }
    try:
        js=info["js"].read_text(encoding="utf-8")
        css=info["css"].read_text(encoding="utf-8")
        with sync_playwright() as pw:
            edge=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            kwargs={"headless":True}
            if edge.is_file(): kwargs["executable_path"]=str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("already installed Edge/Chromium required")
            browser=pw.chromium.launch(**kwargs)
            try:
                all_png:set[str]=set()
                all_rgba:set[str]=set()
                for case in CASES:
                    context=browser.new_context(viewport={"width":960,"height":720},
                        device_scale_factor=1,locale="ar-SA" if case["lang"]=="ar"
                            else "en-US",service_workers="block")
                    require_browser_offline(context)
                    attempts=[]
                    def block(route):
                        attempts.append(route.request.url)
                        route.abort()
                    context.route("**/*",block)
                    try:
                        page=context.new_page()
                        page.set_content(self_authored_html(case).replace(
                            "</style>",css+"</style>"))
                        page.evaluate(js)
                        require(page.evaluate("typeof window.MCDatepicker")=="object",
                                "native original calendar unavailable")
                        page.evaluate("""() => {window.__halaCalendar =
                            MCDatepicker.create({el:'#datepicker',dateFormat:'YYYY-MM-DD'})}""")
                        page.locator("#datepicker").click()
                        target=page.locator(".mc-date.mc-date--active").filter(
                            has_text=re.compile(r"^"+str(case["day"])+r"$"))
                        require(target.count()==1,"target calendar day not visible once")
                        select_point=visible_native_pixel(target)
                        first=page.screenshot(type="png")
                        native_pixel_click(page,select_point)
                        require(page.locator("#datepicker").input_value()==""
                                and page.locator(".mc-calendar--opened").count()==1,
                                "date click already confirmed or native dialog vanished")
                        confirm=page.locator(".mc-btn--success").filter(
                            has_text=re.compile(r"^OK$"))
                        require(confirm.count()==1,"native confirmation target missing")
                        confirm_point=visible_native_pixel(confirm)
                        intermediate=page.screenshot(type="png")
                        native_pixel_click(page,confirm_point)
                        value=page.locator("#datepicker").input_value()
                        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}",value) is not None
                                and int(value[-2:])==case["day"],
                                "actual native final date commit mismatch")
                        final=page.screenshot(type="png")
                        require(not attempts,"offline network attempted")
                        scenes=(first,intermediate,final)
                        frames={}
                        for stage,blob in zip(STAGES,scenes):
                            png=sha256(blob).hexdigest()
                            rgba=_rgba_sha(blob)
                            require(png not in all_png and rgba not in all_rgba,
                                    "new two-step visual duplicated within replay")
                            all_png.add(png);all_rgba.add(rgba)
                            frames[stage]={"png_sha256":png,"rgba_sha256":rgba}
                        def step(idx,instruction,point):
                            return {"step_index":idx,
                                    "instruction":instruction,
                                    "click_point":point,
                                    "native_action":one_click_action(point),
                                    "source_frame":STAGES[idx-1],
                                    "result_frame":STAGES[idx],
                                    "research_only":True,
                                    "train_eligible":False,
                                    "dev_eligible":False,
                                    "ood_eligible":False}
                        step_one=(f"Select day {case['day']} from the open calendar."
                                  if case["lang"]=="en" else
                                  f"اختر اليوم {case['day']} من التقويم المفتوح.")
                        step_two=("Confirm the chosen date with the OK button."
                                  if case["lang"]=="en" else
                                  "أكد التاريخ المحدد بالضغط على زر OK.")
                        row={"case_id":case["id"],"language":case["lang"],
                             "source_repository":REPO,
                             "selected_day":case["day"],
                             "confirmed_date":value,
                             "original_native_calendar_verified":True,
                             "network_requests":0,"frames":frames,
                             "steps":[step(1,step_one,select_point),
                                      step(2,step_two,confirm_point)]}
                        validate_two_step_replay(row,case)
                        for stage,blob in zip(STAGES,scenes):
                            (output/f"{case['id']}-{stage}.png").write_bytes(blob)
                        report["cases"].append(row)
                    finally:context.close()
            finally:browser.close()
        require(len(report["cases"])==COUNT
                and len(all_png)==COUNT*3 and len(all_rgba)==COUNT*3
                and sum(c["language"]=="ar" for c in report["cases"])==4,
                "24-frame bilingual 2-click native replay incomplete")
        (output/"PRIVATE_TWO_STEP_REPLAY_NOT_ADMITTED.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
            encoding="utf-8")
        return report
    except BaseException:
        (output/"INCOMPLETE_NEVER_ADMIT").write_text(
            "No TRAIN, DEV or OOD from a failed calendar action replay.\n",
            encoding="utf-8")
        raise


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for x in ("source-root","output","historical-parent","owner-root",
              "h236-private","h242-private","h245-private","h246-private"):
        a.add_argument("--"+x,type=Path,required=True)
    opts=a.parse_args()
    try:
        result=capture(opts.source_root,opts.output,opts.historical_parent,
                       opts.owner_root,opts.h236_private,opts.h242_private,
                       opts.h245_private,opts.h246_private)
    except (ValueError,OSError,KeyError,TypeError):
        a.exit(2,"H247_TWO_STEP_OWNER_REPLAY_NOT_QUALIFIED\n")
    print(json.dumps({
        "status":result["status"],
        "source_native_cases":len(result["cases"]),
        "one_action_steps":sum(len(c["steps"]) for c in result["cases"]),
        "new_private_frames":len(result["cases"])*len(STAGES),
        "train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
