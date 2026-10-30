"""H246: third original MIT native GUI research source — calendar date selection.

Never TRAIN, DEV, OOD, checkpoint, model inference or optimizer.
Public module; output owner-private create-only, never enter Git.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from h174.split_firewall import canonical_repository_family
from h178.owner_git_inventory import inventory_local_original_sources
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import git, require
from h238.historical_family_floor import known_public_source_family_floor
from h241.expanded_pre173_metadata import find_pre_h173_phases
from h243.browser_offline import require_browser_offline
from h244.private_research_overlap import (
    current_20_metadata, historical_h236_research, private_json,
    verify_h242_receipt, sha_ok,
)
from h245.independent_html5sortable_native import (
    REPO as H245_REPO, COMMIT as H245_COMMIT, FILES as H245_FILES,
)

REPO = "mikecoj/mcdatepicker"
COMMIT = "17dcf1ac963f38395a310f9198b427ea0d36af71"
FILES = {
    "LICENSE": "a251a8e3fe7e463993089a57da100b8e3b328d9289f815f01db4876c51ee7676",
    "dist/mc-calendar.min.js": "6b3204751a0f764091667fdc9ce8bc524e14a09c4b2ad008e7dcd810ad7634b7",
    "dist/mc-calendar.min.css": "747183eb2e22a51e72627afb3fcc9fe46e3551fa9e2ddc81b8677d728b8b7c95",
    "package.json": "25e84ed661f8ca1adddf60c683f5facc2aa6cad165e1f316af8cbf2116236523",
}
COUNT = 8
CASES = tuple({
    "id": f"mcdatepicker-calendar-research-{i+1:02}",
    "lang": "en" if i%2 == 0 else "ar",
    "day": 8+i,
    "shade": ("#f0e5fb","#e5f0e2","#ffe9dd","#e2effa")[i%4],
    "instruction": (f"Select day {8+i} in the calendar and confirm."
                    if i%2 == 0 else f"اختر اليوم {8+i} من التقويم ثم أكد الاختيار."),
} for i in range(COUNT))


def self_authored_html(case: dict) -> str:
    require(case in CASES, "unknown calendar research task")
    return """<!doctype html><html lang='""" + case["lang"] + """'>
<meta charset=utf-8><title>H246 Private Calendar Source Research</title>
<style>body{font:20px Arial,sans-serif;margin:0;background:#f7f7f8}
main{margin:26px;padding:20px;max-width:660px;border:2px solid #333;
background:""" + case["shade"] + """}
label{display:block;margin:15px;font-weight:600}
input{font-size:22px;padding:16px;border:1px solid #666;background:white}
</style><main><label for='datepicker'>""" + case["instruction"] + """</label>
<input id='datepicker' type='text' readonly placeholder='Click for calendar'></main></html>"""


def _h245_exclusions(directory: Path) -> tuple[set[str],set[str]]:
    """Source-only H245 privacy-scope receipt; never H245 PNG bytes."""
    data = private_json(directory/"PRIVATE_RESEARCH_NOT_ADMITTED.json")
    require(data.get("status") ==
            "H245_INDEPENDENT_NATIVE_SOURCE_PRIVATE_RESEARCH_NOT_ADMITTED"
            and data.get("source_repository") == H245_REPO
            and data.get("source_git_sha") == H245_COMMIT
            and data.get("original_source_sha256") == H245_FILES
            and data.get("unmodified_original_mit_js") is True
            and data.get("self_authored_ui_fixture") is True,
            "prior original native source identity changed")
    for label in ("historical_source_and_visual_completeness",
                  "third_party_and_transitive_media_rights_audited",
                  "train_approved","dev_approved","ood_approved"):
        require(data.get(label) is False, "prior HTML5Sortable dataset promoted")
    require(data.get("model_calls")==data.get("optimizer_steps")==0,
            "prior source-native model calls unexpected")
    cases = data.get("cases")
    require(type(cases) is list and len(cases)==8,
            "expected eight prior HTML5Sortable research cases")
    png:set[str]=set()
    rgba:set[str]=set()
    for i,row in enumerate(cases):
        require(type(row) is dict and
                row.get("case_id")==f"html5sortable-research-{i+1:02}"
                and row.get("language")==("en" if i%2==0 else "ar")
                and row.get("real_native_reorder_verified") is True
                and row.get("network_attempts")==0,
                "prior HTML5Sortable native oracle changed")
        for key in ("train_eligible","dev_eligible","ood_eligible"):
            require(row.get(key) is False,"prior source research role changed")
        for stage in ("preaction","postaction"):
            p,v=row.get(stage+"_png_sha256"),row.get(stage+"_rgba_sha256")
            require(sha_ok(p) and sha_ok(v)
                    and p not in png and v not in rgba,
                    "prior native image SHA roster duplicated or untrusted")
            png.add(p)
            rgba.add(v)
    require(len(png)==len(rgba)==16,"not all 16 prior native frames")
    return png,rgba


def preflight(source:Path,output:Path,historical_parent:Path,owner_root:Path,
              h236:Path,h242:Path,h245:Path)->dict:
    source=source.resolve(strict=True)
    output=output.resolve()
    require(Path(git(source,"rev-parse","--show-toplevel")).resolve()==source,
            "source must be exact original Git checkout")
    require(git(source,"rev-parse","HEAD")==COMMIT
            and git(source,"status","--porcelain","--untracked-files=all")=="",
            "original upstream SHA or clean worktree mismatch")
    require(git(source,"remote","get-url","origin").casefold() in (
        "https://github.com/mikecoj/mcdatepicker",
        "https://github.com/mikecoj/mcdatepicker.git",
        "git@github.com:mikecoj/mcdatepicker",
        "git@github.com:mikecoj/mcdatepicker.git",
    ),"unapproved upstream Git origin")
    for filename,digest in FILES.items():
        file=source.joinpath(*filename.split("/"))
        require(file.is_file() and not file.is_symlink()
                and file.resolve().is_relative_to(source)
                and sha256(file.read_bytes()).hexdigest()==digest,
                "original MIT asset or license SHA mismatch")
    require(b"MIT" in (source/"LICENSE").read_bytes(),
            "actual original MIT license missing")
    require(canonical_repository_family(REPO)
            not in known_public_source_family_floor(),
            "previously published historical original donor")
    inventory=inventory_local_original_sources([historical_parent])
    require(canonical_repository_family(REPO) not in
            {repo for repo,_ in inventory["_owner_only_repository_shas"]},
            "already consumed owner-only source family")
    require(output.name.startswith("mcdatepicker-private-research-")
            and not output.exists()
            and not any((p/".git").exists() for p in (output,*output.parents)),
            "private output must be create-only outside all Git")
    phases=find_pre_h173_phases([historical_parent])
    old=scan_legacy_manifest_sha_fields(phases)
    require(old["complete_historical_manifest_coverage_proven"] is False
            and old["training_dataset_admitted"] is False
            and old["old_image_media_opened"] is False,
            "old historical media must remain unqualified and unopened")
    current_png,current_rgba=current_20_metadata(owner_root)
    h236_png,h236_rgba=historical_h236_research(h236)
    prev=private_json(h242/"PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json")
    verify_h242_receipt(prev,set(old["_owner_only_png_sha_candidates"])|current_png|h236_png,
                        set(old["_owner_only_rgba_sha_candidates"])|current_rgba|h236_rgba)
    h242_png={r[s+"_png_sha256"] for r in prev["cases"] for s in ("preaction","postaction")}
    h242_rgba={r[s+"_rgba_sha256"] for r in prev["cases"] for s in ("preaction","postaction")}
    require(len(h242_png)==len(h242_rgba)==16,"prior Dragula frames incomplete")
    h245_png,h245_rgba=_h245_exclusions(h245)
    require(not (h245_png & h242_png) and not (h245_rgba & h242_rgba),
            "prior two native sources visually overlap")
    return {
        "output":output,
        "js":source/"dist/mc-calendar.min.js",
        "css":source/"dist/mc-calendar.min.css",
        "old_png":set(old["_owner_only_png_sha_candidates"])|current_png|h236_png|h242_png|h245_png,
        "old_rgba":set(old["_owner_only_rgba_sha_candidates"])|current_rgba|h236_rgba|h242_rgba|h245_rgba,
        "old_png_count":len(old["_owner_only_png_sha_candidates"]),
        "other_known_research_count":len(current_png),
        "prior_native_frames":len(h236_png)+len(h242_png)+len(h245_png),
    }


def capture(source:Path,output:Path,historical_parent:Path,owner_root:Path,
            h236:Path,h242:Path,h245:Path)->dict:
    info=preflight(source,output,historical_parent,owner_root,h236,h242,h245)
    from playwright.sync_api import sync_playwright
    output=info["output"]
    output.mkdir(parents=True,exist_ok=False)
    result={
        "status":"H246_THIRD_ORIGINAL_CALENDAR_NATIVE_RESEARCH_NOT_ADMITTED",
        "source_repository":REPO,"source_git_sha":COMMIT,
        "original_source_sha256":dict(FILES),
        "original_mit_javascript_unmodified":True,
        "self_authored_html_fixture":True,
        "known_old_png_exclusions":info["old_png_count"],
        "other_research_cases":info["other_known_research_count"],
        "prior_native_visuals":info["prior_native_frames"],
        "historical_source_visual_history_complete":False,
        "transitive_assets_rights_complete":False,
        "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
        "model_calls":0,"optimizer_steps":0,"cases":[],
    }
    seen_png:set[str]=set()
    seen_rgba:set[str]=set()
    try:
        css=info["css"].read_text(encoding="utf-8")
        js=info["js"].read_text(encoding="utf-8")
        with sync_playwright() as pw:
            edge=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            kwargs={"headless":True}
            if edge.is_file():kwargs["executable_path"]=str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("no installed local Edge/Chromium")
            browser=pw.chromium.launch(**kwargs)
            try:
                for row in CASES:
                    ctx=browser.new_context(viewport={"width":960,"height":720},
                        device_scale_factor=1,service_workers="block",
                        locale="ar-SA" if row["lang"]=="ar" else "en-US")
                    require_browser_offline(ctx)
                    attempts=[]
                    def deny(route):
                        attempts.append(route.request.url)
                        route.abort()
                    ctx.route("**/*",deny)
                    try:
                        page=ctx.new_page()
                        markup=self_authored_html(row)
                        page.set_content(markup.replace("</style>",css+"</style>"))
                        page.evaluate(js)
                        require(page.evaluate("typeof window.MCDatepicker")=="object",
                                "original real MCDatepicker global missing")
                        page.evaluate("window.__halaCalendar = MCDatepicker.create({el:'#datepicker',dateFormat:'YYYY-MM-DD'})")
                        page.locator("#datepicker").click()
                        require(page.locator(".mc-calendar--opened").count()==1,
                                "real datepicker calendar failed to appear")
                        target=page.locator(".mc-date.mc-date--active").filter(
                            has_text=re.compile(r"^"+str(row["day"])+r"$"))
                        require(target.count()==1,"chosen day not uniquely active")
                        bounds=target.bounding_box()
                        require(bounds is not None,"calendar selection not visible")
                        point=[int(bounds["x"]+bounds["width"]/2),
                               int(bounds["y"]+bounds["height"]/2)]
                        require(0<=point[0]<960 and 0<=point[1]<720,
                                "calendar click outside screenshot")
                        before=page.screenshot(type="png")
                        target.click(timeout=5000)
                        page.locator(".mc-btn--success").filter(
                            has_text=re.compile(r"^OK$")).click(timeout=5000)
                        value=page.locator("#datepicker").input_value()
                        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}",value) is not None
                                and int(value[-2:])==row["day"],
                                "original date selection not committed to input")
                        after=page.screenshot(type="png")
                        require(not attempts,"network request attempted")
                        fingerprints=[(sha256(b).hexdigest(),_rgba_sha(b)) for b in (before,after)]
                        for p,v in fingerprints:
                            require(p not in info["old_png"] and v not in info["old_rgba"],
                                    "calendar visual reuses known original history")
                            require(p not in seen_png and v not in seen_rgba,
                                    "new calendar scene visual duplicate")
                            seen_png.add(p);seen_rgba.add(v)
                        for stage,blob in (("before",before),("after",after)):
                            (output/f"{row['id']}-{stage}.png").write_bytes(blob)
                        result["cases"].append({
                            "case_id":row["id"],"language":row["lang"],
                            "instruction":row["instruction"],"calendar_day":row["day"],
                            "calendar_click_point":point,
                            "confirmed_input_value":value,
                            "preaction_png_sha256":fingerprints[0][0],
                            "preaction_rgba_sha256":fingerprints[0][1],
                            "postaction_png_sha256":fingerprints[1][0],
                            "postaction_rgba_sha256":fingerprints[1][1],
                            "real_native_calendar_selection_verified":True,
                            "network_requests":0,
                            "train_eligible":False,"dev_eligible":False,
                            "ood_eligible":False,
                        })
                    finally:ctx.close()
            finally:browser.close()
        require(len(result["cases"])==COUNT and len(seen_png)==16 and
                len(seen_rgba)==16 and
                sum(c["language"]=="en" for c in result["cases"])==4,
                "balanced unique original calendar scenes incomplete")
        (output/"PRIVATE_RESEARCH_NOT_ADMITTED.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
            encoding="utf-8")
        return result
    except BaseException:
        (output/"INCOMPLETE_NO_ADMISSION").write_text(
            "Never use incomplete calendar research TRAIN/DEV/OOD.\n",encoding="utf-8")
        raise


def main()->None:
    a=argparse.ArgumentParser(description=__doc__)
    for key in ("source-root","output","historical-parent","owner-root",
                "h236-private","h242-private","h245-private"):
        a.add_argument("--"+key,type=Path,required=True)
    x=a.parse_args()
    try:
        result=capture(x.source_root,x.output,x.historical_parent,
                       x.owner_root,x.h236_private,x.h242_private,x.h245_private)
    except (ValueError,KeyError,OSError,TypeError):
        a.exit(2,"H246_ORIGINAL_CALENDAR_RESEARCH_NOT_QUALIFIED\n")
    print(json.dumps({
        "status":result["status"],
        "cases":len(result["cases"]),
        "en":sum(row["language"]=="en" for row in result["cases"]),
        "ar":sum(row["language"]=="ar" for row in result["cases"]),
        "new_source_families_admitted":0,
        "train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
