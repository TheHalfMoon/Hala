"""H245: independent original MIT html5sortable native browser research scenes.

Source-only public implementation. Owner screenshots and reference task receipts
NEVER leave a separate create-only private directory. NOT TRAIN/DEV/OOD.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from h174.split_firewall import canonical_repository_family
from h178.owner_git_inventory import inventory_local_original_sources
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h198.owner_source_metadata_readiness import census
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import git, require
from h238.historical_family_floor import known_public_source_family_floor
from h241.expanded_pre173_metadata import find_pre_h173_phases
from h243.browser_offline import require_browser_offline
from h244.private_research_overlap import current_20_metadata, historical_h236_research, private_json, verify_h242_receipt

REPO = "lukasoppermann/html5sortable"
COMMIT = "f3859acb7b7f29b833badff569a33c4fca95e1f8"
FILES = {
    "LICENSE.md": "59117c538087ee275ecb1383dd80cee4347b9f0d6bd5f5b7fe9ae9e42b7239e0",
    "dist/html5sortable.min.js": "7db5872cd9c934e51ac68589eccd4f0ce11a229c00d9ccecc2d591b82a93d949",
    "package.json": "512f4326f5b9a4b12aaaf27e37b2265a8f12e697238548490ba23e3d690009db",
}
COUNT = 8
CASES = tuple({
    "id": f"html5sortable-research-{i+1:02}",
    "lang": "en" if i % 2 == 0 else "ar",
    "number": i + 1,
    "shade": ("#f5e5d1","#d8f0ec","#e8e5fa","#eaf0cf")[i % 4],
    "task": (f"Move card {i+1} after the other three cards."
             if i % 2 == 0 else f"انقل البطاقة {i+1} إلى نهاية القائمة."),
} for i in range(COUNT))


def self_authored_html(case: dict) -> str:
    require(case in CASES, "unregistered original-source research case")
    return """<!doctype html><html lang='""" + case["lang"] + """'>
<meta charset='utf-8'><title>H245 private native sortable research</title>
<style>body{font:22px Arial;margin:0;background:#f4f5f6}
main{padding:24px}#list{width:450px;margin:25px;background:white;padding:16px;
border:2px solid #242424;min-height:400px}
.item{padding:26px;margin:8px;border:1px solid #555;background:#eff0f4}
#source{background:""" + case["shade"] + """}
</style><main><div id='list' class='sortable'>
<div id='source' class='item'>CARD """ + str(case["number"]) + """</div>
<div id='second' class='item'>SECOND</div>
<div id='third' class='item'>THIRD</div>
<div id='fourth' class='item'>FOURTH</div>
</div></main></html>"""


def preflight(source: Path, output: Path, historical_parent: Path,
              owner_root: Path, prior_h242: Path, prior_h236: Path) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    require(Path(git(source,"rev-parse","--show-toplevel")).resolve() == source,
            "source root not exact Git checkout")
    require(git(source,"rev-parse","HEAD") == COMMIT,
            "original source commit mismatch")
    require(git(source,"status","--porcelain","--untracked-files=all") == "",
            "original source is dirty")
    require(git(source,"remote","get-url","origin").casefold() in (
        "https://github.com/lukasoppermann/html5sortable",
        "https://github.com/lukasoppermann/html5sortable.git",
        "git@github.com:lukasoppermann/html5sortable",
        "git@github.com:lukasoppermann/html5sortable.git",
    ), "original Git upstream mismatch")
    for filename, digest in FILES.items():
        file = source.joinpath(*filename.split("/"))
        require(file.is_file() and not file.is_symlink()
                and file.resolve().is_relative_to(source)
                and sha256(file.read_bytes()).hexdigest() == digest,
                "original public MIT source asset bytes mismatch")
    require(b"MIT" in (source/"LICENSE.md").read_bytes(),
            "original MIT license missing")
    require(canonical_repository_family(REPO) not in known_public_source_family_floor(),
            "already documented public historical source family")
    history = inventory_local_original_sources([historical_parent])
    seen = {repo for repo, _ in history["_owner_only_repository_shas"]}
    require(canonical_repository_family(REPO) not in seen,
            "already observed private historical original source")
    require(output.name.startswith("html5sortable-research-")
            and not output.exists()
            and not output.is_relative_to(source)
            and not any((p/".git").exists() for p in (output,*output.parents)),
            "output must be fresh, outside every Git tree")
    phases = find_pre_h173_phases([historical_parent])
    old = scan_legacy_manifest_sha_fields(phases)
    require(old["complete_historical_manifest_coverage_proven"] is False
            and old["training_dataset_admitted"] is False
            and old["old_image_media_opened"] is False,
            "unknown historical media must not be considered admitted")
    current_png, current_rgba = current_20_metadata(owner_root)
    h236_png, h236_rgba = historical_h236_research(prior_h236)
    prior = private_json(prior_h242/"PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json")
    verify_h242_receipt(prior, set(old["_owner_only_png_sha_candidates"])
                        |current_png|h236_png,
                        set(old["_owner_only_rgba_sha_candidates"])
                        |current_rgba|h236_rgba)
    prior_png = {r[s+"_png_sha256"] for r in prior["cases"]
                 for s in ("preaction","postaction")}
    prior_rgba = {r[s+"_rgba_sha256"] for r in prior["cases"]
                  for s in ("preaction","postaction")}
    require(len(prior_png) == len(prior_rgba) == 16,
            "previous distinct native frames incomplete")
    return {
        "asset": source/"dist/html5sortable.min.js",
        "output": output,
        "old_png": set(old["_owner_only_png_sha_candidates"])|current_png|h236_png|prior_png,
        "old_rgba": set(old["_owner_only_rgba_sha_candidates"])|current_rgba|h236_rgba|prior_rgba,
        "older_exclusions": len(old["_owner_only_png_sha_candidates"]),
        "current_research": len(current_png),
        "prior_native": len(prior_png)+len(h236_png),
    }


def capture(source: Path, output: Path, historical_parent: Path,
            owner_root: Path, prior_h242: Path, prior_h236: Path) -> dict:
    info=preflight(source,output,historical_parent,owner_root,prior_h242,prior_h236)
    from playwright.sync_api import sync_playwright
    output=info["output"]
    output.mkdir(parents=True,exist_ok=False)
    result={
        "status":"H245_INDEPENDENT_NATIVE_SOURCE_PRIVATE_RESEARCH_NOT_ADMITTED",
        "source_repository":REPO,"source_git_sha":COMMIT,
        "original_source_sha256":dict(FILES),
        "unmodified_original_mit_js":True,
        "self_authored_ui_fixture":True,
        "known_old_png_exclusions":info["older_exclusions"],
        "other_research_cohort_cases":info["current_research"],
        "prior_native_research_excluded_frames":info["prior_native"],
        "historical_source_and_visual_completeness":False,
        "third_party_and_transitive_media_rights_audited":False,
        "train_approved":False,"dev_approved":False,"ood_approved":False,
        "model_calls":0,"optimizer_steps":0,"cases":[],
    }
    png_seen=set()
    rgba_seen=set()
    try:
        with sync_playwright() as pw:
            edge=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            kwargs={"headless":True}
            if edge.is_file():
                kwargs["executable_path"]=str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("already installed Edge/Chromium required")
            browser=pw.chromium.launch(**kwargs)
            try:
                for row in CASES:
                    ctx=browser.new_context(
                        viewport={"width":960,"height":720},
                        device_scale_factor=1,service_workers="block",
                        locale="ar-SA" if row["lang"]=="ar" else "en-US")
                    require_browser_offline(ctx)
                    attempted=[]
                    def abort_route(route):
                        attempted.append(route.request.url)
                        route.abort()
                    ctx.route("**/*",abort_route)
                    try:
                        page=ctx.new_page()
                        page.set_content(self_authored_html(row),wait_until="load")
                        page.add_script_tag(path=str(info["asset"]))
                        require(page.evaluate("typeof window.sortable")=="function",
                                "actual original HTML5Sortable library missing")
                        page.evaluate("sortable('.sortable',{items:'.item',forcePlaceholderSize:true})")
                        start_box=page.locator("#source").bounding_box()
                        end_box=page.locator("#fourth").bounding_box()
                        require(start_box is not None and end_box is not None,
                                "native source/target not visible")
                        xy=lambda b:[int(b["x"]+b["width"]/2),int(b["y"]+b["height"]/2)]
                        start,end=xy(start_box),xy(end_box)
                        require(all(0<=p[0]<960 and 0<=p[1]<720 for p in (start,end)),
                                "drag outside visible screenshot")
                        before=page.screenshot(type="png")
                        require(page.locator("#list > .item").count()==4,
                                "original source list mismatch")
                        page.locator("#source").drag_to(page.locator("#fourth"),timeout=7000)
                        require(page.locator("#list > .item").last.get_attribute("id")=="source",
                                "real original html5sortable native reorder failed")
                        require(not attempted,"network request blocked; capture invalid")
                        after=page.screenshot(type="png")
                        pairs=[(sha256(blob).hexdigest(),_rgba_sha(blob))
                               for blob in (before,after)]
                        require(pairs[0] != pairs[1],"no visible screenshot transition")
                        for h,v in pairs:
                            require(h not in info["old_png"] and v not in info["old_rgba"],
                                    "new source visual collides with previous research/history")
                            require(h not in png_seen and v not in rgba_seen,
                                    "duplicate newly generated native visual")
                            png_seen.add(h);rgba_seen.add(v)
                        for stage,blob in (("before",before),("after",after)):
                            (output/f"{row['id']}-{stage}.png").write_bytes(blob)
                        result["cases"].append({
                            "case_id":row["id"],"language":row["lang"],
                            "instruction":row["task"],
                            "source_point":start,"destination_point":end,
                            "preaction_png_sha256":pairs[0][0],
                            "preaction_rgba_sha256":pairs[0][1],
                            "postaction_png_sha256":pairs[1][0],
                            "postaction_rgba_sha256":pairs[1][1],
                            "real_native_reorder_verified":True,
                            "network_attempts":0,
                            "train_eligible":False,"dev_eligible":False,
                            "ood_eligible":False,
                        })
                    finally:
                        ctx.close()
            finally:
                browser.close()
        require(len(result["cases"])==COUNT and len(png_seen)==16
                and len(rgba_seen)==16
                and sum(row["language"]=="en" for row in result["cases"])==4,
                "new independent-source bilingual scene suite incomplete")
        (output/"PRIVATE_RESEARCH_NOT_ADMITTED.json").write_text(
            json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return result
    except BaseException:
        (output/"INCOMPLETE_NO_ADMISSION").write_text(
            "Failed or incomplete; never TRAIN/DEV/OOD.\n",encoding="utf-8")
        raise


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","output","historical-parent","owner-root","h242-private","h236-private"):
        a.add_argument("--"+name,required=True,type=Path)
    x=a.parse_args()
    try:
        result=capture(x.source_root,x.output,x.historical_parent,x.owner_root,
                       x.h242_private,x.h236_private)
    except (ValueError,OSError,KeyError,TypeError):
        a.exit(2,"H245_INDEPENDENT_ORIGINAL_SOURCE_NOT_QUALIFIED\n")
    print(json.dumps({
        "status":result["status"],"cases":len(result["cases"]),
        "en":sum(c["language"]=="en" for c in result["cases"]),
        "ar":sum(c["language"]=="ar" for c in result["cases"]),
        "source_families_reused":0,"train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
