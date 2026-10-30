"""H242 source-native original MIT Dragula bilingual private research prospects.

ALL source-only public code. Any image, labeled task or receipt is written
ONLY to the owner's private, create-only out-of-Git directory. These are
NOT TRAIN/DEV/OOD, do not load a model, and cannot update the optimizer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from h174.split_firewall import canonical_repository_family
from h178.owner_git_inventory import inventory_local_original_sources
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import (
    COMMIT, PINNED, REPO, preflight, require,
)
from h241.expanded_pre173_metadata import find_pre_h173_phases
from h243.browser_offline import require_browser_offline

CASE_COUNT = 8
CASES = tuple({
    "case_id": f"dragula-private-prospect-{i+1:02}",
    "lang": ("en" if i % 2 == 0 else "ar"),
    "accent": ("#dfe8fd", "#fce5ed", "#dff5e4", "#f9edce")[i % 4],
    "target": ("#edf8e9", "#ecefff", "#fff1e7", "#eaf7ff")[i % 4],
    "card": i + 1,
    "task": (
        f"Move visible card {i+1} into the destination column."
        if i % 2 == 0 else f"انقل البطاقة الظاهرة رقم {i+1} إلى عمود الوجهة."
    ),
} for i in range(CASE_COUNT))


def fixture(case: dict) -> str:
    """Authored offline canvas; original source-native Dragula JS, not mock JS."""
    require(type(case) is dict and case in CASES, "unknown private QA context")
    return """<!doctype html><html lang='""" + case["lang"] + """'>
<meta charset='utf-8'><title>H242 PRIVATE SOURCE RESEARCH - NOT TRAIN</title>
<style>
body{font:21px Arial;margin:0;background:#fafafa}
main{padding:20px}
section{display:inline-block;vertical-align:top;width:350px;min-height:350px;
border:2px solid black;margin:20px;padding:14px;background:white}
.card{padding:24px;border:1px solid #888;margin:15px;cursor:grab}
#moving{background:""" + case["accent"] + """}
#target{background:""" + case["target"] + """}
.gu-mirror{position:fixed!important;margin:0!important;z-index:9999}
.gu-hide{display:none!important}.gu-unselectable{user-select:none!important}
.gu-transit{opacity:.2}
</style><main>
<section id='left' aria-label='Source'>
<div id='moving' class='card'>CARD """ + str(case["card"]) + """</div>
<div id='other' class='card'>OTHER</div></section>
<section id='right' aria-label='Destination'>
<div id='target' class='card'>TARGET</div></section></main></html>"""


def _strict_local_source(source: Path, output: Path,
                         history_parent: Path) -> dict:
    checked = preflight(source, output)
    require(history_parent.is_absolute() and history_parent.is_dir(),
            "explicit owner historical parent mandatory")
    # These two source-only inventories never read any old media/gold.
    history = inventory_local_original_sources([history_parent])
    observed = {repo for repo, _ in history["_owner_only_repository_shas"]}
    require(canonical_repository_family(REPO) not in observed,
            "Dragula is a known locally consumed source family")
    roots = find_pre_h173_phases([history_parent])
    old = scan_legacy_manifest_sha_fields(roots)
    require(old["complete_historical_manifest_coverage_proven"] is False
            and old["training_dataset_admitted"] is False
            and old["old_image_media_opened"] is False,
            "partial historical SHA exclusion must not mean admission")
    checked.update(old=old, source_census=history)
    return checked


def make_private_research_prospects(source: Path, output: Path,
                                    history_parent: Path) -> dict:
    """Synchronous native browser QA; MUST remain private/research-only."""
    checked = _strict_local_source(source, output, history_parent)
    from playwright.sync_api import sync_playwright
    require(not checked["output"].exists(), "output must be create-only")
    require(output.name.lower().startswith("dragula-native-prospects-"),
            "use distinct private research output naming")
    result = {
        "status": "H242_OWNER_PRIVATE_ORIGINAL_DRAGULA_PROSPECTS_NOT_ADMITTED",
        "source_repository": REPO,
        "source_git_sha": COMMIT,
        "source_asset_sha256": dict(PINNED),
        "original_js_runtime_unmodified": True,
        "rendered_html_self_authored": True,
        "historical_local_sources_inspected":
            checked["source_census"]["source_checkouts_inspected"],
        "historical_pre_h173_manifests_screened":
            checked["old"]["manifest_files_scanned"],
        "known_old_png_sha_exclusions":
            checked["old"]["distinct_png_sha_candidates"],
        "known_old_rgba_sha_exclusions":
            checked["old"]["distinct_decoded_rgba_sha_candidates"],
        "full_historical_source_and_visual_history_proven": False,
        "transitive_license_and_media_rights_audited": False,
        "official_train_dev_dataset_admitted": False,
        "train_eligible": False,
        "dev_eligible": False,
        "ood_eligible": False,
        "train_optimizer_approved": False,
        "training_model_calls": 0,
        "optimizer_steps": 0,
        "cases": [],
    }
    output = checked["output"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        with sync_playwright() as playwright:
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            kwargs = {"headless": True}
            if edge.is_file():
                kwargs["executable_path"] = str(edge)
            elif not Path(playwright.chromium.executable_path).is_file():
                raise ValueError("H242 requires existing local Chromium/Edge")
            browser = playwright.chromium.launch(**kwargs)
            try:
                local_png: set[str] = set()
                local_rgba: set[str] = set()
                prior_png = checked["old"]["_owner_only_png_sha_candidates"]
                prior_rgba = checked["old"]["_owner_only_rgba_sha_candidates"]
                for i, case in enumerate(CASES):
                    context = browser.new_context(viewport={"width":960,"height":720},
                        device_scale_factor=1, service_workers="block",
                        locale="ar-SA" if case["lang"] == "ar" else "en-US")
                    require_browser_offline(context)
                    attempts: list[str] = []
                    def block(route):
                        attempts.append(route.request.url)
                        route.abort()
                    context.route("**/*", block)
                    try:
                        page = context.new_page()
                        page.set_content(fixture(case), wait_until="load")
                        page.add_script_tag(path=str(checked["asset"]))
                        page.evaluate("""() => window.dragula([
                          document.querySelector('#left'),
                          document.querySelector('#right')])""")
                        source_box = page.locator("#moving").bounding_box()
                        target_box = page.locator("#right").bounding_box()
                        require(source_box is not None and target_box is not None,
                                "visual target not visible")
                        start = [int(source_box["x"]+source_box["width"]/2),
                                 int(source_box["y"]+source_box["height"]/2)]
                        end = [int(target_box["x"]+target_box["width"]/2),
                               int(target_box["y"]+target_box["height"]/2)]
                        require(all(0 <= pt[0] < 960 and 0 <= pt[1] < 720
                                    for pt in (start,end)),
                                "drag outside screenshot bounds")
                        before = page.screenshot(type="png")
                        page.locator("#moving").drag_to(page.locator("#right"),
                                                      timeout=7000)
                        require(page.locator("#right > #moving").count()==1
                                and page.locator("#left > #moving").count()==0,
                                "original native drag oracle disagrees")
                        after = page.screenshot(type="png")
                        require(not attempts, "network was attempted")
                        checks = [(sha256(blob).hexdigest(), _rgba_sha(blob))
                                  for blob in (before,after)]
                        for png_sha, rgba_sha in checks:
                            require(png_sha not in prior_png and rgba_sha not in prior_rgba,
                                    "candidate collides with known historical SHA")
                            require(png_sha not in local_png and rgba_sha not in local_rgba,
                                    "candidate duplicate among new scenes")
                            local_png.add(png_sha)
                            local_rgba.add(rgba_sha)
                        for phase, blob in (("before",before),("after",after)):
                            (output/f"{case['case_id']}-{phase}.png").write_bytes(blob)
                        result["cases"].append({
                            "case_id": case["case_id"],
                            "language": case["lang"],
                            "instruction": case["task"],
                            "source_point": start,
                            "destination_point": end,
                            "preaction_png_sha256": checks[0][0],
                            "preaction_rgba_sha256": checks[0][1],
                            "postaction_png_sha256": checks[1][0],
                            "postaction_rgba_sha256": checks[1][1],
                            "native_drag_verified": True,
                            "network_requests": 0,
                            "training_eligible": False,
                            "dev_eligible": False,
                            "ood_eligible": False,
                        })
                    finally:
                        context.close()
            finally:
                browser.close()
        require(len(result["cases"]) == CASE_COUNT
                and len({c["case_id"] for c in result["cases"]})==CASE_COUNT
                and sum(c["language"] == "en" for c in result["cases"])==4
                and sum(c["language"] == "ar" for c in result["cases"])==4,
                "expected unique balanced 8 source-native scenes")
        (output/"PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json").write_text(
            json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return result
    except BaseException:
        (output/"INCOMPLETE_NEVER_ADMIT").write_text(
            "Not qualified for TRAIN/DEV/OOD. Never publish.\n",encoding="utf-8")
        raise


def main() -> None:
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--source-root",required=True,type=Path)
    arg.add_argument("--output",required=True,type=Path)
    arg.add_argument("--historical-parent",required=True,type=Path)
    args=arg.parse_args()
    try:
        result=make_private_research_prospects(
            args.source_root,args.output,args.historical_parent)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError):
        arg.exit(2,"H242_PRIVATE_SOURCE_PROSPECTS_NOT_QUALIFIED\\n")
    print(json.dumps({
        "status":result["status"],"cases":len(result["cases"]),
        "en":sum(c["language"]=="en" for c in result["cases"]),
        "ar":sum(c["language"]=="ar" for c in result["cases"]),
        "historical_png_exclusions":result["known_old_png_sha_exclusions"],
        "train":False,"dev":False,"ood":False,"model_calls":0,"optimizer_steps":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
