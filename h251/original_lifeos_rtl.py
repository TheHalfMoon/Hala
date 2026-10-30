"""H251: genuine original complete Arabic RTL + English app source-native replay.

Hala public code, source-only. Private original screenshot/action data quarantined,
zero model calls, zero optimizer, never official TRAIN/DEV/OOD.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import secrets

from h174.split_firewall import canonical_repository_family
from h199.native_action_sft import parse_actions, source_task_prompt
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import require
from h238.historical_family_floor import known_public_source_family_floor
from h243.browser_offline import require_browser_offline
from h247.two_step_calendar_replay import visible_native_pixel, native_pixel_click
from h250.native_original_todo_app import _git
from h244.private_research_overlap import sha_ok

REPO = "3mmar404/life-os-free"
COMMIT = "b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c"
FILES = {
    "LICENSE": "2d3c3327311bac28e11ef3650dfef64eea9a0d786faa309a68dc9f326006d548",
    "index.html": "2eee8fe6f7289965fee5650c6e071634cfb45ee977ab1b94015cce2d1e91d419",
    "index-en.html": "c8963b01d702989fedf61cf70ce83f5c96c579ae7cadd04adb240b47bd864b1d",
    "css/style.css": "df74a9a93236fcd1631ef485b4a033485154bfe30263aaed8084334179f2d7b6",
    "js/core.js": "603488bc9a15d5c2d28bffc85156971d00861524b02594d18bf094d1a6e1452a",
    "js/modules/bookmarks.js": "0999c82c7f12a3230f0b25500b135c4e2d8099fcc9c005144348fb5efa711f92",
    "js/modules/bookmarks-en.js": "625259f6bc3a417888eb2e82e82861715b179350e8f0f4a3b8464ee9ffd03f3c",
    "js/modules/documentation.js": "f94b4f6385abe3e6f14bf777ebaa97639de504f1af0a77c75e4203ae6702824f",
    "js/modules/documentation-en.js": "a61edeff6350d17bbc8fbd5cc30050b8cd5e55235aa8e3513f1cece86f82442d",
    "js/ui.js": "ae842d5431f52c0672fb8dd8ef53545c60fa326c46a58fb602e1821c9a59695f",
}
STATUS = "H251_ORIGINAL_ARABIC_RTL_APP_PRIVATE_RESEARCH_NOT_ADMITTED"
STAGES = ("before_nav", "after_nav", "after_focus", "after_type",
          "after_docs", "after_scroll")
TRANSITIONS = tuple(zip(STAGES, STAGES[1:]))
CASES = (
    {"id": "lifeos-ar-rtl-01", "lang": "ar", "page": "index.html",
     "task": "ابحث عن دليل المشاريع في الروابط",
     "title": "دليل المشاريع", "query": "دليل"},
    {"id": "lifeos-en-ltr-02", "lang": "en", "page": "index-en.html",
     "task": "Find the project guide in saved bookmarks",
     "title": "Project guide", "query": "Project"},
)
KINDS = ("click", "click", "type_text", "click", "scroll")


def source_preflight(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve()
    require((source / ".git").is_dir()
            and Path(_git(source, "rev-parse", "--show-toplevel")).resolve() == source
            and _git(source, "rev-parse", "HEAD") == COMMIT
            and _git(source, "status", "--porcelain", "--untracked-files=all") == "",
            "original LifeOS checkout is not pristine and exact")
    require(_git(source, "remote", "get-url", "origin").casefold() in (
        "https://github.com/3mmar404/life-os-free.git",
        "https://github.com/3mmar404/life-os-free"),
        "source origin is not the original repository")
    for name, digest in FILES.items():
        f = source.joinpath(*name.split("/"))
        require(f.is_file() and not f.is_symlink()
                and f.resolve().is_relative_to(source)
                and sha256(f.read_bytes()).hexdigest() == digest,
                "exact original MIT source file changed")
    require((source / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n"),
            "real source MIT license not pinned")
    require(canonical_repository_family(REPO)
            not in known_public_source_family_floor(),
            "candidate already known in historical PUBLIC sources")
    require(output.name.startswith("h251-owner-private-research-")
            and not output.exists()
            and not output.is_relative_to(source)
            and not any((p / ".git").exists() for p in (output, *output.parents)),
            "research output must be CREATE-ONLY and outside ALL Git checkouts")


def action(kind: str, **attrs) -> dict:
    return parse_actions(json.dumps(
        {"actions":[{"kind":kind, **attrs}]}, ensure_ascii=False
    ), training_target=True)


def validate_result(row: dict, spec: dict) -> None:
    require(type(row) is dict and spec in CASES
            and row.get("case_id")==spec["id"]
            and row.get("language")==spec["lang"]
            and row.get("actual_html_lang")==spec["lang"]
            and row.get("actual_html_dir")==("rtl" if spec["lang"]=="ar" else "ltr")
            and row.get("original_bookmark_saved") is True
            and row.get("native_search_visible") is True
            and row.get("native_documentation_visible") is True
            and row.get("native_scroll_moved") is True
            and row.get("model_calls")==row.get("optimizer_steps")==0,
            "original complete Arabic/LTR app state not witnessed")
    require(all(row.get(k) is False for k in ("train","dev","ood")),
            "research must never be admitted to official splits")
    frames=row.get("frames")
    require(type(frames) is dict and set(frames)==set(STAGES),
            "not all original screenshot phases exist")
    for field in ("png_sha256","rgba_sha256"):
        hashes=[frames[s].get(field) for s in STAGES]
        require(all(sha_ok(x) for x in hashes)
                and len(set(hashes))==len(STAGES),
                "duplicated or corrupt original screenshots")
    steps=row.get("steps")
    require(type(steps) is list and len(steps)==len(TRANSITIONS),
            "five source-native one-action transitions mandatory")
    for i,step in enumerate(steps):
        require(type(step) is dict
                and step.get("source_frame")==TRANSITIONS[i][0]
                and step.get("result_frame")==TRANSITIONS[i][1]
                and step.get("research_only") is True
                and type(step.get("instruction")) is str
                and source_task_prompt(step["instruction"])
                and type(step.get("native_action")) is dict
                and type(step["native_action"].get("actions")) is list
                and len(step["native_action"]["actions"]) == 1,
                "invalid visible single-action step")
        native=step["native_action"]["actions"][0]
        require(action(**native)==step["native_action"]
                and native["kind"] == KINDS[i],
                "single-action type/order disagrees with actual original transitions")
    require(steps[2]["native_action"]["actions"][0]["text"] == spec["query"],
            "original visible query differs from typed label")


def replay(source: Path, output: Path) -> dict:
    source_preflight(source, output)
    source=source.resolve(strict=True)
    output=output.resolve()
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True,exist_ok=False)
    data = {
        "status": STATUS, "source_repository": REPO, "source_git_sha": COMMIT,
        "original_mit_file_sha256":dict(FILES),
        "old_private_history_completeness_proven":False,
        "old_decoded_rgba_completeness_proven":False,
        "transitive_visual_media_rights_complete":False,
        "train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0, "cases":[],
    }
    try:
        with sync_playwright() as pw:
            edge=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            launch={"headless":True}
            if edge.is_file():
                launch["executable_path"]=str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("offline Edge or Chromium executable unavailable")
            browser=pw.chromium.launch(**launch)
            try:
                for spec in CASES:
                    ctx=browser.new_context(
                        viewport={"width":960,"height":720},
                        device_scale_factor=1, locale="ar-SA" if spec["lang"]=="ar" else "en-US",
                        service_workers="block")
                    require_browser_offline(ctx)
                    external_attempts=[]
                    def block_external(route):
                        if route.request.url.startswith(("http://","https://","ws://","wss://")):
                            external_attempts.append(route.request.url)
                            route.abort()
                        else:
                            route.continue_()
                    ctx.route("**/*",block_external)
                    try:
                        page=ctx.new_page()
                        page.goto((source/spec["page"]).as_uri(),wait_until="load")
                        page.wait_for_selector("#new-master-password",state="visible")
                        lang_and_dir=page.evaluate("({lang:document.documentElement.lang,dir:document.documentElement.dir})")
                        require(lang_and_dir=={"lang":spec["lang"],"dir":"rtl" if spec["lang"]=="ar" else "ltr"},
                                "original source is not native language/RTL")
                        # Setup of an isolated fictional encrypted vault is NOT a Hala
                        # supervised action and never emits its random password.
                        secret=secrets.token_urlsafe(16)
                        page.locator("#new-master-password").fill(secret)
                        page.locator("#confirm-master-password").fill(secret)
                        page.locator("#new-master-password").locator("xpath=ancestor::form").locator(
                            'button[type="submit"]').click()
                        page.wait_for_selector("#dashboard.active",state="visible")
                        # Seed one fictitious bookmark using ORIGINAL interface,
                        # not model DOM injection or synthetic wrapper markup.
                        nav=page.locator(".nav-tab[data-module='bookmarks']")
                        nav.click()
                        page.wait_for_selector("#bookmarks-search",state="visible")
                        page.locator("#bookmarks .search-filters .btn-success").click()
                        page.wait_for_selector("#modal.active",state="visible")
                        if spec["lang"]=="ar":
                            page.wait_for_selector("#bookmark-form",state="attached")
                            page.locator("#bookmark-form #title").fill(spec["title"])
                            page.locator("#bookmark-form #url").fill("https://example.invalid/research-only")
                            page.locator("#bookmark-form button[type='submit']").click()
                        else:
                            # Original EN module uses another form API, not the
                            # Arabic module's #bookmark-form. Never modify source.
                            page.wait_for_selector("#bookmark-title",state="visible")
                            page.locator("#bookmark-title").fill(spec["title"])
                            page.locator("#bookmark-url").fill("https://example.invalid/research-only")
                            page.locator("#modal .modal-body button.btn-success").click()
                        page.wait_for_selector("#bookmarks .card-title",state="visible")
                        require(page.locator("#bookmarks .card-title").filter(
                            has_text=spec["title"]).count()==1,
                            "original UI did not save the fictional bookmark")
                        page.locator(".nav-tab[data-module='dashboard']").click()
                        page.wait_for_selector("#dashboard.active",state="visible")
                        frames:dict[str,bytes]={}
                        steps:list[dict]=[]
                        def picture(name):
                            frames[name]=page.screenshot(type="png")
                        def add_step(prompt, act, before, after):
                            steps.append({"instruction":prompt,
                                "native_action":act,
                                "source_frame":before,"result_frame":after,
                                "research_only":True})
                        picture("before_nav")
                        nav=page.locator(".nav-tab[data-module='bookmarks']")
                        pixel=visible_native_pixel(nav)
                        native_pixel_click(page,pixel)
                        page.wait_for_selector("#bookmarks.active",state="visible")
                        picture("after_nav")
                        add_step(spec["task"],action("click",point=pixel),
                                 "before_nav","after_nav")
                        search=page.locator("#bookmarks-search")
                        pixel=visible_native_pixel(search)
                        native_pixel_click(page,pixel)
                        require(page.evaluate("document.activeElement.id === 'bookmarks-search'"),
                                "original native focus not established")
                        picture("after_focus")
                        add_step("Focus the visible bookmark search field.",
                                 action("click",point=pixel),"after_nav","after_focus")
                        page.keyboard.insert_text(spec["query"])
                        page.wait_for_timeout(500)
                        require(search.input_value()==spec["query"]
                                and page.locator("#bookmarks .card-title").filter(
                                has_text=spec["title"]).count()==1
                                and page.evaluate("LifeOS.bookmarks.searchQuery" if spec["lang"]=="ar" else "LifeOS.bookmarksEn.searchQuery") == spec["query"],
                                "original app Arabic/English search state not updated")
                        picture("after_type")
                        add_step(spec["task"],action("type_text",text=spec["query"]),
                                 "after_focus","after_type")
                        docs=page.locator(".nav-tab[data-module='documentation']")
                        pixel=visible_native_pixel(docs)
                        native_pixel_click(page,pixel)
                        page.wait_for_selector("#documentation.active",state="visible")
                        picture("after_docs")
                        add_step("Open documentation from the navigation.",
                                 action("click",point=pixel),"after_type","after_docs")
                        # Single real wheel action must change a measured scrollable
                        # viewport, not just produce a screenshot without motion.
                        before=page.evaluate("window.scrollY")
                        wheelpoint=[480,600]
                        wheel=action("scroll",point=wheelpoint,dy=480)
                        page.mouse.move(*wheelpoint)
                        page.mouse.wheel(0,480)
                        page.wait_for_timeout(400)
                        after=page.evaluate("window.scrollY")
                        require(after>before,
                                "native documentation scroll did not move viewport")
                        picture("after_scroll")
                        add_step("Scroll down the original documentation page.",
                                 wheel,"after_docs","after_scroll")
                        hashes={name:{
                            "png_sha256":sha256(frames[name]).hexdigest(),
                            "rgba_sha256":_rgba_sha(frames[name])
                        } for name in STAGES}
                        row={
                            "case_id":spec["id"],"language":spec["lang"],
                            "actual_html_lang":lang_and_dir["lang"],
                            "actual_html_dir":lang_and_dir["dir"],
                            "original_bookmark_saved":True,
                            "native_search_visible":True,
                            "native_documentation_visible":True,
                            "native_scroll_moved":True,
                            "external_requests_blocked":len(external_attempts),
                            "model_calls":0,"optimizer_steps":0,
                            "train":False,"dev":False,"ood":False,
                            "steps":steps,"frames":hashes,
                        }
                        validate_result(row,spec)
                        for stage in STAGES:
                            (output/f"{spec['id']}-{stage}.png").write_bytes(frames[stage])
                        data["cases"].append(row)
                    finally:
                        ctx.close()
            finally:
                browser.close()
        require(len(data["cases"])==len(CASES)
                and {r["actual_html_dir"] for r in data["cases"]}=={"rtl","ltr"},
                "both genuine original app language directions required")
        (output/"PRIVATE_RESEARCH_NOT_ADMITTED.json").write_text(
            json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
            encoding="utf-8")
        return data
    except BaseException:
        (output/"INCOMPLETE_NEVER_ADMIT").write_text(
            "H251 private original app replay failed; no TRAIN/DEV/OOD admissibility.\n",
            encoding="utf-8")
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=replay(args.source_root,args.output)
    print(json.dumps({
        "status":result["status"],"independent_original_app_families":1,
        "genuine_ui_languages":len(result["cases"]),
        "native_one_action_steps":len(CASES)*len(KINDS),
        "new_owner_private_png":len(CASES)*len(STAGES),
        "train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0,
    },sort_keys=True))
