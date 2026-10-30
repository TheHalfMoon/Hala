"""H250: source-pinned original standalone MIT Todo app; PRIVATE RESEARCH ONLY."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from h174.split_firewall import canonical_repository_family
from h199.native_action_sft import parse_actions, source_task_prompt
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import require
from h238.historical_family_floor import known_public_source_family_floor
from h243.browser_offline import require_browser_offline
from h247.two_step_calendar_replay import visible_native_pixel, native_pixel_click
from h244.private_research_overlap import sha_ok

REPO = "serkanbyx/simple-to-do-list"
COMMIT = "009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff"
FILES = {
    "LICENSE": "a46c7ea4b472ec4498808ac6274eba9a4951eec301d87f1f27b460d252055caf",
    "index.html": "1ed6e5abe5051754ea09929ea63b22a7ef1c5070ce606762147ec623ec9461a7",
    "css/style.css": "45bac81118b50fcc20cdd25131238eda5f77de657168cc61899cb3ea7fb3b1bf",
    "js/app.js": "584d8503b3122ce6b8608074761fbd15a5407c533a8002b7d28ebd5b3646ef72",
}
STATUS = "H250_ORIGINAL_COMPLETE_TODO_APP_RESEARCH_NOT_ADMITTED"
STAGES = ("before_focus", "after_focus", "after_type", "after_add",
          "before_scroll", "after_scroll")
CASES = (
    {"id": "original-todo-en-01", "lang": "en", "text": "Review quarterly report"},
    {"id": "original-todo-ar-02", "lang": "ar", "text": "راجع التقرير الأسبوعي"},
)

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          check=True, capture_output=True, text=True).stdout.strip()

def preflight(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve()
    require((source / ".git").is_dir()
            and Path(_git(source, "rev-parse", "--show-toplevel")).resolve() == source
            and _git(source, "rev-parse", "HEAD") == COMMIT
            and _git(source, "status", "--porcelain", "--untracked-files=all") == "",
            "original app source checkout must be exact and unchanged")
    require(_git(source, "remote", "get-url", "origin").casefold() in (
        "https://github.com/serkanbyx/simple-to-do-list.git",
        "https://github.com/serkanbyx/simple-to-do-list"),
        "unapproved original source Git origin")
    for name, expected in FILES.items():
        f = source.joinpath(*name.split("/"))
        require(f.is_file() and not f.is_symlink()
                and f.resolve().is_relative_to(source)
                and hashlib.sha256(f.read_bytes()).hexdigest() == expected,
                "pinned original MIT source file changed")
    require((source / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n"),
            "separate original source MIT license not present")
    require(canonical_repository_family(REPO) not in known_public_source_family_floor(),
            "candidate in previously known PUBLIC source-family blocklist")
    require(output.name.startswith("h250-owner-private-research-")
            and not output.exists()
            and not output.is_relative_to(source)
            and not any((p / ".git").exists() for p in (output, *output.parents)),
            "owner output must be CREATE-ONLY outside Git or original source")

def action(kind: str, **kwargs) -> dict:
    return parse_actions(json.dumps({"actions": [{"kind":kind, **kwargs}]},
                                    ensure_ascii=False), training_target=True)

def verify_row(row: dict, spec: dict) -> None:
    require(type(row) is dict and spec in CASES
            and row.get("case_id") == spec["id"]
            and row.get("language") == spec["lang"]
            and row.get("native_add_verified") is True
            and row.get("native_reload_verified") is True
            and row.get("native_scroll_verified") is True
            and all(row.get(f) is False for f in ("train", "dev", "ood")),
            "native original app result incomplete or improperly promoted")
    images = row.get("frames")
    require(type(images) is dict and set(images) == set(STAGES), "missing frame")
    for field in ("png_sha256", "rgba_sha256"):
        values = [images[s].get(field) for s in STAGES]
        require(all(sha_ok(v) for v in values) and len(set(values)) == len(values),
                "missing or duplicated original browser observation")
    steps = row.get("steps")
    require(type(steps) is list and len(steps) == 4, "four one-action steps required")
    transitions = (("before_focus","after_focus"),
                   ("after_focus","after_type"),
                   ("after_type","after_add"),
                   ("before_scroll","after_scroll"))
    for i,step in enumerate(steps):
        require(type(step) is dict
                and step.get("source_frame") == transitions[i][0]
                and step.get("result_frame") == transitions[i][1]
                and step.get("research_only") is True
                and source_task_prompt(step["instruction"])
                and action(**step["native_action"]["actions"][0]) == step["native_action"],
                "single screenshot/native action horizon mismatch")
    require([s["native_action"]["actions"][0]["kind"] for s in steps]
            == ["click","type_text","click","scroll"]
            and steps[1]["native_action"]["actions"][0]["text"] == spec["text"],
            "wrong original app action ordering/typed text")

def replay(source: Path, output: Path) -> dict:
    preflight(source, output)
    from playwright.sync_api import sync_playwright
    source = source.resolve(strict=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": STATUS, "source_repository": REPO, "source_git_sha": COMMIT,
        "pinned_original_source_sha256": dict(FILES),
        "historical_source_history_complete": False,
        "historical_image_rgba_history_complete": False,
        "transitive_media_rights_complete": False,
        "train": False, "dev": False, "ood": False,
        "model_calls": 0, "optimizer_steps": 0, "cases": [],
    }
    try:
        with sync_playwright() as pw:
            edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
            opts = {"headless": True}
            if edge.is_file():
                opts["executable_path"] = str(edge)
            elif not Path(pw.chromium.executable_path).is_file():
                raise ValueError("existing Edge or Chromium needed")
            browser = pw.chromium.launch(**opts)
            try:
                for spec in CASES:
                    ctx = browser.new_context(viewport={"width":960,"height":720},
                        device_scale_factor=1, service_workers="block",
                        locale="ar-SA" if spec["lang"] == "ar" else "en-US")
                    require_browser_offline(ctx)
                    remote_attempts: list[str] = []
                    def block(route):
                        if route.request.url.startswith(("http://","https://","ws://","wss://")):
                            remote_attempts.append(route.request.url)
                            route.abort()
                        else:
                            route.continue_()
                    ctx.route("**/*", block)
                    try:
                        page = ctx.new_page()
                        page.goto((source / "index.html").as_uri(), wait_until="load")
                        require(page.title() == "Simple To-Do List", "real original app not loaded")
                        inp = page.locator("#todoInput")
                        add = page.get_by_role("button", name="Add task")
                        require(inp.count() == add.count() == 1
                                and inp.input_value() == "", "real app input absent")
                        photos: dict[str,bytes] = {}
                        steps: list[dict] = []
                        def photo(stage):
                            photos[stage] = page.screenshot(type="png")
                        def step(instr, act, before, after):
                            require(source_task_prompt(instr) != "", "missing visible task")
                            steps.append({"instruction":instr, "native_action":act,
                                "source_frame":before, "result_frame":after,
                                "research_only": True})
                        point = visible_native_pixel(inp)
                        photo("before_focus")
                        native_pixel_click(page, point)
                        require(page.evaluate("document.activeElement.id === 'todoInput'"),
                                "native original input focus absent")
                        photo("after_focus")
                        step("Focus the visible task input.", action("click",point=point),
                             "before_focus","after_focus")
                        text = spec["text"]
                        page.keyboard.insert_text(action("type_text",text=text)["actions"][0]["text"])
                        require(inp.input_value() == text, "original Unicode typing failed")
                        photo("after_type")
                        step("Type the requested task: " + text,
                             action("type_text",text=text), "after_focus","after_type")
                        add_point = visible_native_pixel(add)
                        native_pixel_click(page, add_point)
                        require(page.locator("#todoList .todo-text").filter(has_text=text).count()==1,
                                "real original app did not add text")
                        photo("after_add")
                        step("Click Add to save the visible task.",
                             action("click",point=add_point), "after_type","after_add")
                        page.reload(wait_until="load")
                        require(page.locator("#todoList .todo-text").filter(has_text=text).count()==1,
                                "actual app localStorage persistence not verified")
                        for i in range(20):
                            page.locator("#todoInput").fill(f"Offline native setup task {i+1:02}")
                            add.click()
                        require(page.locator("#todoList .todo-text").count()==21,
                                "original app did not populate scroll content")
                        page.evaluate("window.scrollTo(0,0)")
                        photo("before_scroll")
                        scroll_pt = [480,600]
                        act = action("scroll",point=scroll_pt,dy=480)
                        before = page.evaluate("window.scrollY")
                        page.mouse.move(*scroll_pt)
                        page.mouse.wheel(0,480)
                        page.wait_for_timeout(350)
                        after = page.evaluate("window.scrollY")
                        require(after > before, "native mouse wheel failed to scroll")
                        photo("after_scroll")
                        step("Scroll down the original task list.",act,
                             "before_scroll","after_scroll")
                        frames = {name:{
                            "png_sha256":hashlib.sha256(photos[name]).hexdigest(),
                            "rgba_sha256":_rgba_sha(photos[name])} for name in STAGES}
                        row = {"case_id":spec["id"], "language":spec["lang"],
                               "native_add_verified": True,
                               "native_reload_verified": True,
                               "native_scroll_verified": True,
                               "external_requests_blocked":len(remote_attempts),
                               "train":False,"dev":False,"ood":False,
                               "frames":frames,"steps":steps}
                        verify_row(row,spec)
                        for stage in STAGES:
                            (output / (spec["id"] + "-" + stage + ".png")).write_bytes(photos[stage])
                        report["cases"].append(row)
                    finally:
                        ctx.close()
            finally:
                browser.close()
        require(len(report["cases"]) == 2 and
                {r["language"] for r in report["cases"]} == {"en","ar"},
                "one native English and one native Arabic case are required")
        (output / "PRIVATE_RESEARCH_NOT_ADMITTED.json").write_text(
            json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return report
    except BaseException:
        (output / "INCOMPLETE_NEVER_ADMIT").write_text(
            "Research-only app replay failed. Never admit images.\n",encoding="utf-8")
        raise

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    data=replay(args.source_root,args.output)
    print(json.dumps({"status":data["status"],
        "source_app_count":1,
        "bilingual_native_cases":len(data["cases"]),
        "one_action_steps":4*len(data["cases"]),
        "new_private_png":6*len(data["cases"]),
        "train":False,"dev":False,"ood":False,
        "model_calls":0,"optimizer_steps":0},sort_keys=True))
