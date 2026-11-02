"""H358 from-scratch bilingual original multi-step GUI candidate factory.

Only NEW source-family review-candidates, not an official TRAIN corpus,
an independent OOD score or an adapted Qwen model. Source browser DOM
is used ONLY by the synthetic label oracle, never as model input.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from h199.native_action_sft import parse_actions, source_task_prompt

SOURCE = Path(__file__).with_name("noterelay.html")
FAMILY = "noterelay-original-multistep"
ROLE = "ORIGINAL_MULTISTEP_TRAIN_DEV_REVIEW_CANDIDATE_NOT_ADMITTED"
SCHEMA = "hala.h358.noterelay.original-candidate.v1"
CASES = 32
STEPS = ("open", "focus", "type", "save")
COUNT = CASES * len(STEPS)


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H358_UNADMITTED_ORIGINAL_SOURCE_CANDIDATE_REFUSED")


def scenes():
    result = []
    for index in range(CASES):
        # Pair each Arabic and English task with the SAME invented code,
        # workspace and theme. The former index-based code first letter
        # and workspace parity made language directly predictable from
        # nonlinguistic context before reading any UI text.
        paired_case = index // 2
        code = chr(ord("A") + paired_case) + str(1000 + paired_case * 73)
        require(len(code) == 5 and code[1:].isdigit())
        language = "ar" if index % 2 else "en"
        task = (("أنشئ ملاحظة بالرمز " + code + " ثم احفظها")
                if language == "ar" else
                ("Create a note with code " + code + " and save it"))
        result.append({
            "case_id": "h358-original-%03d" % index,
            "family": FAMILY,
            "language": language, "task": task,
            "code": code,
            "theme": "dark" if (index // 2) % 2 else "light",
            "workspace": paired_case + 1,
            "split": ROLE,
        })
    require(len(result) == CASES and
            len({item["case_id"] for item in result}) == CASES
            and len({item["code"] for item in result}) == CASES // 2
            and all(
                (result[2*j]["code"], result[2*j]["workspace"],
                 result[2*j]["theme"]) ==
                (result[2*j+1]["code"], result[2*j+1]["workspace"],
                 result[2*j+1]["theme"])
                for j in range(CASES // 2)
            )
            and {item["language"] for item in result} == {"en", "ar"}
            and {item["theme"] for item in result} == {"dark", "light"})
    return result


def _click_at(page, selector: str) -> list[int]:
    rect = page.locator(selector).bounding_box()
    require(rect is not None and rect["width"] >= 30
            and rect["height"] >= 25)
    point = [int(round(rect["x"] + rect["width"] / 2)),
             int(round(rect["y"] + rect["height"] / 2))]
    require(0 <= point[0] < 960 and 0 <= point[1] < 720)
    return point


def _action(kind: str, point: list[int] | None = None,
            code: str | None = None) -> str:
    if kind == "click":
        action = {"kind": kind, "point": point}
    else:
        require(kind == "type_text" and type(code) is str)
        action = {"kind": kind, "text": code}
    native = json.dumps({"actions": [action]}, separators=(",", ":"),
                        ensure_ascii=False, sort_keys=True)
    parse_actions(native, training_target=True)
    return native


def capture_candidate(root: Path, *, browser_kind: str = "chromium"):
    """Generate only ephemeral labeled original-source QA under caller-owned root.

    Original browser action oracle references DOM; screenshots + task alone
    are model-visible in any future separately authorized TRAIN run.
    """
    require(isinstance(root, Path) and not root.exists()
            and browser_kind in ("chromium", "msedge")
            and not root.resolve().is_relative_to(SOURCE.parent.parent.resolve()))
    original = SOURCE.read_bytes()
    require(len(original) < 25000 and
            all(term not in original for term in
                (b"http://", b"https://", b"fetch(", b"<script src=")))
    from playwright.sync_api import sync_playwright

    root.mkdir(parents=True, exist_ok=False)
    images = root / "images"
    images.mkdir()
    sha = sha256(original).hexdigest()
    records = []
    stable_initial = 0
    state_passes = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            **({"channel": "msedge"} if browser_kind == "msedge" else {}))
        try:
            page = browser.new_page(viewport={"width": 960, "height": 720},
                                    device_scale_factor=1)
            page.goto(SOURCE.resolve().as_uri(), wait_until="load")
            for scene in scenes():
                page.evaluate("s=>window.configureOriginalScene(s)", {
                    "code": scene["code"], "language": scene["language"],
                    "theme": scene["theme"], "workspace": scene["workspace"],
                })
                require(page.evaluate("window.getOriginalState().stage") == "start")
                for position, step in enumerate(STEPS):
                    state = page.evaluate("window.getOriginalState()")
                    expected_state = ("start", "open", "focused", "typed")[position]
                    require(state["stage"] == expected_state)
                    page.mouse.move(0, 0)
                    image = page.screenshot(animations="disabled")
                    if position == 0:
                        require(image == page.screenshot(animations="disabled"))
                        stable_initial += 1
                    if step == "open":
                        point = _click_at(page, "#open")
                        native = _action("click", point)
                        page.mouse.click(*point)
                    elif step == "focus":
                        point = _click_at(page, "#code")
                        native = _action("click", point)
                        page.mouse.click(*point)
                        require(page.evaluate("window.getOriginalState().focused"))
                    elif step == "type":
                        native = _action("type_text", code=scene["code"])
                        require(page.evaluate("window.getOriginalState().focused"))
                        page.keyboard.type(scene["code"])
                        require(page.locator("#code").input_value() == scene["code"])
                    else:
                        point = _click_at(page, "#save")
                        native = _action("click", point)
                        page.mouse.click(*point)
                        final = page.evaluate("window.getOriginalState()")
                        require(final["stage"] == "saved" and
                                final["committed"] == scene["code"])
                    file = images / (scene["case_id"] + "-" + step + ".png")
                    file.write_bytes(image)
                    records.append({
                        "id": scene["case_id"] + "-" + step,
                        "family": FAMILY,
                        "language": scene["language"],
                        "split": ROLE,
                        "phase": step,
                        "prompt": source_task_prompt(scene["task"]),
                        "completion": native,
                        "image_path": "images/" + file.name,
                        "image_sha256": sha256(image).hexdigest(),
                        "original_source_sha256": sha,
                    })
                    state_passes += 1
        finally:
            browser.close()
    require(len(records) == COUNT and state_passes == COUNT
            and stable_initial == CASES
            and len({r["id"] for r in records}) == COUNT
            and len({r["image_sha256"] for r in records}) == COUNT)
    (root / "candidate-rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                for r in records), encoding="utf-8")
    manifest = {
        "schema": SCHEMA, "role": ROLE, "family": FAMILY,
        "original_source_sha256": sha,
        "case_count": CASES, "steps_per_case": len(STEPS),
        "rows": COUNT, "original_native_state_successes": state_passes,
        "stable_initial_screenshots": stable_initial,
        "language_cases": {"en": CASES // 2, "ar": CASES // 2},
        "has_source_disjoint_dev": False,
        "independent_rights_approved": False,
        "official_train_admitted": False,
        "fresh_blind_ood": False,
        "actual_model_calls": 0, "gpu_calls": 0,
        "can_train": False, "can_release": False,
    }
    (root / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    raise SystemExit("H358 is an inert source candidate. Use capture_candidate with a private temporary directory.")
