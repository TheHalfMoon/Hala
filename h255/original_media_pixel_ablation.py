"""H255: paired in-memory pixel ablation for original-app visual-rights research.

Never stores screenshots or promotes any candidate to TRAIN/DEV/OOD.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image
from h243.browser_offline import require_browser_offline
from h253.original_app_asset_rights import audit
from h254.original_media_runtime_evidence import git, within_root


def changed_pixels(first: bytes, second: bytes, box: dict) -> dict:
    """Count RGB differences inside/outside a strict screenshot-coordinate rectangle."""
    with Image.open(io.BytesIO(first)) as a, Image.open(io.BytesIO(second)) as b:
        if a.size != b.size or a.size != (960, 720):
            raise ValueError("Expected identical 960x720 screenshots")
        if a.format != "PNG" or b.format != "PNG":
            raise ValueError("PNG screenshots required")
        x, y, w, h = (box.get(k) for k in ("x", "y", "width", "height"))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("Finite rectangle required")
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 960 or y + h > 720:
            raise ValueError("Visible in-bounds rectangle required")
        left, top = math.floor(x), math.floor(y)
        right, bottom = math.ceil(x + w), math.ceil(y + h)
        inside = outside = 0
        for idx, (p, q) in enumerate(zip(a.convert("RGB").get_flattened_data(),
                                         b.convert("RGB").get_flattened_data())):
            if p == q:
                continue
            py, px = divmod(idx, 960)
            if left <= px < right and top <= py < bottom:
                inside += 1
            else:
                outside += 1
    return {"inside_changed_pixels": inside, "outside_changed_pixels": outside}


def classify(control: dict, ablation: dict, restored: dict) -> str:
    for item in (control, ablation, restored):
        if set(item) != {"inside_changed_pixels", "outside_changed_pixels"}:
            raise ValueError("Malformed pixel counts")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in item.values()):
            raise ValueError("Invalid pixel counts")
    if any(control.values()) or any(restored.values()):
        return "INDETERMINATE_DYNAMIC_PAGE"
    if ablation["inside_changed_pixels"] == 0 and ablation["outside_changed_pixels"] == 0:
        return "NO_VISIBLE_PIXEL_EFFECT"
    if ablation["outside_changed_pixels"] > 0:
        return "VISIBLE_DEPENDENCY_OUTSIDE_IMAGE_RECT"
    return "VISIBLE_PIXEL_DEPENDENCY_OBSERVED"


def observe(root: Path, sha: str, entry: str, selector: str) -> dict:
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Full immutable Git SHA required")
    if not selector or len(selector) > 200:
        raise ValueError("Bounded selector required")
    static = audit(root, sha, [entry])
    if static["status"] != "INCOMPLETE_NO_ADMISSION" or not static["pinned_source_verified"]:
        raise ValueError("Original source preflight refused")
    if git(root, "status", "--porcelain"):
        raise ValueError("Original checkout must be clean")
    from playwright.sync_api import sync_playwright
    blocked, responses, escapes, websockets = [], [], [], []
    with sync_playwright() as pw:
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        if not edge.is_file():
            raise RuntimeError("Installed Edge required")
        browser = pw.chromium.launch(executable_path=str(edge), headless=True)
        try:
            context = browser.new_context(viewport={"width": 960, "height": 720},
                                          device_scale_factor=1,
                                          reduced_motion="reduce", service_workers="block")
            require_browser_offline(context)
            context.on("response", lambda r: responses.append(1)
                       if urlsplit(r.url).scheme in {"http", "https"} else None)
            context.on("websocket", lambda _: websockets.append(1))

            def guard(route):
                url = route.request.url
                scheme = urlsplit(url).scheme
                if scheme in {"http", "https", "ws", "wss"}:
                    blocked.append(1)
                    route.abort()
                elif scheme == "file" and within_root(url, root):
                    route.continue_()
                else:
                    escapes.append(1)
                    route.abort()

            context.route("**/*", guard)
            page = context.new_page()
            page.goto((root / entry).as_uri(), wait_until="load", timeout=20000)
            page.wait_for_timeout(250)
            locator = page.locator(selector)
            if locator.count() != 1:
                raise ValueError("Exactly one image target required")
            target = locator.first
            if target.evaluate("(el) => el.tagName.toLowerCase() !== 'img' || !el.complete || el.naturalWidth <= 0"):
                raise ValueError("Target must be a loaded original image")
            box = target.bounding_box()
            if box is None:
                raise ValueError("Target must be visible")
            screenshot = lambda: page.screenshot(type="png", animations="disabled", timeout=10000)
            before = screenshot()
            control = changed_pixels(before, screenshot(), box)
            original_visibility = target.evaluate("(el) => el.style.visibility")
            target.evaluate("(el) => { el.style.visibility = 'hidden'; }")
            hidden = screenshot()
            target.evaluate("(el, value) => { el.style.visibility = value; }", original_visibility)
            restored = screenshot()
            ablation = changed_pixels(before, hidden, box)
            restoration = changed_pixels(before, restored, box)
            context.close()
        finally:
            browser.close()
    if responses or escapes or websockets:
        raise ValueError("Forbidden browser network/file activity")
    result = classify(control, ablation, restoration)
    return {
        "status": "INCOMPLETE_NO_ADMISSION",
        "pixel_ablation_result": result,
        "baseline_control": control,
        "image_hidden_comparison": ablation,
        "restoration_control": restoration,
        "blocked_external_requests_observed": len(blocked),
        "external_responses_observed": len(responses),
        "file_escapes_observed": len(escapes),
        "websockets_observed": len(websockets),
        "image_authorship_or_rights_proven": False,
        "historic_source_visual_completeness_proven": False,
        "train_dev_ood_admitted": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--entry", default="index.html")
    parser.add_argument("--selector", default="img")
    args = parser.parse_args()
    try:
        result = observe(args.repo, args.sha, args.entry, args.selector)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "INCOMPLETE_NO_ADMISSION", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
