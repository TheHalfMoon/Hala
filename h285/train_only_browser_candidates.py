"""H285: TRAIN-CANDIDATE ONLY synthetic, self-authored bilingual browser data.

No images, model outputs or gold coordinates ever stored in GitHub/Kaggle
output here. A trusted future private trainer may consume samples in
memory ONLY after all 24 actual browser actions and visual hash checks
pass. This is NOT official TRAIN admission and MUST NOT be used as DEV/OOD
or evidence that an adapted Qwen model improved.
"""
from __future__ import annotations

from hashlib import sha1, sha256
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h264.screenshot_locked_research import _observed_png

SOURCE_BLOB = "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac"
IDS = ("S-401", "S-415", "S-428", "S-439",
       "S-452", "S-467", "S-480", "S-493")
ROLE = "TRAIN_CANDIDATE_ONLY_NOT_OFFICIALLY_ADMITTED"
SCENES = tuple(
    (f"candidate_train_{i:02d}",
     "en" if (i // 3) % 2 == 0 else "ar",
     i % 3,
     "light" if (i // 6) % 2 == 0 else "dark",
     "standard" if (i // 12) % 2 == 0 else "compact",
     IDS[(i * 5 + i // 3) % 8])
    for i in range(24)
)


def need(value: bool, reason: str) -> None:
    if not value:
        raise ValueError("H285 original TRAIN candidate refused: " + reason)


def checked_source(source: Path) -> tuple[Path, str]:
    need(isinstance(source, Path), "explicit source Path required")
    try:
        p = source.absolute()
        need(p.name == "shiftdesk.html" and p.parent.name == "h285"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p),
             "original authored source path or link invalid")
        contents = p.read_bytes()
        blob = b"blob " + str(len(contents)).encode("ascii") + b"\x00" + contents
        need(compare_digest(sha1(blob).hexdigest(), SOURCE_BLOB),
             "pinned original GUI byte history drift")
        return p, p.as_uri()
    except (OSError, RuntimeError):
        raise ValueError("H285 original source unavailable") from None


def exact_task(scene: tuple) -> str:
    need(scene in SCENES and type(scene) is tuple,
         "unregistered candidate TRAIN scene")
    target = scene[-1]
    return ("أنجز بطاقة العمل الخيالية " + target
            if scene[1] == "ar" else
            "Complete fictional queue item " + target)


def inspect_all_eight(page: Any) -> dict[str, bool]:
    """Original-source evaluator/trainer only, never a policy input."""
    rows = page.evaluate("() => window.shiftDeskState()")
    need(type(rows) is list and len(rows) == 8, "eight states required")
    result = {}
    for row in rows:
        need(type(row) is dict and set(row) == {"id", "done"}
             and type(row["id"]) is str and row["id"] in IDS
             and row["id"] not in result and type(row["done"]) is bool,
             "invalid or duplicate source-owned identity")
        result[row["id"]] = row["done"]
    need(set(result) == set(IDS), "missing original source identity")
    return result


def _open_scene(browser: Any, source: Path, url: str, scene: tuple):
    context = new_offline_context(browser)
    try:
        def block_external(route: Any) -> None:
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        context.route("**/*", block_external)
        page = context.new_page()
        page.goto(url, wait_until="load")
        _, lang, layout, theme, density, _ = scene
        page.evaluate(
            "params => window.configureShiftDesk(params)",
            {"lang": lang, "layout": layout, "theme": theme,
             "density": density})
        need(page.url == url and page.viewport_size == {
            "width": WIDTH, "height": HEIGHT},
            "source-only browser URL/viewport drift")
        need(page.locator("article.card").count() == 8,
             "missing original TRAIN source cards")
        need(page.locator("html").get_attribute("dir") ==
             ("rtl" if lang == "ar" else "ltr"),
             "bilingual original GUI rendering mismatch")
        yield context, page
    finally:
        context.close()


def _candidate_one(browser: Any, source: Path, url: str, scene: tuple):
    from PIL import Image

    for _, page in _open_scene(browser, source, url, scene):
        before = inspect_all_eight(page)
        need(not any(before.values()), "TRAIN source must reset each scene")
        target = scene[-1]
        button = page.locator(
            'article.card[data-item-id="' + target + '"] button.complete')
        need(button.count() == 1 and button.is_visible(),
             "TRAIN reference target hidden or ambiguous")
        rect = button.bounding_box()
        need(rect is not None and rect["width"] >= 30
             and rect["height"] >= 24, "target button geometry invalid")
        point = [
            int(rect["x"] + rect["width"] / 2),
            int(rect["y"] + rect["height"] / 2),
        ]
        need(0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT,
             "TRAIN target outside model screenshot")
        png = _observed_png(page)
        with Image.open(BytesIO(png)) as im:
            need(im.format == "PNG" and im.size == (WIDTH, HEIGHT),
                 "TRAIN original full screenshot invalid")
            image = im.convert("RGBA")
            pixels = image.tobytes()
        visual_digest = sha256(pixels).hexdigest()
        byte_digest = sha256(png).hexdigest()
        task = exact_task(scene)
        completion = (
            '{"actions":[{"kind":"click","point":['
            + str(point[0]) + "," + str(point[1]) + "]}]}"
        )
        parse_actions(completion, training_target=True)
        # Image + task alone are policy inputs. The reference point was
        # extracted here ONLY because this is a self-authored TRAIN fixture.
        need(page.url == url and
             sha256(_observed_png(page)).hexdigest() == byte_digest,
             "TRAIN screenshot changed between oracle extraction and action")
        outcome = execute_one(page, completion)
        after = inspect_all_eight(page)
        expected = {key: (key == target) for key in IDS}
        need(outcome == {"kind": "click", "executed": True}
             and after == expected,
             "real native supervised target failed original source task")
        count = page.locator("#counter").inner_text()
        need(count in ("Processed: 1", "تم الإنجاز: 1"),
             "TRAIN source state count incorrect")
        return {
            "scene": scene[0], "lang": scene[1],
            "png": png, "png_sha256": byte_digest,
            "rgba_sha256": visual_digest,
            "user_task": source_task_prompt(task),
            "completion": completion,
            "source_state_success": True,
        }
    raise ValueError("H285 original browser generator did not yield")


def produce_train_only_batch(
    browser: Any,
    source_path: Path,
    sink: Callable[[bytes, str, str], None],
    *,
    role: str,
) -> dict:
    """Validate 24 real GUI actions, then feed fully validated TRAIN only.

    A trusted private training process supplies sink(image, prompt, label).
    Do not pass a model/evaluator/OOD callback as the sink. Never persist
    images, point labels, prompts or raw examples in public source output.
    """
    need(role == ROLE and type(role) is str,
         "not admitted for training/DEV/OOD evaluation")
    need(callable(sink), "explicit trusted TRAIN-only sink required")
    source, url = checked_source(source_path)
    rows = []
    png_hashes = set()
    rgba_hashes = set()
    seen = set()
    for scene in SCENES:
        row = _candidate_one(browser, source, url, scene)
        need(row["scene"] not in seen and
             row["png_sha256"] not in png_hashes and
             row["rgba_sha256"] not in rgba_hashes,
             "duplicate source screenshot, pixel content or scene ID")
        seen.add(row["scene"])
        png_hashes.add(row["png_sha256"])
        rgba_hashes.add(row["rgba_sha256"])
        rows.append(row)
    need(len(rows) == len(SCENES) == 24
         and len(png_hashes) == 24 and len(rgba_hashes) == 24,
         "candidate TRAIN batch incomplete or visually duplicated")
    # Only after ALL real browser state checks pass do TRAIN labels
    # flow into a caller-owned sink. No DEV, OOD, old benchmarks or
    # adapters were opened or scored.
    for row in rows:
        sink(row["png"], row["user_task"], row["completion"])
    return {
        "status": "H285_ORIGINAL_SYNTHETIC_BROWSER_TRAIN_CANDIDATES",
        "role": ROLE,
        "original_source_blob": SOURCE_BLOB,
        "actual_browser_source_actions_successful": 24,
        "distinct_png_screenshots": len(png_hashes),
        "distinct_decoded_rgba_screenshots": len(rgba_hashes),
        "english_candidates": sum(r["lang"] == "en" for r in rows),
        "arabic_candidates": sum(r["lang"] == "ar" for r in rows),
        "valid_h199_supervision_labels": len(rows),
        "official_train_manifest_admitted": False,
        "independent_dev_ood_evaluated": False,
        "old_consumed_ood_opened": False,
        "pretrained_model_generations": 0,
        "optimizer_steps": 0,
        "adapter_model_improvement_proven": False,
        "public_private_media_or_labels_saved": False,
    }
