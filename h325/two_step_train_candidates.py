"""H325: original RailKit two-stage dynamic TRAIN candidates ONLY.

A distinct author-owned GUI source, no external media, patients or accounts.
Rendered frame, user task and reference action labels stay exclusively in
volatile RAM. No official TRAIN/DEV/OOD admission or model inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha1, sha256
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
import json

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h322.source_capture import source_capture_png

SOURCE_BLOB = "743b5e1b54774a1deea4dbe6bfbe0613adead7e4"
ROLE = "ORIGINAL_RAILKIT_TWO_STEP_TRAIN_CANDIDATES_ONLY_NOT_ADMITTED"
IDS = ("R-61", "R-73", "R-85", "R-97")
SCENES = tuple((
    "rail_train_%02d" % i,
    "en" if i < 8 else "ar",
    i % 4,
    (i // 4) % 2 == 1,
    (i * 3 + i // 4) % 4,
    "LAB-%02d" % (30 + i),
) for i in range(16))


def need(value: bool) -> None:
    if not value:
        raise ValueError("H325 ORIGINAL_TRAIN_CANDIDATE_REFUSED")


def checked_source(source: Path) -> tuple[Path, str]:
    need(isinstance(source, Path))
    try:
        p = source.absolute()
        need(p.name == "railkit.html" and p.parent.name == "h325"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p))
        raw = p.read_bytes()
        blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                    + b"\x00" + raw).hexdigest()
        need(compare_digest(blob, SOURCE_BLOB))
        return p, p.as_uri()
    except (OSError, RuntimeError, ValueError):
        raise ValueError("H325 ORIGINAL_SOURCE_PIN_REFUSED") from None


def exact_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in SCENES)
    _, language, _, _, index, code = scene
    return (
        "Connect fictional rail " + IDS[index] + " then type code " + code
        + " into the newly revealed input; use two steps."
        if language == "en" else
        "اربط المسار الخيالي " + IDS[index] + " ثم اكتب الرمز "
        + code + " في الحقل الجديد، بخطوتين منفصلتين."
    )


def inspected_state(page: Any) -> dict[str, tuple[bool, str]]:
    state = page.evaluate("() => window.railKitState()")
    need(type(state) is list and len(state) == 4)
    rows = {}
    for record in state:
        need(type(record) is dict and set(record) == {"id", "open", "value"}
             and record["id"] in IDS and record["id"] not in rows
             and type(record["open"]) is bool
             and type(record["value"]) is str and len(record["value"]) <= 12)
        rows[record["id"]] = (record["open"], record["value"])
    need(set(rows) == set(IDS))
    return rows


@contextmanager
def open_original(browser: Any, url: str, scene: tuple):
    context = new_offline_context(browser)
    try:
        def offline(route: Any) -> None:
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        context.route("**/*", offline)
        page = context.new_page()
        page.goto(url, wait_until="load")
        _, lang, order, dark, _, _ = scene
        page.evaluate("x => window.configureRailKit(x)",
                      {"lang": lang, "order": order, "dark": dark})
        need(page.url == url and page.viewport_size ==
             {"width": WIDTH, "height": HEIGHT}
             and page.locator("html").get_attribute("dir") ==
             ("rtl" if lang == "ar" else "ltr")
             and page.locator("button[data-route]").count() == 4)
        yield page
    finally:
        context.close()


def _visible_rgb_hash(png: bytes) -> bytes:
    from PIL import Image
    need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n"))
    with Image.open(BytesIO(png)) as im:
        need(im.format == "PNG" and im.size == (WIDTH, HEIGHT))
        rgb = im.convert("RGB")
        rgb.load()
        return sha256(rgb.tobytes()).digest()


def one_source_case(browser: Any, url: str, scene: tuple) -> list[dict]:
    need(type(scene) is tuple and scene in SCENES)
    target = IDS[scene[4]]
    code = scene[5]
    prompt = source_task_prompt(exact_task(scene))
    with open_original(browser, url, scene) as page:
        empty = {route: (False, "") for route in IDS}
        need(inspected_state(page) == empty)
        button = page.locator('button[data-route="' + target + '"]')
        need(button.count() == 1 and button.is_visible())
        box = button.bounding_box()
        need(box is not None and box["width"] >= 30 and box["height"] >= 30)
        point = [int(box["x"] + box["width"] / 2),
                 int(box["y"] + box["height"] / 2)]
        need(0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT)
        first = source_capture_png(page)
        digest_first = _visible_rgb_hash(first)
        # Gold belongs only to the trusted source TRAIN extractor, never a VLM.
        click = json.dumps({"actions": [{"kind": "click", "point": point}]},
                           separators=(",", ":"))
        parse_actions(click, training_target=True)
        need(_visible_rgb_hash(source_capture_png(page)) == digest_first)
        need(execute_one(page, click) == {"kind": "click", "executed": True})
        expected = {route: (route == target, "") for route in IDS}
        need(inspected_state(page) == expected)
        focus = page.evaluate("""() => {
            const element=document.activeElement;
            return element instanceof HTMLInputElement
                ? element.dataset.route : null;
        }""")
        need(focus == target)
        second = source_capture_png(page)
        digest_second = _visible_rgb_hash(second)
        need(digest_second != digest_first)
        write = json.dumps({"actions": [{"kind": "type_text", "text": code}]},
                           separators=(",", ":"))
        parse_actions(write, training_target=True)
        need(_visible_rgb_hash(source_capture_png(page)) == digest_second)
        need(execute_one(page, write) ==
             {"kind": "type_text", "executed": True})
        final = {route: (route == target,
                          code if route == target else "") for route in IDS}
        need(inspected_state(page) == final)
        return [
            {"png": first, "rgb_digest": digest_first,
             "task": prompt, "native_label": click},
            {"png": second, "rgb_digest": digest_second,
             "task": prompt, "native_label": write},
        ]


def produce_two_step_train_candidates(
    browser: Any, source_path: Path,
    trusted_sink: Callable[[bytes, str, str], None] | None = None, *,
    role: str,
    trusted_batch_sink: Callable[[tuple[tuple[bytes, str, str], ...]], bool]
    | None = None,
) -> dict:
    """Validate ALL cases and distinct model-visible frames before sink.

    Sink is only a trusted transient TRAIN-candidate consumer. It must not
    be an eval/OOD model callback or a public/private file writer. A caller
    must separately prove ownership, custody and future data admission.

    Opt-in trusted_batch_sink receives one immutable tuple of 32 immutable
    (PNG, task, label) tuples after ALL source checks. It must return the
    literal True. One callback invocation is NOT a storage transaction:
    the consumer must independently ensure all-or-nothing effects and
    rollback on errors. Legacy trusted_sink retains its 32 calls.
    """
    need(type(role) is str and role == ROLE)
    need((trusted_sink is None) != (trusted_batch_sink is None))
    need(callable(trusted_batch_sink if trusted_batch_sink is not None
                  else trusted_sink))
    _, url = checked_source(source_path)
    rows = []
    all_rgb = set()
    for scene in SCENES:
        pairs = one_source_case(browser, url, scene)
        need(len(pairs) == 2)
        for row in pairs:
            digest = row["rgb_digest"]
            need(digest not in all_rgb)
            all_rgb.add(digest)
            rows.append(row)
    need(len(rows) == 32 and len(all_rgb) == 32)
    # All source checks precede any consumer invocation. Legacy calls can
    # still partially deliver if a consumer fails; opt in to one-shot.
    if trusted_batch_sink is not None:
        batch = tuple((row["png"], row["task"], row["native_label"])
                      for row in rows)
        need(len(batch) == 32 and all(
            type(png) is bytes and type(task) is str and type(label) is str
            for png, task, label in batch))
        need(trusted_batch_sink(batch) is True)
    else:
        for row in rows:
            trusted_sink(row["png"], row["task"], row["native_label"])
    return {
        "status": "H325_ORIGINAL_TWO_STEP_TRAIN_CANDIDATE_ONLY",
        "source_html_git_blob": SOURCE_BLOB,
        "source_scenes": len(SCENES),
        "original_native_action_states_verified": len(SCENES) * 2,
        "distinct_decoded_rgb_frames": len(all_rgb),
        "train_candidate_messages": len(rows),
        "candidate_handoff_mode": ("one_shot" if trusted_batch_sink is not None
                                   else "legacy_per_row"),
        "english_scenes": 8, "arabic_scenes": 8,
        "official_train_admitted": False,
        "independent_dev_ood_qualified": False,
        "pretrained_or_adapted_generations": 0,
        "optimizer_steps": 0,
        "model_skill_gain_proven": False,
        "public_model_media_labels_or_coords_saved": False,
    }
