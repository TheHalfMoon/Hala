"""H338: 128 author-original SignalDesk two-step TRAIN candidates ONLY.

No pretrained/adapted model, private image, hidden evaluation, TRAIN/DEV/OOD
admission, weights, or optimizer. Full 256-frame distinct source-native browser
validation finishes before exactly ONE trusted *in-memory* batch handoff.
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

SOURCE_BLOB = "3d787aa3aafb999d342cbd3685c20d0f34d5fcd8"
ROLE = "H338_ORIGINAL_SIGNALDESK_LARGE_TRAIN_CANDIDATES_ONLY_NOT_ADMITTED"
IDS = ("S-14", "S-28", "S-42", "S-56")
SCENES = tuple((
    "sig_train_%03d" % i,
    "en" if i < 64 else "ar",
    (i % 64) // 16,
    bool(((i % 64) // 8) % 2),
    bool(((i % 64) // 4) % 2),
    (i % 64) % 4,
    "SIM-%03d" % i,
    i,
) for i in range(128))


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H338 ORIGINAL_LARGE_CANDIDATES_REFUSED")


def checked_source(source: Path) -> tuple[Path, str]:
    need(isinstance(source, Path))
    try:
        p = source.absolute()
        need(p.name == "signaldesk.html" and p.parent.name == "h338"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p))
        raw = p.read_bytes()
        blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                    + b"\x00" + raw).hexdigest()
        need(compare_digest(blob, SOURCE_BLOB))
        return p, p.as_uri()
    except (OSError, RuntimeError, ValueError):
        raise ValueError("H338 SOURCE_GIT_BLOB_PIN_REFUSED") from None


def exact_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in SCENES)
    _, lang, _, _, _, target_idx, code, _ = scene
    return (
        "Activate fictional signal " + IDS[target_idx]
        + ", then type invented code " + code
        + " in the newly revealed field; use two separate steps."
        if lang == "en" else
        "فعّل الإشارة الخيالية " + IDS[target_idx]
        + " ثم اكتب الرمز الافتراضي " + code
        + " في الحقل الجديد على خطوتين منفصلتين."
    )


def inspected_state(page: Any) -> dict[str, tuple[bool, str]]:
    state = page.evaluate("() => window.signalDeskState()")
    need(type(state) is list and len(state) == len(IDS))
    rows = {}
    for rec in state:
        need(type(rec) is dict and set(rec) == {"id", "open", "value"}
             and type(rec["id"]) is str and rec["id"] in IDS
             and rec["id"] not in rows and type(rec["open"]) is bool
             and type(rec["value"]) is str and len(rec["value"]) <= 14)
        rows[rec["id"]] = (rec["open"], rec["value"])
    need(set(rows) == set(IDS))
    return rows


@contextmanager
def open_original(browser: Any, url: str, scene: tuple):
    ctx = new_offline_context(browser)
    try:
        def offline(route: Any) -> None:
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        ctx.route("**/*", offline)
        page = ctx.new_page()
        page.goto(url, wait_until="load")
        _, lang, order, dark, dense, _, _, run = scene
        page.evaluate(
            "p => window.configureSignalDesk(p)",
            {"lang": lang, "order": order, "dark": dark,
             "dense": dense, "run": run},
        )
        need(page.url == url
             and page.viewport_size == {"width": WIDTH, "height": HEIGHT}
             and page.locator("html").get_attribute("dir") ==
             ("rtl" if lang == "ar" else "ltr")
             and page.locator("button[data-signal]").count() == len(IDS)
             and page.locator("#scenario").inner_text() == "SIM %03d" % run)
        yield page
    finally:
        ctx.close()


def decoded_rgb_hash(png: bytes) -> bytes:
    from PIL import Image
    need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n"))
    with Image.open(BytesIO(png)) as im:
        need(im.format == "PNG" and im.size == (WIDTH, HEIGHT))
        rgb = im.convert("RGB")
        rgb.load()
        return sha256(rgb.tobytes()).digest()


def one_original_case(browser: Any, url: str, scene: tuple) -> list[dict]:
    need(type(scene) is tuple and scene in SCENES)
    _, _, _, _, _, idx, code, _ = scene
    target = IDS[idx]
    prompt = source_task_prompt(exact_task(scene))
    with open_original(browser, url, scene) as page:
        empty = {k: (False, "") for k in IDS}
        need(inspected_state(page) == empty)
        button = page.locator('button[data-signal="' + target + '"]')
        need(button.count() == 1 and button.is_visible())
        box = button.bounding_box()
        need(box is not None and box["width"] >= 30 and box["height"] >= 30)
        point = [int(box["x"] + box["width"] / 2),
                 int(box["y"] + box["height"] / 2)]
        need(0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT)
        first = source_capture_png(page)
        first_rgb = decoded_rgb_hash(first)
        click = json.dumps({"actions": [{"kind": "click", "point": point}]},
                           separators=(",", ":"))
        parse_actions(click, training_target=True)
        need(decoded_rgb_hash(source_capture_png(page)) == first_rgb)
        need(execute_one(page, click) == {"kind": "click", "executed": True})
        expected = {k: (k == target, "") for k in IDS}
        need(inspected_state(page) == expected)
        focused = page.evaluate("""() => {
            const e = document.activeElement;
            return e instanceof HTMLInputElement ? e.dataset.signal : null;
        }""")
        need(focused == target)
        second = source_capture_png(page)
        second_rgb = decoded_rgb_hash(second)
        need(second_rgb != first_rgb)
        typing = json.dumps(
            {"actions": [{"kind": "type_text", "text": code}]},
            separators=(",", ":"))
        parse_actions(typing, training_target=True)
        need(decoded_rgb_hash(source_capture_png(page)) == second_rgb)
        need(execute_one(page, typing)
             == {"kind": "type_text", "executed": True})
        final = {k: (k == target, code if k == target else "") for k in IDS}
        need(inspected_state(page) == final)
        return [
            {"png": first, "rgb_hash": first_rgb,
             "task": prompt, "native_label": click},
            {"png": second, "rgb_hash": second_rgb,
             "task": prompt, "native_label": typing},
        ]


def produce_large_train_candidates(
    browser: Any,
    source_path: Path,
    trusted_batch_sink: Callable[[tuple[tuple[bytes, str, str], ...]], bool],
    *, role: str,
) -> dict:
    """All-or-nothing SOURCE validation then ONE ephemeral trusted handoff.

    The callback must separately provide atomic rollback; returning True
    cannot independently prove storage/transaction semantics. No data is
    output before EVERY original browser scene has been validated.
    """
    need(type(role) is str and role == ROLE
         and callable(trusted_batch_sink))
    _, url = checked_source(source_path)
    need(len(SCENES) == 128
         and sum(scene[1] == "en" for scene in SCENES) == 64
         and sum(scene[1] == "ar" for scene in SCENES) == 64)
    rows: list[dict] = []
    rgb_set = set()
    for scene in SCENES:
        pair = one_original_case(browser, url, scene)
        need(len(pair) == 2)
        for row in pair:
            need(row["rgb_hash"] not in rgb_set)
            rgb_set.add(row["rgb_hash"])
            rows.append(row)
    need(len(rows) == 256 and len(rgb_set) == 256)
    batch = tuple((row["png"], row["task"], row["native_label"])
                  for row in rows)
    need(len(batch) == 256
         and all(type(image) is bytes and type(task) is str
                 and type(label) is str for image, task, label in batch))
    need(trusted_batch_sink(batch) is True)
    return {
        "status": "H338_ORIGINAL_SIGNALDESK_TRAIN_CANDIDATES_ONLY",
        "source_html_git_blob": SOURCE_BLOB,
        "source_scenes": 128,
        "english_scenes": 64, "arabic_rtl_scenes": 64,
        "validated_native_action_stages": 256,
        "distinct_decoded_rgb_images": 256,
        "train_candidate_messages": 256,
        "batch_handoff_mode": "one_shot_transient",
        "official_train_admitted": False,
        "independent_dev_ood_qualified": False,
        "genuine_pretrained_or_adapted_generations": 0,
        "optimizer_steps": 0,
        "measured_model_improvement": False,
        "images_labels_tasks_or_coordinates_persisted": False,
        "can_release": False,
    }
