"""H357: newly authored original ControlGarden single-app candidate factory.

Source-only development research, NOT independently approved TRAIN or blind
OOD. Every image/label is generated under caller-owned TEMP directories.
NO H199 official TRAIN admission, NO weights, GPU, paid API, or models.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import random

from h199.native_action_sft import parse_actions, source_task_prompt

ROLE = "TRAIN_DEV_REVIEW_CANDIDATE_ONLY_NOT_ADMITTED"
SCHEMA = "hala.h357.controlgarden.original-candidate.v1"
SHAPES = ("circle", "square", "triangle", "diamond")
COLORS = ("red", "blue", "green", "gold")
EN_SHAPES = ("circle", "square", "triangle", "diamond")
AR_SHAPES = ("الدائرة", "المربع", "المثلث", "المعين")
AR_COLORS = {
    "red": ("الحمراء", "الأحمر"), "blue": ("الزرقاء", "الأزرق"),
    "green": ("الخضراء", "الأخضر"), "gold": ("الذهبية", "الذهبي"),
}
SOURCE = Path(__file__).with_name("controlgarden.html")
COUNT = 64


def require(ok: bool):
    if not ok:
        raise ValueError("H357_ORIGINAL_SOURCE_CANDIDATE_REFUSED")


def candidate_specs():
    """Public/synthetic family; never represent any of it as sealed TEST."""
    roster = []
    shape_orders = tuple(permutations(SHAPES))
    color_orders = tuple(permutations(COLORS))
    used_layouts = set()
    for index in range(COUNT):
        # Each bilingual arm has 32 items: two observations per (slot, shape)
        # and eight per slot, shape and color. H360 balanced slot/color
        # margins but accidentally retained asymmetric shape frequencies.
        bilingual_pair = index // 2
        chosen = bilingual_pair % 4
        shape = SHAPES[(bilingual_pair // 4) % 4]
        color = COLORS[(chosen + (bilingual_pair // 4) % 4
                        + 2 * (bilingual_pair // 16)) % 4]
        language = "ar" if index % 2 else "en"
        # A deterministic public permutation of the *non-target* tiles
        # prevents identical screenshot layouts without deriving the
        # answer from language, color, shape or source ordering.
        options = [
            tuple((s, c) for s, c in zip(shapes, colors))
            for shapes in shape_orders if shapes[chosen] == shape
            for colors in color_orders if colors[chosen] == color
        ]
        random.Random(357001 + index).shuffle(options)
        selection = next((layout for layout in options
                          if layout not in used_layouts), None)
        require(selection is not None)
        used_layouts.add(selection)
        tiles = [[s, c] for s, c in selection]
        if language == "ar":
            color_word = AR_COLORS[color][0 if shape == "circle" else 1]
            instruction = "اختر " + AR_SHAPES[SHAPES.index(shape)] + " " + color_word
        else:
            instruction = "Select the " + color + " " + shape
        roster.append({
            "case_id": "h357-original-%03d" % index,
            "family": "controlgarden-single-app",
            "language": language,
            "task": instruction,
            "source_tiles": tiles,  # source/DEV only, NEVER model input
            "oracle_tile": chosen,  # private candidate owner gold, NOT OOD
            "split": "TRAIN_DEV_CANDIDATE_NOT_ADMITTED",
        })
    require(len(roster) == COUNT and
            len({r["case_id"] for r in roster}) == COUNT and
            {r["language"] for r in roster} == {"en", "ar"} and
            all(len({tuple(t) for t in r["source_tiles"]}) == 4 for r in roster))
    return roster


def capture_source_candidate(root: Path, *, browser_kind: str = "chromium"):
    """Capture actual original browser pixel raster; original native-state QA.

    Writes only caller-provided scratch files. Never writes inside Git source.
    All generated artifacts are UNAPPROVED source-only candidates and may not
    be used to qualify TRAIN rights, private OOD or pretrained model scores.
    """
    require(isinstance(root, Path) and not root.exists() and
            not root.resolve().is_relative_to(SOURCE.parent.resolve()) and
            browser_kind in ("chromium", "msedge"))
    original = SOURCE.read_bytes()
    require(b"http://" not in original and b"https://" not in original
            and b"fetch(" not in original and b"<script src=" not in original)
    source_sha = sha256(original).hexdigest()
    from playwright.sync_api import sync_playwright

    root.mkdir(parents=True, exist_ok=False)
    images = root / "images"
    images.mkdir()
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, **({"channel": "msedge"} if browser_kind == "msedge" else {}))
        try:
            page = browser.new_page(viewport={"width": 960, "height": 720},
                                    device_scale_factor=1)
            page.goto(SOURCE.resolve().as_uri(), wait_until="load")
            for spec in candidate_specs():
                page.evaluate("s=>window.configureOriginalScene(s)", spec["source_tiles"])
                page.mouse.move(0, 0)
                # Explicit original page state and stable double screenshot
                require(page.evaluate("window.getOriginalSelection()") is None)
                image = page.screenshot(animations="disabled")
                require(image == page.screenshot(animations="disabled"))
                button = page.locator(".board button").nth(spec["oracle_tile"])
                box = button.bounding_box()
                require(box is not None)
                point = [round(box["x"] + box["width"] / 2),
                         round(box["y"] + box["height"] / 2)]
                native = json.dumps({"actions": [{"kind": "click", "point": point}]},
                                    sort_keys=True, separators=(",", ":"))
                parse_actions(native, training_target=True)
                actual = spec["source_tiles"][spec["oracle_tile"]]
                page.mouse.click(*point)
                require(page.evaluate("window.getOriginalSelection()") ==
                        actual[0] + "-" + actual[1])
                path = images / (spec["case_id"] + ".png")
                path.write_bytes(image)
                records.append({
                    "id": spec["case_id"], "family": spec["family"],
                    "split": spec["split"], "language": spec["language"],
                    "prompt": source_task_prompt(spec["task"]),
                    "completion": native,
                    "image_path": "images/" + path.name,
                    "image_sha256": sha256(image).hexdigest(),
                    "original_source_sha256": source_sha,
                })
        finally:
            browser.close()
    require(len(records) == COUNT
            and len({r["image_sha256"] for r in records}) == COUNT)
    (root / "candidate-rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                for r in records), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "role": ROLE,
        "original_source_sha256": source_sha,
        "families": ["controlgarden-single-app"],
        "case_count": COUNT,
        "language_counts": {"en": COUNT // 2, "ar": COUNT // 2},
        "original_native_browser_successes": COUNT,
        "exact_pixel_double_captures": COUNT,
        "official_train_admitted": False,
        "independent_rights_reviewed": False,
        "fresh_blind_ood": False,
        "gpu_calls": 0,
        "can_train": False, "can_release": False,
    }
    (root / "candidate-manifest.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit("H357 produces only review-candidates under an explicitly supplied TEMP path; import and call capture_source_candidate().")
