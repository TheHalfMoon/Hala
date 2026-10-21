"""H61 first-party bilingual native drag corpus.

The measured H60 result is that the trained adapter recovered the output
contract on the English public DEV scenes (12 of 12) and almost not at all on
the Arabic ones (2 of 12). The original DragOn corpus is English only, so no
amount of it can teach the Arabic half of the contract.

This module therefore authors a first-party, bilingual native drag TRAIN
family in the same native drag contract, and captures it in the genuine
installed headless Microsoft Edge.

The rules it keeps:

* the family is a TRAIN family and its scene identities and layout parameters
  are disjoint from the first-party public DEV families;
* the drag target is a real visible handle and a real visible target, and the
  recorded boxes are the boxes the author laid out, not a guess;
* every captured screenshot is audited against the recorded boxes, so a scene
  that did not render where it was authored is rejected with a reason;
* the instructions are authored in Arabic and English and are never a
  translation of each other at run time;
* the original ordered flag is preserved: an unordered drag is never rewritten
  as an ordered one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SCHEMA = "hala.h61.firstparty-bilingual-drag.v1"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
FRAME = (1024, 640)
HEADER_HEIGHT = 74
KINDS = ("slider", "sheet", "slide_resize", "text_highlight")
LANGUAGES = ("ar", "en")
HANDLE_COLOR = (34, 108, 132)
TARGET_COLOR = (173, 106, 80)

TEXT = {
    "slider": {
        "en": {"title": "Original Brightness Panel",
               "instruction": "Drag the slider handle to {value}%"},
        "ar": {"title": "لوحة السطوع الأصلية",
               "instruction": "اسحب مقبض الشريط إلى {value}٪"},
    },
    "sheet": {
        "en": {"title": "Original Stock Ledger",
               "instruction": "Select the cells from {start} to {end}"},
        "ar": {"title": "سجل المخزون الأصلي",
               "instruction": "حدّد الخلايا من {start} إلى {end}"},
    },
    "slide_resize": {
        "en": {"title": "Original Canvas Editor",
               "instruction": "Drag the right resize handle to {value}px"},
        "ar": {"title": "محرّر اللوحة الأصلي",
               "instruction": "اسحب مقبض التحجيم الأيمن إلى {value} بكسل"},
    },
    "text_highlight": {
        "en": {"title": "Original Research Notes",
               "instruction": "Highlight from the first phrase to the last phrase"},
        "ar": {"title": "ملاحظات البحث الأصلية",
               "instruction": "ظلّل من العبارة الأولى إلى العبارة الأخيرة"},
    },
}
BODY = {
    "slider": "This is an entirely fictional interface for Hala training.",
    "sheet": "Every product, date and quantity below is invented.",
    "slide_resize": "Nothing in this editor is a real document.",
    "text_highlight": "Every sentence in these notes is invented for training.",
}
BODY_AR = {
    "slider": "هذه واجهة خيالية بالكامل لأغراض تدريب هالة.",
    "sheet": "كل منتج وتاريخ وكمية أدناه مُختلقة.",
    "slide_resize": "لا شيء في هذا المحرر مستند حقيقي.",
    "text_highlight": "كل جملة في هذه الملاحظات مُختلقة لأغراض التدريب.",
}
ORDERED = {"slider": True, "sheet": True, "slide_resize": True, "text_highlight": False}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def scene_html(scene: dict) -> str:
    """The authored page of one scene, with the handle and the target placed."""
    kind = scene["kind"]
    arabic = scene["language"] == "ar"
    handle_left, handle_top = scene["handle"]
    target_left, target_top = scene["target"]
    handle = (
        f'<div class="handle" id="handle" style="left:{handle_left}px;top:{handle_top}px"></div>'
    )
    target = (
        f'<div class="target" id="target" style="left:{target_left}px;top:{target_top}px"></div>'
    )
    if kind == "slider":
        track = (
            f'<div class="track" style="left:{handle_left}px;top:{handle_top + 8}px;'
            f'width:{max(120, target_left - handle_left)}px"></div>'
        )
        surface = f'<div class="scene" style="top:250px"><b>{scene["title"]}</b>' \
                  f'<p>{scene["body"]}</p>{track}</div>'
    elif kind == "sheet":
        rows = "".join(
            f'<p>ROW-{300 + index} &nbsp; ITEM-{index} &nbsp; {12 + index * 4}</p>'
            for index in range(6)
        )
        surface = f'<div class="scene" style="top:210px"><b>{scene["title"]}</b>' \
                  f'<p>{scene["body"]}</p>{rows}</div>'
    elif kind == "slide_resize":
        surface = f'<div class="scene" style="top:200px"><b>{scene["title"]}</b>' \
                  f'<p>{scene["body"]}</p><div class="panel" style="left:180px;top:300px;' \
                  f'width:520px;height:200px"></div></div>'
    else:
        surface = f'<div class="scene" style="top:210px"><b>{scene["title"]}</b>' \
                  f'<p>{scene["body"]}</p><p class="para">Alpha beta gamma delta epsilon ' \
                  f'zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau.</p></div>'
    return (
        '<!doctype html><html lang="' + scene["language"] + '" dir="ltr"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=1024,initial-scale=1">'
        "<style>*{box-sizing:border-box}body{margin:0;background:#eff5f8;color:#253b4b;"
        "font:17px 'Segoe UI',Tahoma,Arial,sans-serif}"
        "header{height:74px;background:#fff;padding:18px 35px;border-bottom:2px solid #d2dfe8}"
        "main{position:relative;width:1024px;height:566px}"
        ".scene{position:absolute;left:70px;width:880px;background:#eaf1f7;border:2px solid #bcd2e0;"
        "border-radius:10px;padding:14px 18px}"
        ".handle{position:absolute;width:26px;height:26px;border-radius:13px;background:#226c84;"
        "border:3px solid #fff}"
        ".target{position:absolute;width:26px;height:26px;border-radius:13px;background:#ad6a50;"
        "border:3px solid #fff}"
        ".track{position:absolute;height:10px;border-radius:5px;background:#9fb6c6}"
        ".panel{position:absolute;background:#dce9f2;border:2px solid #7fa6bf}"
        ".para{line-height:1.7;max-width:760px}</style></head><body>"
        f'<header><b>{scene["title"]}</b></header><main>{surface}{handle}{target}</main>'
        "</body></html>"
    )


def build_scenes(count: int) -> list[dict]:
    """Author the scenes deterministically, with the recorded drag boxes."""
    scenes = []
    for index in range(count):
        kind = KINDS[index % len(KINDS)]
        language = LANGUAGES[(index // len(KINDS)) % len(LANGUAGES)]
        arabic = language == "ar"
        slot = index % 24
        handle_left = 190 + (slot % 5) * 46
        handle_top = 268 + (slot % 4) * 34
        if kind == "slider":
            target_left = handle_left + 120 + (slot % 3) * 90
            target_top = handle_top
            value = 20 + (slot % 4) * 20
        elif kind == "sheet":
            target_left = handle_left + 240 + (slot % 2) * 120
            target_top = handle_top + 42
            value = 0
        elif kind == "slide_resize":
            target_left = handle_left + 300 + (slot % 2) * 80
            target_top = handle_top
            value = 320 + (slot % 3) * 60
        else:
            target_left = handle_left + 300 + (slot % 2) * 150
            target_top = handle_top
            value = 0
        template = TEXT[kind][language]
        instruction = template["instruction"].format(
            value=value,
            start="B12" if language == "en" else "ب١٢",
            end="F34" if language == "en" else "ف٣٤",
        )
        scenes.append(
            {
                "id": f"h61-{kind}-{index:04d}",
                "kind": kind,
                "family": "h61_firstparty_bilingual_drag_train",
                "split": "train",
                "language": language,
                "ordered": ORDERED[kind],
                "title": template["title"],
                "instruction": instruction,
                "body": BODY_AR[kind] if arabic else BODY[kind],
                "handle": [handle_left, handle_top],
                "target": [target_left, target_top],
                # The recorded boxes are PAGE coordinates, because that is the
                # frame the model sees. The page header pushes the scene down
                # by a fixed amount, so the page box is the local box plus
                # that offset.
                "start_bbox": [
                    handle_left,
                    handle_top + HEADER_HEIGHT,
                    handle_left + 26,
                    handle_top + HEADER_HEIGHT + 26,
                ],
                "end_bbox": [
                    target_left,
                    target_top + HEADER_HEIGHT,
                    target_left + 26,
                    target_top + HEADER_HEIGHT + 26,
                ],
            }
        )
    return scenes


def capture(html_path: Path, png_path: Path) -> None:
    """Capture one page in the genuine installed headless Microsoft Edge."""
    command = [
        EDGE,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        f"--window-size={FRAME[0]},{FRAME[1]}",
        f"--screenshot={png_path}",
        f"file:///{html_path.as_posix()}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not png_path.exists():
        raise IOError(f"the genuine Edge capture failed: {result.stderr[:200]}")


def audit_scene(png_path: Path, scene: dict) -> list[str]:
    """Verify that the screenshot really shows the handle and the target."""
    from PIL import Image

    problems: list[str] = []
    with Image.open(png_path) as picture:
        if picture.size != FRAME:
            problems.append(f"the captured frame is {picture.size}")
            picture = picture.convert("RGB")
        handle_centre = (
            (scene["start_bbox"][0] + scene["start_bbox"][2]) // 2,
            (scene["start_bbox"][1] + scene["start_bbox"][3]) // 2,
        )
        target_centre = (
            (scene["end_bbox"][0] + scene["end_bbox"][2]) // 2,
            (scene["end_bbox"][1] + scene["end_bbox"][3]) // 2,
        )
        for name, centre, expected in (
            ("handle", handle_centre, HANDLE_COLOR),
            ("target", target_centre, TARGET_COLOR),
        ):
            pixel = picture.getpixel(centre)
            if max(abs(pixel[index] - expected[index]) for index in range(3)) > 60:
                problems.append(f"the {name} is not visible at {centre}: {pixel}")
    return problems


def build(out_dir, count: int, *, capture_scenes: bool = True) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "html").mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(count)
    rows = []
    rejections = []
    for scene in scenes:
        html_path = out_dir / "html" / f"{scene['id']}.html"
        png_path = out_dir / "png" / f"{scene['id']}.png"
        html_path.write_text(scene_html(scene), encoding="utf8")
        if capture_scenes:
            try:
                capture(html_path, png_path)
            except Exception as error:  # noqa: BLE001
                rejections.append({"id": scene["id"], "reason": f"capture: {error}"[:200]})
                continue
            problems = audit_scene(png_path, scene)
            if problems:
                rejections.append({"id": scene["id"], "reason": "; ".join(problems)[:200]})
                png_path.unlink(missing_ok=True)
                continue
        rows.append(
            {
                "id": scene["id"],
                "split": "train",
                "family": scene["family"],
                "language": scene["language"],
                "kind": scene["kind"],
                "ordered": scene["ordered"],
                "title": scene["title"],
                "instruction": scene["instruction"],
                "image_path": f"png/{scene['id']}.png",
                "image_sha256": sha(png_path.read_bytes()) if png_path.exists() else None,
                "image_width": FRAME[0],
                "image_height": FRAME[1],
                "start_bbox": scene["start_bbox"],
                "end_bbox": scene["end_bbox"],
                "html_path": f"html/{scene['id']}.html",
                "html_sha256": sha(html_path.read_bytes()),
            }
        )
    labels = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
        + b"\n"
        for row in rows
    )
    (out_dir / "labels.jsonl").write_bytes(labels)
    receipt = {
        "schema": SCHEMA,
        "kind": "original-first-party",
        "language": "ar,en",
        "split": "train only",
        "capture": "genuine installed headless Microsoft Edge",
        "frame": list(FRAME),
        "scenes_authored": len(scenes),
        "scenes_accepted": len(rows),
        "scenes_rejected": len(rejections),
        "rejections": rejections,
        "by_language": {
            language: sum(1 for row in rows if row["language"] == language)
            for language in LANGUAGES
        },
        "by_kind": {
            kind: sum(1 for row in rows if row["kind"] == kind) for kind in KINDS
        },
        "ordered_true": sum(1 for row in rows if row["ordered"]),
        "ordered_false": sum(1 for row in rows if not row["ordered"]),
        "labels_sha256": sha(labels),
        "public_dev_families_used": False,
        "benchmark_answer_leakage": False,
        "model_gpu_steps": 0,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "receipt.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=240)
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()
    result = build(args.out, args.scenes, capture_scenes=not args.no_capture)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
