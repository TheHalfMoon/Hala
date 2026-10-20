"""Freeze H37 genuine offline Edge browser pixel captures and DOM oracles.

Host-generated synthetic HTML is NOT an independent external app screenshot.
Model projection excludes evaluator-only bounds/control identity/oracle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import zlib

from h37.scenes import cases, scenes_sha256, model_projection

PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def checked_png(image: bytes) -> tuple[int, int]:
    if not isinstance(image, bytes) or len(image) > 2_000_000 or not image.startswith(PNG_MAGIC):
        raise ValueError("invalid or oversized H37 screenshot")
    offset, width, height = 8, None, None
    seen_ending = False
    while offset < len(image):
        if len(image) - offset < 12:
            raise ValueError("truncated PNG chunk")
        size = struct.unpack(">I", image[offset:offset + 4])[0]
        name = image[offset + 4:offset + 8]
        end = offset + 12 + size
        if size > 2_000_000 or end > len(image):
            raise ValueError("truncated or oversized PNG chunk")
        data = image[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", image[end - 4:end])[0]
        if zlib.crc32(name + data) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        if name == b"IHDR":
            if width is not None or size != 13:
                raise ValueError("invalid PNG image header")
            width, height = struct.unpack(">II", data[:8])
            if (width, height) != (960, 540):
                raise ValueError("unexpected screenshot dimensions")
        if name == b"IEND":
            seen_ending = end == len(image) and size == 0
            break
        offset = end
    if not seen_ending or width is None:
        raise ValueError("PNG terminal record missing")
    return width, height


def freeze(source: Path, dest: Path) -> dict:
    """Require all cases/images before create-only saving into source control."""
    source, dest = Path(source), Path(dest)
    if dest.exists():
        raise FileExistsError("frozen H37 evidence already exists")
    capture_bytes = (source / "capture.json").read_bytes()
    capture = json.loads(capture_bytes)
    meta = json.loads((source / "meta.json").read_text("utf-8"))
    authored = cases()
    if (capture.get("schema") != "hala.h37.browser_capture.v1"
            or capture.get("rendered_records") is None or
            len(capture["rendered_records"]) != 64 or
            capture.get("native_actions_executed") is not False or
            capture.get("model_inference") is not False or
            capture.get("playwright_core") != "1.56.1" or
            capture.get("cases_sha256") != scenes_sha256() or
            meta["scenes_sha256"] != scenes_sha256()):
        raise ValueError("H37 capture/source identification changed")
    if json.loads((source / "scenes.json").read_text("utf-8")) != authored:
        raise ValueError("authored scene plan changed")
    if json.loads((source / "model_projection.json").read_text("utf-8")) != model_projection(authored):
        raise ValueError("H37 original model projection changed")
    proof = []
    controls_snapshot = []
    for scene, record in zip(authored, capture["rendered_records"]):
        for key in ("id", "language", "layout", "family", "split", "variant", "goal", "expected_control_id"):
            if scene[key] != record.get(key):
                raise ValueError("H37 trial order or oracle differs")
        if (record["width"], record["height"]) != (960, 540):
            raise ValueError("H37 browser viewport differed")
        html_bytes = (source / "pages" / (scene["id"] + ".html")).read_bytes()
        html_sha = hashlib.sha256(html_bytes).hexdigest()
        if html_sha != meta["html_sha256"][scene["id"]] or html_sha != record["html_sha256"]:
            raise ValueError("H37 original page bytes differ")
        png = (source / "screenshots" / (scene["id"] + ".png")).read_bytes()
        checked_png(png)
        if hashlib.sha256(png).hexdigest() != record["image_sha256"] or len(png) != record["bytes"]:
            raise ValueError("H37 browser PNG does not match capture receipt")
        elements = record.get("controls")
        if type(elements) is not list or len(elements) != 4:
            raise ValueError("H37 expected exactly four observed browser buttons")
        bounds = []
        for index, (element, expected_label) in enumerate(zip(elements, scene["options"])):
            x, y, w, h = element["bounds"]
            if (element["id"] != f"control-{index}" or
                    element["label"] != expected_label or
                    element["visible"] is not True or
                    not all(type(n) in (int, float) for n in (x,y,w,h)) or
                    w < 40 or h < 35 or x < 0 or y < 0 or x+w > 960 or y+h > 540):
                raise ValueError("H37 DOM control geometry/label invalid")
            bounds.append((x,y,w,h))
        for i,(x,y,w,h) in enumerate(bounds):
            for q,(a,b,c,e) in enumerate(bounds):
                if q > i and max(x,a) < min(x+w,a+c) and max(y,b) < min(y+h,b+e):
                    raise ValueError("H37 clickable controls overlap")
        target = [e for e in elements if e["id"] == scene["expected_control_id"]]
        if len(target) != 1:
            raise ValueError("H37 independent oracle missing or ambiguous")
        controls_snapshot.append({"id": scene["id"], "split": scene["split"],
                                 "language": scene["language"], "layout": scene["layout"],
                                 "family": scene["family"], "goal": scene["goal"],
                                 "width": 960, "height":540,
                                 "image": scene["id"]+".png",
                                 "image_sha256":record["image_sha256"],
                                 "controls":[{"element_ref":e["id"],"label":e["label"],
                                              "bounds":e["bounds"]} for e in elements],
                                 "expected_label":target[0]["label"],
                                 "expected_control_id":scene["expected_control_id"]})
        proof.append({"id":scene["id"],"page_sha256":html_sha,
                      "image_sha256":record["image_sha256"], "bytes":len(png)})
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "pages").mkdir()
    (dest / "screenshots").mkdir()
    for item in proof:
        for source_subdir, suffix, dest_subdir in (("pages",".html","pages"),
                                                   ("screenshots",".png","screenshots")):
            name=item["id"]+suffix
            shutil.copyfile(source/source_subdir/name,dest/dest_subdir/name)
    (dest / "capture.json").write_bytes(capture_bytes)
    (dest / "meta.json").write_bytes((source / "meta.json").read_bytes())
    (dest / "scenes.json").write_bytes((source / "scenes.json").read_bytes())
    (dest / "evaluator_only.json").write_text(
        json.dumps(controls_snapshot,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8",newline="\n")
    (dest / "model_inputs.json").write_text(
        json.dumps(model_projection(authored),ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8",newline="\n")
    manifest={"schema":"hala.h37.browser_rendered_dev_test.v1",
              "scenes_sha256":scenes_sha256(),
              "capture_sha256":hashlib.sha256(capture_bytes).hexdigest(),
              "model_inputs_sha256":hashlib.sha256((dest/"model_inputs.json").read_bytes()).hexdigest(),
              "evaluator_sha256":hashlib.sha256((dest/"evaluator_only.json").read_bytes()).hexdigest(),
              "browser_version_observed":capture["browser_version"],
              "playwright_core":"1.56.1", "count":64,
              "dev":32, "test":32, "images":proof,
              "real_external_apps":False, "independently_blinded_test":False,
              "model_evaluated":False,"native_actions_executed":False}
    (dest/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8",newline="\n")
    return manifest
