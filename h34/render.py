"""Frozen synthetic GUI rendering with fail-closed Linux/RAQM/font checks."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import io
import urllib.request
import zipfile

PILLOW_PIN = "12.3.0"
RAQM_PIN = "0.10.5"
FONT_URL = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-sans-ttf-2.37.zip"


def fetch_pinned_font(output: Path, font_pin: dict) -> Path:
    """Fetch one fixed, bounded upstream archive and extract only its pinned member."""
    if FONT_URL != font_pin["upstream"] or font_pin["archive_max_bytes"] > 500_000:
        raise RuntimeError("unreviewed font source or archive limit")
    output = Path(output)
    if output.exists():
        if output.is_symlink() or output.stat().st_size != font_pin["bytes"] or hashlib.sha256(output.read_bytes()).hexdigest() != font_pin["sha256"]:
            raise RuntimeError("existing font cache differs from frozen bytes")
        return output
    request = urllib.request.Request(FONT_URL, headers={"User-Agent": "Hala-H34-research/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(font_pin["archive_max_bytes"] + 1)
    if len(payload) > font_pin["archive_max_bytes"] or hashlib.sha256(payload).hexdigest() != font_pin["archive_sha256"]:
        raise RuntimeError("font archive bytes failed frozen hash/size")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = [entry for entry in archive.infolist() if entry.filename == font_pin["member"]]
        if len(infos) != 1 or infos[0].file_size != font_pin["bytes"]:
            raise RuntimeError("exact pinned font member missing")
        font_bytes = archive.read(infos[0])
    if hashlib.sha256(font_bytes).hexdigest() != font_pin["sha256"]:
        raise RuntimeError("font member bytes failed frozen hash")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(font_bytes)
    return output


def render_case(case: dict, output: Path, font_path: Path, font_pin: dict) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("H34 actual media requires Linux RAQM")
    if font_pin.get("status") != "pinned" or not font_pin.get("upstream_revision") or not font_pin.get("sha256"):
        raise RuntimeError("H34 font pin awaits independently verified upstream revision and bytes")
    font_path = Path(font_path)
    raw_font = font_path.read_bytes()
    if hashlib.sha256(raw_font).hexdigest() != font_pin["sha256"]:
        raise RuntimeError("font bytes do not match frozen font pin")
    from PIL import Image, ImageDraw, ImageFont, features, __version__
    if __version__ != PILLOW_PIN or features.version_feature("raqm") != RAQM_PIN:
        raise RuntimeError("unqualified Pillow/RAQM version")
    if not features.check_feature("raqm"):
        raise RuntimeError("Pillow RAQM support required")
    image = Image.new("RGB", (case["width"], case["height"]), "#f4f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 22, layout_engine=ImageFont.Layout.RAQM)
    draw.rounded_rectangle((32, 26, 928, 122), radius=12, fill="white", outline="#c5ccd3", width=2)
    rtl = case["language"] == "ar"
    draw.text((900 if rtl else 56, 48), case["title"], font=font, fill="#17212b",
              anchor="ra" if rtl else "la", direction="rtl" if rtl else "ltr")
    draw.rounded_rectangle((32, 152, 928, 494), radius=12, fill="white", outline="#c5ccd3", width=2)
    for control in case["controls"]:
        x, y, w, h = control["bounds"]
        draw.rounded_rectangle((x, y, x+w, y+h), radius=8, fill="#e8f0fe", outline="#4976b8", width=2)
        label = control["label"]
        draw.text((x+w-12 if rtl else x+12, y+h//2), label, font=font, fill="#17212b",
                  anchor="rm" if rtl else "lm", direction="rtl" if rtl else "ltr")
    output = Path(output)
    image.save(output, format="PNG", optimize=False, compress_level=9)
    payload = output.read_bytes()
    return {"path": output.name, "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "width": image.width, "height": image.height,
            "font_sha256": font_pin["sha256"], "font_revision": font_pin["upstream_revision"],
            "license": font_pin["license"], "pillow": __version__,
            "raqm": features.version_feature("raqm"), "renderer": "hala-h34-fixed-v1"}


def load_font_pin() -> dict:
    return json.loads((Path(__file__).parent / "font_pin.json").read_text(encoding="utf-8"))
