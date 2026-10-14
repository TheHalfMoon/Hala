"""Optional genuine PNG fixtures; Arabic shaping capability is explicit."""
from pathlib import Path


def render_case(case, path, font_path=None):
    from PIL import Image, ImageDraw, ImageFont, features
    frame = case["frame"]
    image = Image.new("RGB", (frame["width"], frame["height"]), "#edf1f7")
    draw = ImageDraw.Draw(image)
    candidates = [font_path] if font_path else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    font_file = next((p for p in candidates if p and Path(p).is_file()), None)
    if font_file is None:
        raise ValueError("A Unicode font is required; pass --font with an installed font path")
    shaping = features.check_feature("raqm")
    font = ImageFont.truetype(font_file, max(12, int(12 * frame["scale"])))
    visible = [c for c in case["controls"] if all(case["initial_state"].get(k) == v for k, v in c["visible_if"].items())]
    for control in visible:
        x, y, width, height = control["bounds"]
        draw.rectangle((x, y, x + width, y + height), fill="white", outline="#26374a", width=max(1, int(frame["scale"])))
        # Long unique IDs are clipped to the control boundary, just as GUI labels.
        label = control["name"]
        while len(label) > 1 and draw.textlength(label, font=font, **({"direction": "rtl"} if case["rtl"] and shaping else {})) > width - 14:
            label = label[:-1]
        options = {"direction": "rtl", "anchor": "ra"} if case["rtl"] and shaping else {}
        point = (x + width - 7, y + 12 * frame["scale"]) if options else (x + 7, y + 12 * frame["scale"])
        draw.text(point, label, font=font, fill="#152535", **options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return {"path": str(path.resolve()), "rtl_layout": case["rtl"], "rtl_shaping_supported": shaping, "font": str(font_file), "note": "Initial-state synthetic image; not a captured real application. Arabic shaping requires Pillow RAQM."}
