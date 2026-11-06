"""H426 first-party synthetic GUI drag task generator.

Pure offline Pillow geometry: no external assets, scraped screenshots, model
weights, upload, training, clinical documents or private user information.
This is research TRAIN CANDIDATE data, not an admitted licensed source set.
"""
from __future__ import annotations

import hashlib
import io
import json
import random

from PIL import Image, ImageDraw

WIDTH = 960
HEIGHT = 640
DOMAINS = ("slider", "slide_resize", "text_highlight")
SPLITS = {"train": (0, 1, 2), "dev": (3, 4)}
SCHEMA = "hala.h426.firstparty-invented-bilingual-paired-drag.v2"
TARGET_COLOR = (50, 160, 110)
MAX_INDEX = 1000000


def _rng(split: str, index: int, domain: str):
    if split not in SPLITS or domain not in DOMAINS:
        raise ValueError("H426 synthetic split/domain is unregistered")
    if type(index) is not int or not 0 <= index < MAX_INDEX:
        raise ValueError("H426 synthetic index must be a bounded natural number")
    seed = int.from_bytes(hashlib.sha256(
        f"H426-2026:{split}:{index}:{domain}".encode("ascii")
    ).digest()[:8], "big")
    return random.Random(seed)


def _norm(p):
    if (type(p) is not tuple or len(p) != 2 or
        any(type(v) is not int for v in p) or
        not (0 <= p[0] <= WIDTH and 0 <= p[1] <= HEIGHT)):
        raise ValueError("H426 synthetic action has offscreen native source coordinates")
    return [round(p[0] * 1000 / WIDTH), round(p[1] * 1000 / HEIGHT)]


def _draw_scene(split, index, domain):
    rng = _rng(split, index, domain)
    family = SPLITS[split][index % len(SPLITS[split])]
    paper = (241, 243, 246) if family < 3 else (228, 237, 234)
    ink = (40, 55, 74) if family < 3 else (30, 65, 65)
    accent = (28, 129 + rng.randint(0, 35), 176) if family < 3 else (52, 110, 92)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), paper)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH - 1, 62), fill=ink)
    draw.text((25, 23), f"LOCAL STUDIO / {domain.upper()}", fill=(245, 245, 245))
    draw.rounded_rectangle((55, 96, 904, 585), radius=18 if family < 3 else 3,
                           fill=(255, 255, 255), outline=(189, 201, 207), width=2)
    # Unique visible per-case marker avoids distinct labels sharing identical PNGs.
    draw.text((84, 121), f"Workspace {index:06d} / family {family}", fill=ink)
    # Every screenshot exposes TWO independent GUI targets and source actions.
    if domain == "slider":
        rows = []
        for item, y in enumerate((260, 430)):
            x0, x1 = 225, 755
            start = (x0 + 65 + rng.randint(0, 100), y)
            target = (x1 - 75 - rng.randint(0, 95), y)
            draw.line((x0, y, x1, y), fill=(199, 207, 214), width=14)
            draw.line((x0, y, start[0], y), fill=accent, width=14)
            draw.ellipse((start[0] - 18, y - 18, start[0] + 18, y + 18),
                         fill=(255, 255, 255), outline=ink, width=4)
            draw.text((100, y - 12), f"LEVEL {item+1}", fill=ink)
            draw.ellipse((target[0] - 12, y - 12, target[0] + 12, y + 12),
                         outline=TARGET_COLOR, width=3)
            draw.ellipse((target[0] - 3, y - 3, target[0] + 3, y + 3),
                         fill=TARGET_COLOR)
            rows.append((start, target))
        return canvas, family, rows
    if domain == "slide_resize":
        rows = []
        for item, (x0, y0) in enumerate(((180, 190), (520, 350))):
            width = 145 + rng.randint(0, 35)
            height = 90 + rng.randint(0, 25)
            rect = (x0, y0, x0 + width, y0 + height)
            draw.rounded_rectangle(rect, radius=10, fill=(236, 242, 245),
                                   outline=accent, width=3)
            draw.text((x0 + 20, y0 + 24), f"CARD {item+1}", fill=ink)
            start = (rect[2], rect[3])
            end = (start[0] + 55 + rng.randint(0, 25),
                   start[1] + 55 + rng.randint(0, 25))
            draw.rectangle((start[0]-7,start[1]-7,start[0]+7,start[1]+7),fill=ink)
            # A visible ghost boundary makes the intended end point inferable.
            draw.rectangle((x0, y0, end[0], end[1]), outline=TARGET_COLOR, width=2)
            draw.ellipse((end[0]-4,end[1]-4,end[0]+4,end[1]+4),fill=TARGET_COLOR)
            rows.append((start,end))
        return canvas, family, rows
    rows = []
    for item,(x0,y0,word) in enumerate(((200,250,"ORBIT"),(510,420,"WILLOW"))):
        draw.rounded_rectangle((x0-20,y0-27,x0+160,y0+30),radius=8,
                               fill=(246,248,250),outline=(181,199,208))
        draw.text((x0,y0-4),word,fill=ink)
        bbox=draw.textbbox((x0, y0-4), word)
        start=(bbox[0],y0+8)
        end=(bbox[2],y0+8)
        draw.line((start[0],y0+14,end[0],y0+14),fill=accent,width=2)
        draw.ellipse((start[0]-3,start[1]-3,start[0]+3,start[1]+3),fill=accent)
        draw.ellipse((end[0]-3,end[1]-3,end[0]+3,end[1]+3),fill=TARGET_COLOR)
        rows.append((start,end))
    return canvas,family,rows


def generate(split: str, index: int, domain: str):
    """Return deterministic image + two distinct bilingual source drag actions.

    No files are written; images only become training bytes if a separately
    authorized training pipeline explicitly admits first-party frames.
    """
    image, family, rows = _draw_scene(split, index, domain)
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False)
    source_bytes = stream.getvalue()
    digest = hashlib.sha256(source_bytes).hexdigest()
    actions = []
    # Counterbalance instruction language vs visual target index.
    # Without this, EN always means object 1, AR always means object 2,
    # creating a spurious language->position shortcut instead of grounding.
    assignment_offset = (index + DOMAINS.index(domain)) % 2
    for n, language in enumerate(("en", "ar")):
        visual_index = (n + assignment_offset) % 2
        start, end = rows[visual_index]
        action = {
            "type": "drag", "start_point": _norm(start),
            "end_point": _norm(end), "ordered": True,
            "coordinate_space": "normalized_1000",
        }
        if start == end or _norm(start) == _norm(end):
            raise ValueError("H426 synthetic action cannot have zero-distance drag")
        if domain == "slider":
            body = (f"Move LEVEL {visual_index+1} slider knob to its outlined green target." if n == 0
                    else f"حرّك مقبض المستوى {visual_index+1} إلى الدائرة الخضراء المحددة.")
        elif domain == "slide_resize":
            body = (f"Enlarge CARD {visual_index+1} to its outlined target corner." if n == 0
                    else f"كبّر البطاقة رقم {visual_index+1} حتى تصل الزاوية إلى العلامة الخضراء.")
        else:
            word = "ORBIT" if visual_index == 0 else "WILLOW"
            body = (f"Highlight only the word {word} in the document." if n == 0
                    else f"ظلّل كلمة {word} فقط في المستند.")
        actions.append({
            "id": f"h426-generated-{split}-{domain}-{index:06d}-{language}",
            "domain": domain, "language": language,
            "source_role": "firstparty_generated_synthetic_research_not_admitted",
            "family": f"invented-ui-family-{family}",
            "image_sha256": digest,
            "body": body,
            "start": list(start), "end": list(end),
            "target": action,
        })
    if (len(actions) != 2 or actions[0]["target"] == actions[1]["target"]
        or actions[0]["body"] == actions[1]["body"]):
        raise ValueError("H426 synthetic bilingual paired tasks are not independent")
    return {"schema": SCHEMA, "split": split, "image": image,
            "image_sha256": digest, "original_png_bytes": source_bytes,
            "family": f"invented-ui-family-{family}", "actions": actions}


def audit_balanced(count_per_domain: int = 12):
    """Finite audit; generator supports lazy scale without storing datasets."""
    if type(count_per_domain) is not int or not 1 <= count_per_domain <= 512:
        raise ValueError("H426 synthetic CPU audit count bounded to 512 per domain")
    counts = {}
    sha_seen = set()
    for split in SPLITS:
        for domain in DOMAINS:
            for index in range(count_per_domain):
                case = generate(split, index, domain)
                if (case["image_sha256"] in sha_seen
                    or any(row["family"] != case["family"]
                           for row in case["actions"])):
                    raise ValueError("H426 synthetic images identical or family mixed")
                sha_seen.add(case["image_sha256"])
                for row in case["actions"]:
                    name = f"{split}/{domain}/{row['language']}"
                    counts[name] = counts.get(name,0)+1
    train_families = {f"invented-ui-family-{n}" for n in SPLITS["train"]}
    dev_families = {f"invented-ui-family-{n}" for n in SPLITS["dev"]}
    if train_families & dev_families:
        raise ValueError("H426 generated synthetic train/dev family overlap")
    return {"schema": SCHEMA, "generated_images": len(sha_seen),
            "generated_action_rows": 2*len(sha_seen),
            "train_dev_source_family_overlap": False,
            "independent_external_ood": False,
            "trained_model": False,
            "source_rights_signoff_for_other_real_screenshots": False,
            "counts": counts}


if __name__ == "__main__":
    print(json.dumps(audit_balanced(), sort_keys=True))
