"""H212 self-authored, source-verifiable GUI point SFT research candidates.

Produces only NEW synthetic 960x720 browser-like screenshots and
source-computed click labels. Never reads user media, historical images,
gold, consumed H173 OOD, or reserved applications. These are unadmitted
CANDIDATES, not TRAIN/DEV/OOD, no fine-tune or release.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, HEIGHT, WIDTH, parse_actions,
    source_task_prompt, training_messages_for_synthetic_contract,
)

SCHEMA = "hala.h212.generated-original-gui-candidates.v1"
SPLITS = (
    ("RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED", "toolbar", 32),
    ("RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED", "sidebar", 32),
    ("RESEARCH_DEV_CANDIDATE_NOT_SEALED", "dialog", 16),
    ("RESEARCH_EXTRA_CANDIDATE_NOT_OOD", "footer", 16),
)
COUNT = 96
COLORS = ("#1960c5", "#89929e")
INSTRUCTIONS = {
    "en": "Click the blue SAVE button visible in the screenshot.",
    "ar": "اضغط على زر SAVE الأزرق الظاهر في لقطة الشاشة.",
}


@dataclass(frozen=True)
class Generated:
    index: int
    split: str
    family: str
    language: str
    instruction: str
    image_png: bytes
    image_sha256: str
    visual_rgba_sha256: str
    action_json: str
    source_bbox: tuple[int, int, int, int]


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H212 synthetic candidates: " + reason)


def enumerate_specs() -> tuple[tuple[int, str, str, int], ...]:
    results: list[tuple[int, str, str, int]] = []
    next_idx = 0
    for split, family, count in SPLITS:
        for variant in range(count):
            results.append((next_idx, split, family, variant))
            next_idx += 1
    require(len(results) == COUNT
            and [row[0] for row in results] == list(range(COUNT)),
            "bounded fixed source roster required")
    return tuple(results)


def _rectangles(family: str, variant: int) -> tuple[
    tuple[int, int, int, int], tuple[int, int, int, int]
]:
    """Distinct procedural layouts; no coordinates in model prompt."""
    require(type(variant) is int and 0 <= variant < 32
            and family in {"toolbar", "sidebar", "dialog", "footer"},
            "invalid self-authored scene specification")
    flip = variant % 2
    if family == "toolbar":
        # Across a horizontal toolbar; other action is opposite.
        left_x = 96 + ((variant * 19) % 115)
        right_x = 655 + ((variant * 13) % 100)
        y = 125 + ((variant * 11) % 100)
        a, b = (left_x, y, left_x+166, y+62), (
            right_x, y, right_x+166, y+62)
    elif family == "sidebar":
        left_x, right_x = 88, 675
        y = 215 + ((variant * 31) % 290)
        a, b = (left_x, y, left_x+168, y+65), (
            right_x, y, right_x+168, y+65)
    elif family == "dialog":
        y = 430 + ((variant * 5) % 31)
        a, b = (294, y, 428, y+58), (535, y, 669, y+58)
    else:
        y = 582 + ((variant * 7) % 31)
        a, b = (92, y, 265, y+65), (685, y, 858, y+65)
    return (a, b) if flip == 0 else (b, a)


def render_case(index: int):
    """In-memory only; no filesystem image loading or image retention."""
    from PIL import Image, ImageDraw, ImageFont
    require(type(index) is int and 0 <= index < COUNT,
            "source example index outside bounded roster")
    _id, split, family, variant = enumerate_specs()[index]
    language = "en" if variant % 2 == 0 else "ar"
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "source-only research must not silently enable training")
    save, cancel = _rectangles(family, variant)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f6f9")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH-1, 70), fill="#202737")
    draw.text((35, 25), "Hala original research board",
              fill="white", font=ImageFont.load_default(size=20))
    if family == "toolbar":
        draw.rectangle((40, 98, 915, 305), outline="#d2d7df", width=3)
        draw.text((65, 335), "Write your note", fill="#303b52")
    elif family == "sidebar":
        draw.rectangle((34, 90, 920, 635), outline="#bfc9d3", width=3)
        draw.line((460, 90, 460, 635), fill="#c2ccd5", width=3)
        draw.text((280, 122), "Options", fill="#303b52")
    elif family == "dialog":
        draw.rounded_rectangle((225, 128, 735, 525),
                               radius=22, fill="#ffffff", outline="#c0c8d1", width=4)
        draw.text((320, 205), "Save changes?", fill="#303b52",
                  font=ImageFont.load_default(size=22))
    else:
        draw.rectangle((32, 105, 925, 565), outline="#c0c8d1", width=3)
        draw.text((72, 145), "Document editor", fill="#303b52",
                  font=ImageFont.load_default(size=22))
        draw.line((35, 550, 925, 550), fill="#c0c8d1", width=3)
    for box, color, caption in (
        (save, COLORS[0], "SAVE"),
        (cancel, COLORS[1], "CANCEL"),
    ):
        draw.rounded_rectangle(box, radius=10, fill=color)
        draw.text(((box[0]+box[2])//2, (box[1]+box[3])//2),
                  caption, fill="white", anchor="mm",
                  font=ImageFont.load_default(size=26))
    # Small procedural border pattern provides non-index-based visual
    # variability without leaking geometry or the intended source label.
    accent = 90 + ((variant * 37) % 100)
    draw.line((40, 683, accent+120, 683), fill="#7888a0", width=3)
    buff = BytesIO()
    image.save(buff, format="PNG", optimize=False)
    png = buff.getvalue()
    image_rgba = image.convert("RGBA").tobytes()
    midpoint = [(save[0]+save[2])//2, (save[1]+save[3])//2]
    action = json.dumps(
        {"actions": [{"kind": "click", "point": midpoint}]},
        sort_keys=True, separators=(",", ":"),
    )
    parse_actions(action, training_target=True)
    text = INSTRUCTIONS[language]
    source_task_prompt(text)
    return Generated(
        index, split, family, language, text, png,
        sha256(png).hexdigest(), sha256(image_rgba).hexdigest(),
        action, save,
    )


def research_multimodal_messages(index: int) -> tuple[list[dict], list[dict]]:
    """Real PIL-image HF SFT message candidates; no model/official admission.

    Unlike the H199 synthetic placeholder, this supplies the actual
    generated RGB pixels so an external processor can tokenize them
    once a distinct approved experimental workflow is established.
    """
    from PIL import Image
    case = render_case(index)
    with Image.open(BytesIO(case.image_png)) as decoded:
        require(decoded.format == "PNG" and decoded.size == (WIDTH, HEIGHT),
                "actual generated 960x720 PNG required")
        rgb = decoded.convert("RGB")
        rgb.load()
    action = parse_actions(case.action_json, training_target=True)
    task = source_task_prompt(case.instruction)
    question = [{"role": "user", "content": [
        {"type": "image", "image": rgb},
        {"type": "text", "text": task},
    ]}]
    target = json.dumps(
        action, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    answer = question + [{"role": "assistant", "content": [
        {"type": "text", "text": target},
    ]}]
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "research-only pixels must not become official training")
    return question, answer


def candidate_manifest() -> dict:
    """Full SHA pairs + bounded source identities, **never** a release."""
    source_text = Path(__file__).read_text(encoding="utf-8")
    source_sha = sha256(source_text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    cases = []
    observed_png = set()
    observed_rgba = set()
    for i in range(COUNT):
        case = render_case(i)
        require(case.image_sha256 not in observed_png
                and case.visual_rgba_sha256 not in observed_rgba,
                "duplicate visual candidate or split leakage")
        observed_png.add(case.image_sha256)
        observed_rgba.add(case.visual_rgba_sha256)
        question, answer = training_messages_for_synthetic_contract(
            case.instruction, case.action_json)
        require(question[0]["content"][0] == {"type": "image"}
                and answer[-1]["role"] == "assistant",
                "invalid multimodal completion-only candidate")
        # Source action/gold is *not* in the public candidate manifest:
        # these are NOT sealed DEV and must never be promoted to OOD.
        cases.append({
            "index": i, "family": case.family,
            "role": case.split, "language": case.language,
            "image_png_sha256": case.image_sha256,
            "image_decoded_rgba_sha256": case.visual_rgba_sha256,
        })
    roles = {role: sum(c["role"] == role for c in cases)
             for role, _, _ in SPLITS}
    require(roles == {
        "RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED": 64,
        "RESEARCH_DEV_CANDIDATE_NOT_SEALED": 16,
        "RESEARCH_EXTRA_CANDIDATE_NOT_OOD": 16,
    }, "research roles not as preregistered")
    return {
        "schema": SCHEMA,
        "generator_source_sha256": source_sha,
        "registered_candidate_count": COUNT,
        "role_counts": roles,
        "official_admitted": False,
        "historical_source_history_complete": False,
        "old_rgba_history_complete": False,
        "dev_gold_sealed": False,
        "official_ood_evaluated": False,
        "gpu_optimizer_steps": 0,
        "cases": cases,
    }


def main() -> None:
    doc = candidate_manifest()
    print(
        "H212_NEW_SELF_AUTHORED_GUI_SFT_RESEARCH_CANDIDATES_PASS",
        "examples="+str(doc["registered_candidate_count"]),
        "train_research=64 dev_research=16 extra_research=16",
        "distinct_png_sha=96 distinct_decoded_rgba_sha=96",
        "bilingual_en_ar=true",
        "official_train_admitted=0 sealed_dev=0 ood=0 model_steps=0",
    )


if __name__ == "__main__":
    main()
