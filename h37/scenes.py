"""H37 first-party browser-rasterized bilingual button-grounding fixtures.

DEV and disjoint-domain one-shot TEST are both authored synthetic browser pages,
not independent external websites or a blind user task. No user information.
"""
from __future__ import annotations

import hashlib
import html
import json

WIDTH, HEIGHT = 960, 540
LAYOUTS = ("toolbar", "sidebar", "grid", "table")
FAMILIES = {
    "makerspace": {
        "split": "dev",
        "en": ["Reserve soldering station", "Book 3D printer", "Request laser cutter", "Join safety class"],
        "ar": ["احجز محطة اللحام", "احجز الطابعة ثلاثية الأبعاد", "اطلب آلة القطع بالليزر", "انضم لدورة السلامة"],
        "title_en": "Makerspace bookings", "title_ar": "حجوزات ورشة الابتكار",
    },
    "community_theater": {
        "split": "dev",
        "en": ["Reserve stage rehearsal", "Request costume fitting", "Book lighting desk", "Schedule sound check"],
        "ar": ["احجز تدريب المسرح", "اطلب تجربة الأزياء", "احجز طاولة الإضاءة", "حدد فحص الصوت"],
        "title_en": "Community theater", "title_ar": "المسرح المجتمعي",
    },
    "trail_maintenance": {
        "split": "test",
        "en": ["Report fallen branch", "Request trail map", "Schedule bridge check", "Join cleanup shift"],
        "ar": ["أبلغ عن غصن ساقط", "اطلب خريطة المسار", "حدد فحص الجسر", "انضم لفريق التنظيف"],
        "title_en": "Trail stewardship", "title_ar": "صيانة مسارات المشي",
    },
    "library_workshops": {
        "split": "test",
        "en": ["Register for writing lab", "Reserve study room", "Join reading circle", "Book archive orientation"],
        "ar": ["سجل في ورشة الكتابة", "احجز غرفة الدراسة", "انضم لحلقة القراءة", "احجز تعريف الأرشيف"],
        "title_en": "Library learning", "title_ar": "برامج المكتبة التعليمية",
    },
}
PALETTE = ("#2d7190", "#7157a3", "#567d44", "#ad6642")


def cases() -> list[dict]:
    out = []
    for family_index, (family, defn) in enumerate(FAMILIES.items()):
        for language in ("en", "ar"):
            for layout_index, layout in enumerate(LAYOUTS):
                for variant in (0, 1):
                    labels = list(defn[language])
                    rotation = (family_index + layout_index + variant) % len(labels)
                    options = labels[rotation:] + labels[:rotation]
                    target_index = (family_index + layout_index * 2 + variant + (language == "ar")) % 4
                    goal_label = options[target_index]
                    goal = (f"Select the action labeled '{goal_label}' on the page."
                            if language == "en" else
                            f"اختر الإجراء المكتوب عليه «{goal_label}» في الصفحة.")
                    identifier = f"h37-{family}-{language}-{layout}-v{variant}"
                    out.append({
                        "id": identifier, "family": family, "split": defn["split"],
                        "language": language, "layout": layout, "variant": variant,
                        "title": defn["title_"+language], "goal": goal,
                        "options": options, "expected_control_id": f"control-{target_index}",
                        "width": WIDTH, "height": HEIGHT,
                        "palette": PALETTE[family_index],
                    })
    if len(out) != 64:
        raise ValueError("H37 fixture count drift")
    return out


def normalized_scenes() -> bytes:
    return (json.dumps(cases(), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def scenes_sha256() -> str:
    return hashlib.sha256(normalized_scenes()).hexdigest()


def html_for_case(case: dict) -> str:
    if case["id"] not in {x["id"] for x in cases()}:
        raise ValueError("unknown H37 case identity")
    rtl = case["language"] == "ar"
    esc = lambda s: html.escape(str(s), quote=True)
    body = []
    for index, text in enumerate(case["options"]):
        # DOM id is for evaluator geometry only and never a displayed answer marker.
        body.append(
            '<button type="button" class="action" data-control-id="control-'+str(index)+'">'
            '<span class="action-label">'+esc(text)+'</span></button>')
    actions = "\n".join(body)
    direction = "rtl" if rtl else "ltr"
    subtitle = ("Choose an available service" if not rtl else "اختر الخدمة المتاحة")
    caption = ("Service center" if not rtl else "مركز الخدمات")
    description = ("Available requests today" if not rtl else "الطلبات المتاحة اليوم")
    # No network resources, scripts, oracle attributes, target class or offsite navigation.
    return f"""<!doctype html>
<html lang="{case['language']}" dir="{direction}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --brand: {case["palette"]}; color-scheme: light; }}
* {{ box-sizing: border-box; }}
html,body {{ margin:0; padding:0; height:100%; }}
body {{ color: #192735; background: #edf1f5;
font: 19px/1.4 "Segoe UI", Tahoma, Arial, sans-serif; }}
header {{ height:104px; background:#fff; border-bottom:2px solid #d2d9e2;
padding: 16px 42px; display:flex; gap:18px; align-items:center; }}
mark {{ border-radius:9px; padding:10px; background:var(--brand);color:white;font-weight:700; }}
header .title {{ flex:1; }}
h1 {{ margin:0; font-size:26px; letter-spacing:.1px; }}
.subtitle {{ color:#576471; font-size:15px; }}
main {{ margin:28px 36px; border-radius:14px; background:white;
min-height:360px; padding:24px 28px; border:1px solid #d4dbe2; }}
h2 {{ font-size:21px; margin:2px 0 18px; }}
.actions {{ display:flex; gap:16px; align-items:stretch; }}
.action {{ border:2px solid #7f98a9; color:#182a3b; background:#eaf3fc;
padding:14px 18px; border-radius:12px; min-height:65px;
font: 600 16px/1.3 "Segoe UI",Tahoma,Arial,sans-serif;cursor:default; }}
.action-label {{ display:block; text-align:center; }}
.actions.toolbar {{ flex-direction:row; }}
.toolbar .action {{ flex:1; font-size:14px; padding:10px 8px; min-width:0; }}
.actions.sidebar {{ flex-direction:column; width:360px; gap:12px; }}
.sidebar .action {{ text-align:start; }}
.actions.grid {{ display:grid;grid-template-columns:repeat(2,1fr);max-width:680px; gap:18px; }}
.actions.table {{ display:grid;grid-template-columns:repeat(1,1fr);max-width:690px;gap:6px; }}
.table .action {{ border-radius:5px;min-height:54px;text-align:start;background:white; }}
small {{ font-size:13px; color:#708092; }}
</style></head>
<body><header><mark>{esc(caption)}</mark><div class="title"><h1>{esc(case['title'])}</h1>
<div class="subtitle">{esc(subtitle)}</div></div></header>
<main><h2>{esc(description)}</h2>
<div class="actions {esc(case['layout'])}">{actions}</div>
</main></body></html>
"""


def model_projection(scenes: list[dict]) -> list[dict]:
    """Exclude expected index/control geometry from model-bound rows."""
    return [{"id": x["id"], "split": x["split"], "language": x["language"],
             "image": f"{x['id']}.png", "goal": x["goal"],
             "width": WIDTH, "height": HEIGHT} for x in scenes]
