"""Deterministic synthetic H34 cases. Oracle geometry never enters model inputs."""
from __future__ import annotations

import hashlib
import json

SCHEMA = "hala.h34.bilingual_visual_dev.v1"
DOMAINS = {
    "mail": ("Mail", "البريد", "Archive", "أرشفة", "Send", "إرسال", "Delete", "حذف", "Snooze", "تأجيل"),
    "calendar": ("Calendar", "التقويم", "Accept", "قبول", "Maybe", "ربما", "Decline", "رفض", "Details", "التفاصيل"),
    "files": ("Files", "الملفات", "Rename", "إعادة تسمية", "Move", "نقل", "Share", "مشاركة", "Copy", "نسخ"),
    "shop": ("Store", "المتجر", "Cart", "السلة", "Save", "حفظ", "Remove", "إزالة", "Compare", "مقارنة"),
}
LAYOUTS = ("row", "column", "rtl_row", "scattered")


def build_cases() -> list[dict]:
    cases = []
    for domain_index, (domain, strings) in enumerate(DOMAINS.items()):
        title_en, title_ar, *labels = strings
        for lang in ("en", "ar"):
            title = title_en if lang == "en" else title_ar
            names = labels[::2] if lang == "en" else labels[1::2]
            for layout_index, layout in enumerate(LAYOUTS):
                language_index = 0 if lang == "en" else 1
                target_index = (domain_index * 3 + layout_index * 2 + language_index) % 4
                control_names = names[:]
                # Rotate target identity independently of position; all alternatives remain plausible.
                target_name = control_names.pop(target_index)
                goal = (f"In {title}, choose the {target_name} button for this task."
                        if lang == "en" else f"في {title}، اختر زر «{target_name}» لتنفيذ المهمة.")
                rotation = (layout_index + domain_index + language_index) % 4
                ordered = names[rotation:] + names[:rotation]
                controls = []
                for position, name in enumerate(ordered):
                    if layout in ("row", "rtl_row"):
                        visual_position = 3 - position if layout == "rtl_row" else position
                        bounds = [72 + visual_position * 216, 300, 184, 64]
                    elif layout == "column":
                        bounds = [380, 170 + position * 82, 200, 60]
                    else:
                        slots = ((100, 150), (630, 180), (180, 340), (600, 350))
                        x, y = slots[position]
                        bounds = [x, y, 184, 64]
                    controls.append({"label": name, "control_index": position,
                                     "bounds": bounds, "is_expected": name == target_name})
                cases.append({"id": f"h34-{domain}-{lang}-{layout}", "domain": domain,
                              "language": lang, "layout": layout, "title": title,
                              "goal": goal, "expected_label": target_name,
                              "controls": controls, "width": 960, "height": 540})
    return cases


def canonical_cases(cases: list[dict] | None = None) -> bytes:
    return (json.dumps(cases if cases is not None else build_cases(), ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def case_sha256(cases: list[dict] | None = None) -> str:
    return hashlib.sha256(canonical_cases(cases)).hexdigest()
