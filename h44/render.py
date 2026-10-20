"""Render original H43 first-step task pages; NEVER add target oracle to the page.

No web/network, user files, external assets, JavaScript or real app permissions.
All 24 pages are synthetic and their model input is screenshot+goal only.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
from h43.factory import author_cases,canonical,digest

W,H=960,540


def page(case:dict)->str:
    escape=lambda text:html.escape(str(text),quote=True)
    direction="rtl" if case["rtl"] else "ltr"
    rows=[]
    for c in case["controls"]:
        label=escape(c["name"])
        ref=escape(c["ref"])
        if c["role"]=="Edit":
            rows.append(f'<label class="control"><span>{label}</span><input type="number" value="" data-ref="{ref}" aria-label="{label}" placeholder="{label}"></label>')
        else:
            rows.append(f'<button type="button" data-ref="{ref}" aria-label="{label}">{label}</button>')
    grid={"list":"columns","grid":"tiles","table":"rows","sidebar":"rail"}[case["layout"]]
    eyebrow="Sample task · offline" if not case["rtl"] else "مهمة تجريبية · بدون اتصال"
    heading="Action board" if not case["rtl"] else "لوحة المهام"
    desc=("Select the correct item, enter the requested quantity, then confirm."
          if not case["rtl"] else "اختر العنصر الصحيح ثم أدخل الكمية المطلوبة وأكّد الإجراء.")
    return f"""<!doctype html>
<html lang="{case["language"]}" dir="{direction}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{box-sizing:border-box}}
html,body{{width:{W}px;height:{H}px;overflow:hidden;margin:0;padding:0}}
body{{font-family:'Segoe UI',Tahoma,Arial,sans-serif;color:#15243b;background:#eef3f9}}
main{{margin:24px auto;width:902px;height:492px;background:white;border:1px solid #d9e1eb;border-radius:18px;padding:20px 28px;box-shadow:0 8px 35px #172b4522}}
small{{letter-spacing:.035em;color:#536b8a;font-size:13px}}
h1{{font-size:25px;margin:5px 0 3px;font-weight:700}}
p{{font-size:14px;color:#52667f;margin:4px 0 22px}}
section{{height:318px;padding:16px;background:#f8fafe;border:1px solid #d5dfed;border-radius:12px;gap:12px}}
section.columns{{display:flex;flex-direction:column}}
section.rows{{display:grid;grid-template-columns:1fr;grid-template-rows:repeat(5,1fr)}}
section.tiles{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(3,1fr)}}
section.rail{{display:grid;grid-template-columns:250px 1fr;grid-template-rows:repeat(3,1fr)}}
button,.control{{display:flex;align-items:center;justify-content:flex-start;gap:12px;min-height:43px;min-width:0;width:100%;
 border:1px solid #c4d2e3;border-radius:10px;background:white;color:#263e60;padding:8px 15px;
 text-align:start;font-size:14px;font-weight:600;line-height:1.2}}
button{{cursor:default}}
button[data-ref="confirm_button"]{{background:#1e628c;color:#fff;border-color:#1e628c}}
button[data-ref="unrelated_button"]{{color:#935042;background:#fff8f4;border-color:#e9d1c9}}
.control{{justify-content:space-between}}
.control span{{white-space:nowrap}}
input{{font:inherit;min-height:36px;width:115px;border:1px solid #a9bed7;border-radius:6px;padding:4px 8px}}
.rail button[data-ref="row_target"],.rail button[data-ref="row_decoy"]{{background:#ecf3fa}}
</style></head>
<body><main><small>{escape(eyebrow)} · {escape(case["id"])}</small>
<h1>{escape(case["app"])} — {escape(heading)}</h1>
<p>{escape(desc)}</p><section class="{grid}">{"".join(rows)}</section></main></body></html>
"""


def generate(root:Path)->dict:
    root=Path(root)
    html_dir=root/"pages"
    if html_dir.exists() and any(html_dir.iterdir()):
        raise FileExistsError("Source HTML assets must be create-only")
    html_dir.mkdir(parents=True,exist_ok=True)
    source=author_cases()
    scene=[]
    for case in source["cases"]:
        payload=page(case).encode("utf-8")
        path=html_dir/(case["id"]+".html")
        path.write_bytes(payload)
        scene.append({"id":case["id"],"family":case["family"],"split":case["split"],
                      "language":case["language"],"layout":case["layout"],
                      "rtl":case["rtl"],"goal":case["goal"],
                      "page_sha256":hashlib.sha256(payload).hexdigest(),
                      "width":W,"height":H})
    info={"schema":"hala.h44.original-html-plan.v1",
          "h43_manifest_sha256":digest(source),"no_native_actions":True,
          "external_app":False,"model_evaluation":False,
          "scenes":scene}
    (root/"scenes.json").write_bytes(canonical(info))
    return info

if __name__=="__main__":
    print(json.dumps({"generated":len(generate(Path("h44/assets"))["scenes"])},sort_keys=True))
