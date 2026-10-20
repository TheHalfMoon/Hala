"""H48 new original screenshot-grounding TRAIN-only app surfaces.

Eight new fictional app families and eight distinct layout themes; 128 seeded
training scenes balanced Arabic/English. No H43/H44/H45 images or test answers.
No benchmark status, no model score. Output is deterministic and first party.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path

SCHEMA="hala.h48.original-train-only-visual-scenes.v2"
FAMILIES=(
 ("cobalt_dispatch","Cobalt Dispatch","إرسال كوبالت","sidebar"),
 ("lumen_archive","Lumen Archive","أرشيف لومين","table"),
 ("sunrise_ops","Sunrise Operations","عمليات شروق","toolbar"),
 ("marble_settings","Marble Settings","إعدادات ماربل","settings"),
 ("ember_catalog","Ember Catalog","كتالوج إمبر","grid"),
 ("cedrus_notices","Cedrus Notices","إشعارات سيدروس","dialog"),
 ("falcon_tasks","Falcon Tasks","مهام فالكون","kanban"),
 ("ribbon_reports","Ribbon Reports","تقارير ريبون","tabs"),
)
WORDS={
 "en":("Overview","Messages","Invoices","Documents","Projects","Calendar","Analytics","History",
       "Exports","Approvals","Assets","Members","Drafts","Archive","Templates","Preferences",
       "Queues","Alerts","Reviews","Campaigns","Reports","Workflows","Receipts","Inventory"),
 "ar":("الرئيسية","الرسائل","الفواتير","المستندات","المشاريع","التقويم","التحليلات","السجل",
       "التصدير","الموافقات","الملفات","الأعضاء","المسودات","الأرشيف","القوالب","التفضيلات",
       "الطوابير","التنبيهات","المراجعات","الحملات","التقارير","المهام","الإيصالات","المخزون"),
}
GOALS={"en":("Open {label} in {app}.","Choose {label} from {app}.","Navigate to {label} in {app}.",
             "Select {label} in {app}."),
       "ar":("افتح {label} في {app}.","اختر {label} من {app}.",
              "انتقل إلى {label} في {app}.","حدد {label} في {app}.")}
VIEW_W,VIEW_H=960,540


def canon(x:object)->bytes:
    return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")


def sha(x:bytes)->str:
    return hashlib.sha256(x).hexdigest()


def source_cases()->list[dict]:
    result=[]
    for fi,(fam,title_en,title_ar,layout) in enumerate(FAMILIES):
        for n in range(16):
            lang="en" if n%2==0 else "ar"
            lex=WORDS[lang]
            # Four labels, no duplicates, with position and target cycling.
            offsets=[(n+fi*5+j*7)%len(lex) for j in range(4)]
            labels=[lex[i] for i in offsets]
            if len(set(labels))!=4:raise ValueError("Source vocabulary unexpectedly collided")
            target=(3*n+fi)%4
            name=title_en if lang=="en" else title_ar
            goal=GOALS[lang][n%4].format(label=labels[target],app=name)
            result.append({"id":f"h48-{fam}-{n:02d}-{lang}","family":fam,"layout":layout,
                           "language":lang,"rtl":lang=="ar","split":"train_only",
                           "app":name,"goal":goal,"labels":labels,
                           "target_option_index":target,
                           "source":"new_original_first_party_synthetic",
                           "license":"CC0-1.0","independent_test":False,
                           "contains_real_personal_data":False,"benchmark_task":False})
    assert len(result)==128
    return result


def page(case:dict)->bytes:
    esc=lambda s:html.escape(str(s),quote=True)
    rtl=case["rtl"]
    layout=case["layout"]
    title=esc(case["app"])
    goal="Pick a section" if not rtl else "اختر قسمًا"
    opts=[]
    for j,lab in enumerate(case["labels"]):
        opts.append(f'<button type="button" data-option="{j}" aria-label="{esc(lab)}">'
                    f'<span class="mark">{["◈","◉","▣","◒"][j]}</span><span>{esc(lab)}</span></button>')
    buttons="".join(opts)
    style="""*{box-sizing:border-box}
html,body{width:960px;height:540px;margin:0;overflow:hidden}
body{font:15px 'Segoe UI',Tahoma,Arial,sans-serif;background:#eef2f8;color:#1b2c49}
main{position:relative;margin:32px auto;width:904px;height:476px;border-radius:15px;
background:#fff;border:1px solid #c7d6e5;padding:27px 34px;box-shadow:0 10px 35px #182c4930}
h1{font-size:26px;margin:3px 0 6px;font-weight:700}
p{font-size:14px;color:#526681;margin:0 0 24px}
small{font-size:13px;color:#547293}
section{background:#f6f9fd;border:1px solid #d6e0eb;border-radius:12px;padding:15px;gap:12px}
button{display:flex;align-items:center;gap:14px;text-align:start;background:#fff;color:#243958;
min-height:53px;padding:13px 20px;border-radius:9px;border:1px solid #b6cbe0;
font:600 17px 'Segoe UI',Tahoma,Arial,sans-serif;white-space:nowrap}
.mark{font-size:19px;color:#32769d}
section.sidebar{display:flex;flex-direction:column;width:310px;height:332px}
section.sidebar button{flex:1}
section.table{display:grid;grid-template-columns:1fr;grid-template-rows:repeat(4,1fr);height:320px}
section.toolbar{display:flex;align-items:start;justify-content:space-between;height:194px}
section.toolbar button{flex:1;min-width:0;font-size:14px;padding:14px 9px}
section.settings{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:repeat(2,1fr);height:285px}
section.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:repeat(2,1fr);height:319px}
section.dialog{margin:12px auto;width:610px;height:300px;display:grid;grid-template-columns:1fr 1fr;
grid-template-rows:repeat(2,1fr);box-shadow:0 12px 35px #34486635}
section.kanban{display:flex;justify-content:space-around;align-items:center;height:285px}
section.kanban button{flex:1;min-width:0;height:220px;white-space:normal;text-align:center;flex-direction:column;justify-content:center}
section.tabs{display:flex;align-items:flex-start;height:270px}
section.tabs button{flex:1;min-width:0;white-space:normal;padding:14px 11px;font-size:14px}
[dir="rtl"] button{text-align:right}"""
    body=f'<!doctype html><html lang="{case["language"]}" dir="{"rtl" if rtl else "ltr"}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style></head><body><main><small>{title} — {"offline demo" if not rtl else "تجربة محلية"}</small><h1>{title}</h1><p>{esc(goal)}</p><section class="{layout}">{buttons}</section></main></body></html>'
    return body.encode("utf-8")


def create(root:Path)->dict:
    root=Path(root)
    pages=root/"pages"
    if pages.exists() and any(pages.iterdir()):raise FileExistsError("Never overwrite original training HTML")
    pages.mkdir(parents=True,exist_ok=True)
    rows=source_cases()
    description=[]
    for c in rows:
        b=page(c)
        (pages/(c["id"]+".html")).write_bytes(b)
        description.append({"id":c["id"],"family":c["family"],"language":c["language"],
            "layout":c["layout"],"goal":c["goal"],"source_sha256":sha(b),
            "target_option_index":c["target_option_index"],"labels":c["labels"]})
    out={"schema":SCHEMA,"train_only":True,
         "all_original_cc0":True,"external_benchmark":False,
         "blind_holdout":False,"model_qualified":False,
         "sources":description}
    path=root/"source_manifest.json"
    if path.exists():raise FileExistsError("Original source manifest exists")
    path.write_bytes(canon(out))
    return out


if __name__=="__main__":
    print(len(create(Path("h48/assets-v2"))["sources"]))
