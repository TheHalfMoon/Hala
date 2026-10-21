"""H49: original, offline, fictional Arabic public-service and English FHIR GUI.

Inspired only by official public service taxonomy and FHIR R4 terminology.
No government sites, user accounts, official assets, PHI, or API actions.
"""
import hashlib
import html
import json
import random
from pathlib import Path

SCHEMA = "hala.h49.original-offline-domain-grounding.v1"
WIDTH, HEIGHT = 1024, 640
RIGHTS = dict(origin="original_first_party_offline_simulation", license="CC0-1.0",
              official_interface=False, external_assets=False, personal_data=False,
              real_accounts=False, benchmark_answers=False)
SOURCES = dict(
    najiz="https://najiz.sa/applications/landing/",
    absher="https://www.absher.sa/",
    fhir_observation="https://hl7.org/fhir/R4/observation.html",
    fhir_bundle="https://hl7.org/fhir/R4/bundle.html")
# Each choice pairs an original UI label with its independent natural-language goal.
FAMILIES = (
 ("justice_execution","train","ar","الخدمات العدلية التجريبية","grid",(
  ("متابعة ملف تنفيذ","ابحث عن متابعة وضع طلب تنفيذ موجود"),
  ("بدء إجراء تنفيذي","اختر خدمة بدء طلب تنفيذ جديد"),
  ("مستندات ملف قضائي","افتح قائمة مستندات القضية المرفقة"),
  ("مواعيد الجلسات","انتقل لتنظيم موعد جلسة"),
  ("الإشعارات العدلية","اعرض الإشعارات المرتبطة بالملفات"),
  ("أرشيف الإجراءات","اختر عرض سجل الإجراءات السابقة"))),
 ("traffic_services","train","ar","خدمات المركبات التجريبية","sidebar",(
  ("صلاحية تأمين مركبة","تحقق من سريان تغطية التأمين على سيارة"),
  ("المركبات المحجوزة","اعرض قائمة السيارات المحجوزة"),
  ("المواعيد المرورية","انتقل لتحديد موعد لخدمة مرورية"),
  ("بيانات رخصة القيادة","ابحث عن تفاصيل رخصة القيادة"),
  ("المخالفات المرورية","افتح سجل المخالفات المرتبطة بالمرور"),
  ("بيانات تسجيل المركبة","استعرض تفاصيل تسجيل سيارة"))),
 ("fhir_resources","train","en","Fictional FHIR Resource Viewer","grid",(
  ("Observation","Find the resource representing a measured value"),
  ("DiagnosticReport","Locate the summary of a diagnostic test report"),
  ("Patient","Open the demographic identity resource"),
  ("Encounter","Find the record of a healthcare visit"),
  ("MedicationRequest","Locate the medication order resource"),
  ("Practitioner","Find the healthcare professional resource"))),
 ("fhir_bundles","train","en","Fictional FHIR Exchange Console","table",(
  ("Bundle searchset","Find the container for a search response"),
  ("Bundle transaction","Find a package of atomic resource changes"),
  ("Bundle document","Find a composition-led clinical document package"),
  ("Bundle message","Locate a FHIR message exchange package"),
  ("Bundle collection","Open a group of independent resources"),
  ("Bundle history","Find the resource version history results"))),
 ("municipal_portal","dev","ar","لوحة خدمات بلدية افتراضية","grid",(
  ("متابعة تصريح موقع","تحقق من حالة إذن موقع سابق"),
  ("نموذج طلب جديد","ابدأ رفع استمارة طلب تجريبي"),
  ("مخططات الأراضي","اعرض الرسم التخطيطي لقطعة أرض"),
  ("بيانات التواصل","انتقل لتحديث بيانات الاتصال"),
  ("موعد مراجعة","اختر موعدًا للزيارة المكتبية"),
  ("إشعارات الموقع","راجع آخر رسائل حالة الموقع"))),
 ("research_lab","dev","en","Fictional Research Laboratory","sidebar",(
  ("Specimen receipt","Check the arrival record of a specimen"),
  ("Calibration log","Find the instrument calibration history"),
  ("Protocol version","Review the current procedure revision"),
  ("Instrument queue","Open the waiting list for an analyzer"),
  ("Quality audit","Inspect the laboratory quality review"),
  ("Inventory request","Locate a supplies replenishment request"))),
)
def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+"\n").encode("utf-8")
def sha(data):
    return hashlib.sha256(data).hexdigest()
def scenes():
    rows=[]
    for fi,(family,split,lang,title,layout,options) in enumerate(FAMILIES):
        for i in range(16):
            order=list(range(6))
            random.Random(490017+fi*17+i*101).shuffle(order)
            target=(i*5+fi)%6
            rows.append(dict(id=f"h49-{family}-{i:02d}",family=family,
                split=split,language=lang,title=title,layout=layout,
                goal=options[target][1],choices=[dict(key=k,label=options[k][0]) for k in order],
                target_index=order.index(target),target_key=target,
                variant=i%4,rights=dict(RIGHTS)))
    return rows
def validate(rows):
    if rows != scenes() or len(rows)!=96:
        raise ValueError("H49 immutable original authoring scenes do not match")
    for row in rows:
        if row["choices"][row["target_index"]]["key"]!=row["target_key"]:
            raise ValueError("Incorrect semantic target")
    train={r["family"] for r in rows if r["split"]=="train"}
    dev={r["family"] for r in rows if r["split"]=="dev"}
    if train&dev or len(train)!=4 or len(dev)!=2:
        raise ValueError("H49 dev/training family overlap")
    if {(r["language"],r["split"]) for r in rows}!={("ar","train"),("en","train"),("ar","dev"),("en","dev")}:
        raise ValueError("Missing language/split")
    for split,n in (("train",32),("dev",16)):
        for lang in ("ar","en"):
            if sum(r["split"]==split and r["language"]==lang for r in rows)!=n:
                raise ValueError("Unbalanced H49 languages")
CSS="""
*{box-sizing:border-box}
html,body{width:1024px;height:640px;margin:0;overflow:hidden}
body{background:#f3f6fa;color:#23374a;font-family:Tahoma,'Segoe UI',Arial,sans-serif}
header{height:79px;background:#fff;border-bottom:2px solid #dbe4ee;display:flex;
align-items:center;justify-content:space-between;padding:0 43px}
header strong{font-size:22px}header small{font-size:12px;color:#627386}
main{margin:17px 40px}h1{font-size:23px;margin:0 0 10px}
.sub{font-size:12px;margin:0 0 20px;color:#657281}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.table{display:flex;flex-direction:column;gap:7px}
.side{display:grid;grid-template-columns:206px 1fr;gap:18px}
aside{height:490px;border-radius:12px;padding:22px;background:#e5edf6}
aside p{line-height:2.8;font-size:13px}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px;
background:white;border:1px solid #ccd8e4;border-radius:10px;padding:12px}
.grid .row{height:163px;flex-direction:column;align-items:stretch;justify-content:space-around}
.table .row,.side .row{height:62px}
.label{font-weight:650;font-size:15px;line-height:1.6;min-width:0}
button.pick{background:var(--accent);border:none;border-radius:8px;color:white;
font:700 14px Tahoma,'Segoe UI',Arial,sans-serif;padding:10px;min-width:109px;
min-height:45px;cursor:default}
.grid button{width:100%}
footer{position:fixed;bottom:12px;width:100%;text-align:center;font-size:11px;
direction:ltr;color:#64758a}
"""
def render(row):
    if row["rights"]!=RIGHTS:
        raise ValueError("Not original-rights H49 source")
    e=lambda s:html.escape(str(s),quote=True)
    language=row["language"]
    action="اختر" if language=="ar" else "Select"
    buttons="".join(
      f'<section class="row"><span class="label">{e(c["label"])}</span>'
      f'<button class="pick" data-idx="{i}" data-key="{c["key"]}" '
      f'aria-label="{e(action)}: {e(c["label"])}">{action}</button></section>'
      for i,c in enumerate(row["choices"]))
    if row["layout"]=="grid":
        body=f'<div class="grid">{buttons}</div>'
    elif row["layout"]=="table":
        body=f'<div class="table">{buttons}</div>'
    else:
        nav="الفهرس" if language=="ar" else "Navigation"
        body=f'<div class="side"><aside><h3>{nav}</h3><p>◇ &nbsp; ◇ &nbsp; ◇<br>◦ &nbsp; ◦ &nbsp; ◦</p></aside><div class="table">{buttons}</div></div>'
    notice=("واجهة تجريبية مستقلة — ليست خدمة حكومية رسمية" if language=="ar"
            else "ORIGINAL OFFLINE SIMULATION — NOT A CLINICAL SYSTEM")
    accent=("#17617f","#69509d","#268066","#80593c")[row["variant"]]
    return (
      f'<!doctype html><html lang="{language}" dir="{"rtl" if language=="ar" else "ltr"}">'
      f'<head><meta charset="utf-8"><style>:root{{--accent:{accent}}}{CSS}</style></head>'
      f'<body><header><strong>{e(row["title"])}</strong><small>HALA · Offline Research</small></header>'
      f'<main><h1>{e(row["title"])}</h1><p class="sub">{e(notice)}</p>{body}</main>'
      f'<footer>{e(notice)}</footer>'
      '<script>document.querySelectorAll("button.pick").forEach(b=>b.addEventListener("click",'
      '()=>{document.body.dataset.selected=b.dataset.key;'
      'document.body.dataset.selectionCount=String(Number(document.body.dataset.selectionCount||0)+1)}));'
      '</script></body></html>'
    ).encode("utf-8")
def generate(root):
    root=Path(root)
    if root.exists():
        raise FileExistsError("H49 generation is create-only")
    root.mkdir(parents=True)
    (root/"html").mkdir()
    (root/"png").mkdir()
    rows=scenes();validate(rows)
    manifest=dict(schema=SCHEMA, rights=RIGHTS, sources_for_vocabulary_only=SOURCES,
        count=96,train=64,public_dev=32,independent_blind_test=False,
        real_service_test=False,model_inference=False,rows=[])
    for row in rows:
        payload=render(row)
        (root/"html"/(row["id"]+".html")).write_bytes(payload)
        manifest["rows"].append(dict(id=row["id"],family=row["family"],split=row["split"],
            language=row["language"],goal=row["goal"],target_index=row["target_index"],
            target_key=row["target_key"],html_sha256=sha(payload)))
    (root/"source.json").write_bytes(canonical(manifest))
    return manifest
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="h49/assets")
    print(json.dumps({"created":generate(p.parse_args().root)["count"]}))
