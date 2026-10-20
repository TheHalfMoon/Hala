"""H48: 96 truly screenshot-rendered original bilingual grounding scenes.

6 different visual task families; 4 authoring, 2 public DEV. Fictional text,
locally generated HTML, no third-party assets/scraped tasks. NOT blind TEST.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path

SCHEMA="hala.h48.diverse-first-party-visual-grounding.v1"
WIDTH,HEIGHT=1024,640
PROVENANCE={"origin":"original_first_party_fictional","license":"CC0-1.0",
            "external_images":False,"benchmark_trajectories":False,
            "personal_data":False}
FAMILIES=(
 ("modal_approval","authoring","Review a request","مراجعة طلب","modal","Approve","موافقة"),
 ("inbox_triage","authoring","Inbox actions","إجراءات البريد","inbox","Archive","أرشفة"),
 ("settings_access","authoring","Access preferences","إعدادات الوصول","settings","Enable","تفعيل"),
 ("ticket_board","authoring","Ticket board","لوحة التذاكر","board","Open","فتح"),
 ("calendar_slots","dev","Session calendar","تقويم الجلسات","calendar","Reserve","حجز"),
 ("expense_review","dev","Expense review","مراجعة المصروفات","expenses","Accept","اعتماد"),
)
PALETTES=(("#f2f6fa","#176a88","#d5e3ec"),("#fbf7f4","#a24f3e","#ebd9ce"),
          ("#f6f4fc","#61449b","#e0d9f4"),("#f0f7f2","#297759","#d2e7da"))
CODE_PREFIX=("KITO","VENA","NORI","SERA","MILO","TARO")


def enc(value:object)->str:return html.escape(str(value),quote=True)

def canonical(obj:object)->bytes:
    return (json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")

def sha(blob:bytes)->str:return hashlib.sha256(blob).hexdigest()

def scenes()->list[dict]:
    cases=[]
    for fidx,(family,split,eng,ara,layout,action_en,action_ar) in enumerate(FAMILIES):
        for i in range(16):
            rtl=i%2==1
            language="ar" if rtl else "en"
            target_pos=(i*5+fidx*3)%6
            code=f"{CODE_PREFIX[fidx]}-{i+21:03d}"
            identifiers=[f"{CODE_PREFIX[fidx]}-{i+100+k*17:03d}" for k in range(6)]
            identifiers[target_pos]=code
            # Collision guard; generated code and decoys must remain distinct.
            if len(set(identifiers))!=6:raise ValueError("Ambiguous same-code distractor")
            title=ara if rtl else eng
            verb=action_ar if rtl else action_en
            goal=(f"في {title}، اضغط «{verb}» للعنصر {code} فقط." if rtl else
                  f"In {title}, press '{verb}' for item {code} only.")
            cases.append({"id":f"h48-{family}-{i:02d}-{language}",
                          "family":family,"split":split,"layout":layout,
                          "language":language,"rtl":rtl,"title":title,
                          "verb":verb,"goal":goal,"target_code":code,
                          "target_index":target_pos,"items":identifiers,
                          "palette":i%len(PALETTES),"source":dict(PROVENANCE)})
    return cases

def validate(rows:list[dict])->None:
    if type(rows) is not list or len(rows)!=96:raise ValueError("H48 expects 96 exact original cases")
    expected=scenes()
    if rows!=expected:raise ValueError("Original source/provenance/split/image labels changed")
    training={x["family"] for x in rows if x["split"]=="authoring"}
    dev={x["family"] for x in rows if x["split"]=="dev"}
    if training&dev or len(training)!=4 or len(dev)!=2:
        raise ValueError("DEV family leakage")

def body(case:dict)->str:
    layout=case["layout"]
    targets=case["items"]
    verb=enc(case["verb"])
    def button(k:int)->str:
        code=enc(targets[k])
        return f'<button class="act" data-idx="{k}" aria-label="{verb} {code}">{verb}</button>'
    if layout=="modal":
        items="".join(f'<div class="choice"><strong>{enc(x)}</strong>{button(k)}</div>'
                      for k,x in enumerate(targets))
        return f'<div class="overlay"><div class="dialog"><h2>{enc(case["title"])}</h2><p>Ticket selection · اختيار التذكرة</p><div class="choices">{items}</div></div></div>'
    if layout=="inbox":
        items="".join(f'<div class="mail"><span class="dot">●</span><div class="mail-copy"><strong>{enc(x)}</strong><small>Message / رسالة {k+1}</small></div>{button(k)}</div>' for k,x in enumerate(targets))
        return f'<div class="inbox"><h2>{enc(case["title"])}</h2>{items}</div>'
    if layout=="settings":
        items="".join(f'<div class="pref"><div><strong>{enc(x)}</strong><p>Permission / إذن • Scope {k+1}</p></div>{button(k)}</div>' for k,x in enumerate(targets))
        return f'<div class="settings"><h2>{enc(case["title"])}</h2><div class="prefs">{items}</div></div>'
    if layout=="board":
        items="".join(f'<div class="ticket"><span>#{k+1:02d} · Active</span><h3>{enc(x)}</h3><small>Assigned queue / قائمة المهام</small>{button(k)}</div>' for k,x in enumerate(targets))
        return f'<div class="board"><h2>{enc(case["title"])}</h2><div class="board-grid">{items}</div></div>'
    if layout=="calendar":
        days="".join(f'<div class="slot"><span>DAY {k+1:02d} / يوم</span><strong>{enc(x)}</strong>{button(k)}</div>' for k,x in enumerate(targets))
        return f'<div class="calendar"><h2>{enc(case["title"])}</h2><div class="slot-grid">{days}</div></div>'
    if layout=="expenses":
        rows="".join(f'<div class="expense"><span class="rank">{k+1:02d}</span><strong>{enc(x)}</strong><span class="amount">{(k+2)*35:,} SAR</span>{button(k)}</div>' for k,x in enumerate(targets))
        return f'<div class="expenses"><h2>{enc(case["title"])}</h2><div class="expense-head">Reference / المرجع　　　　　Amount / المبلغ</div>{rows}</div>'
    raise ValueError("Unknown H48 scene family")

CSS="""
*{box-sizing:border-box}
html,body{width:1024px;height:640px;overflow:hidden;margin:0}
body{font-family:'Segoe UI',Tahoma,Arial,sans-serif;background:var(--bg);color:#243349}
header{height:74px;display:flex;align-items:center;justify-content:space-between;
 background:white;border-bottom:1px solid var(--line);padding:0 40px;font-size:24px;font-weight:750}
header small{font-size:12px;font-weight:500;color:#526579}
main{padding:16px 40px;height:565px}
h2{font-size:23px;margin:4px 0 17px}h3{font-size:19px;margin:8px 0}
p,small{color:#586b7b;font-size:12px}.act{min-width:102px;min-height:43px;border-radius:9px;
 font-size:14px;font-weight:700;color:#fff;background:var(--accent);
 border:none;cursor:default;padding:9px 13px}
.overlay{height:520px;display:grid;place-items:center;background:#21324c18;border-radius:13px}
.dialog{background:white;border:1px solid var(--line);border-radius:18px;
 box-shadow:0 15px 40px #19263833;width:675px;padding:25px 30px}
.choices{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:17px}
.choice,.mail,.pref,.ticket,.slot,.expense{background:white;border:1px solid var(--line);border-radius:9px}
.choice{display:flex;align-items:center;justify-content:space-between;padding:9px;gap:10px;min-height:60px}
.choice strong{font-size:14px}
.inbox,.settings,.calendar,.expenses{background:white;border:1px solid var(--line);padding:18px 23px;border-radius:15px}
.mail{display:flex;align-items:center;height:67px;margin-top:5px;gap:14px;padding:9px 16px}
.mail-copy{flex:1;display:flex;flex-direction:column;gap:6px;font-size:14px}
.dot{color:var(--accent);font-size:12px}
.prefs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pref{height:116px;padding:14px;display:flex;align-items:center;justify-content:space-between}
.pref strong{font-size:15px}
.board-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.ticket{height:194px;padding:15px;display:flex;flex-direction:column;align-items:flex-start}
.ticket span{font-size:11px;color:#65758a}
.ticket .act{margin-top:auto}
.slot-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.slot{height:170px;display:flex;align-items:center;justify-content:center;
 gap:13px;flex-direction:column}
.slot span{font-size:12px;color:#718091}
.slot strong{font-size:17px}
.expense-head{padding:9px 15px;background:var(--bg);font-size:13px}
.expense{height:60px;margin:5px 0;display:flex;align-items:center;padding:7px 15px;gap:16px}
.rank{width:35px;color:#8896a4;font-weight:700}
.expense strong{flex:1}.amount{width:125px;color:#445a6c;font-size:13px}
"""

def render(case:dict)->bytes:
    if case["source"]!=PROVENANCE:raise ValueError("Unqualified scene rights")
    bg,accent,line=PALETTES[case["palette"]]
    direction="rtl" if case["rtl"] else "ltr"
    result=f"""<!doctype html><html lang="{case['language']}" dir="{direction}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{{--bg:{bg};--accent:{accent};--line:{line}}}{CSS}</style></head>
<body><header><span>HALA · Research Apps</span><small>Offline first-party mock UI / واجهة تجريبية</small></header>
<main>{body(case)}</main></body></html>"""
    return result.encode("utf-8")

def generate(root:Path)->dict:
    root=Path(root)
    html_root=root/"html"
    if any(html_root.iterdir()):raise FileExistsError("Refuse to overwrite H48 originals")
    cases=scenes();validate(cases)
    records=[]
    for row in cases:
        payload=render(row)
        dst=html_root/(row["id"]+".html")
        dst.write_bytes(payload)
        records.append({"id":row["id"],"family":row["family"],"split":row["split"],
          "language":row["language"],"goal":row["goal"],
          "html_sha256":sha(payload),"target_index":row["target_index"]})
    record={"schema":SCHEMA,"provenance":PROVENANCE,"count":len(records),
            "authoring_count":64,"public_dev_count":32,
            "independent_blind_test":False,"actual_model_result":False,"rows":records}
    (root/"source_manifest.json").write_bytes(canonical(record))
    return record

if __name__=="__main__":
    print(json.dumps({"cases":len(generate(Path("h48/assets"))["rows"])}))
