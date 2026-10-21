"""H52 Task Factory: deterministic first-party fictional three-step desktop tasks.

Creates HTML and a sealed authoring plan; does not generate screenshots or
invent model outputs. Actual PNGs/actions require independent Edge replay.
"""
from __future__ import annotations
import hashlib, html, json, random
from pathlib import Path

VERSION="hala.h52.original-three-step-browser-train.v1"
RIGHTS={"origin":"original_first_party_fictional_task_factory",
        "license":"CC0-1.0","official_interface":False,
        "external_assets":False,"personal_data":False,
        "real_accounts":False,"benchmark_answers":False}
# Real public-service sites are not contacted, copied or visually impersonated.
FAMILIES={
 "ar_judicial_office":("ar","مكتب الإجراءات التجريبي","نظام تدريبي خيالي",(
     "طلب تنفيذ تجريبي","مراجعة مستند","إشعار استلام","تحديث حالة",
     "عرض قائمة","حفظ مسودة","فتح طلب","التدقيق النهائي")),
 "ar_traffic_office":("ar","مركز الخدمات التجريبي","بوابة تعلم غير حكومية",(
     "مراجعة رخصة","طلب تصريح","استعراض مخالفة","تحديث بيانات",
     "حفظ طلب","طباعة ملخص","سجل المتابعة","قائمة المواعيد")),
 "en_fhir_workspace":("en","Synthetic FHIR Workbench","No real records or EHR connections",(
     "Observation","Encounter","MedicationRequest","Bundle transaction",
     "Bundle searchset","DiagnosticReport","Provenance","Practitioner")),
 "en_clinical_ops":("en","Fictional Care Operations","Simulated workflow with zero patient data",(
     "Specimen intake","Batch review","Device calibration","FHIR export",
     "Audit queue","Review findings","Archive draft","Task board")),
}
def digest(b): return hashlib.sha256(b).hexdigest()
def canon(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode("utf-8")

def plans():
    out=[]
    for group,(lang,title,disclaimer,labels) in FAMILIES.items():
        for i in range(24):
            rng=random.Random(520000+len(out)*599)
            targets=[rng.randrange(8) for _ in range(3)]
            if len(set(targets))!=3:
                targets=list(rng.sample(range(8),3))
            orders=[rng.sample(range(8),8) for _ in range(3)]
            # Different layouts, option permutations and palettes make same-button
            # positional heuristics unreliable across steps and workflows.
            out.append(dict(id=f"h52-{group}-{i:02d}",family=group,lang=lang,
               split="train",title=title,notice=disclaimer,
               labels=list(labels),targets=targets,orders=orders,
               layout=i%4,accent=i%5,scenario=f"{group.upper()}-{i+100:03d}",
               rights=RIGHTS))
    return out

STYLE="""
*{box-sizing:border-box}
body{margin:0;background:#edf2f8;color:#213244;font:15px Tahoma,'Segoe UI',Arial,sans-serif}
header{height:79px;display:flex;justify-content:space-between;align-items:center;
  padding:0 36px;background:#fff;border-bottom:2px solid #c9d6e3}
header strong{font-size:21px}
header small{color:#61768c}
main{margin:20px 32px}
.title{display:flex;align-items:center;gap:18px;justify-content:space-between}
h1{font-size:21px;margin:0 0 8px}
p{margin:0;color:#5f7183}
.progress{padding:12px 16px;background:#fff;border-radius:9px;border:1px solid #c6d5e4}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
.cards.linear{grid-template-columns:repeat(2,1fr)}
.cards.third{grid-template-columns:repeat(3,1fr)}
.card{height:161px;background:white;border:1px solid #c4d1df;border-radius:12px;
 display:flex;flex-direction:column;justify-content:space-between;padding:13px}
.cards.linear .card{height:80px;flex-direction:row;align-items:center}
.cards.third .card{height:128px}
.letter{font-size:13px;color:#7894a9}.text{font-weight:650;line-height:1.5}
button{border:0;border-radius:7px;background:var(--accent);color:white;
 padding:9px 14px;font:700 14px Tahoma,'Segoe UI',Arial,sans-serif;
 min-width:94px;min-height:42px}
footer{font-size:11px;color:#5b7488;position:fixed;bottom:10px;
 left:22px;right:22px;text-align:center}
#message{font-size:12px;color:#915d41;margin-top:15px}
"""
def render(plan):
    enc=lambda s:html.escape(str(s),quote=True)
    lang=plan["lang"]
    action="اختر" if lang=="ar" else "Select"
    acc=["#166785","#6754a0","#3b7966","#83594d","#5476a6"][plan["accent"]]
    flow="عرض الخطوات" if lang=="ar" else "Workflow step"
    letters="ABCDEFGH"
    body=[
        f'<!doctype html><html lang="{lang}" dir="{"rtl" if lang=="ar" else "ltr"}">',
        f'<head><meta charset="utf-8"><meta name="viewport" content="width=1024,initial-scale=1">',
        f'<style>:root{{--accent:{acc}}}{STYLE}</style></head><body>',
        f'<header><strong>{enc(plan["title"])}</strong><small>{enc(plan["scenario"])}</small></header>',
        '<main><div class="title">',
        f'<div><h1>{enc(plan["title"])}</h1><p>{enc(plan["notice"])}</p></div>',
        f'<div class="progress">{enc(flow)}: <strong id="step">1 / 3</strong></div>',
        '</div><div id="choices" class="cards"></div><p id="message"></p></main>',
        f'<footer>{enc(plan["notice"])} — ORIGINAL SYNTHETIC DATA, NO LIVE ACCOUNTS</footer>',
        '<script>',
        "const L="+json.dumps(plan["labels"],ensure_ascii=False)+";",
        "const O="+json.dumps(plan["orders"])+";",
        "const T="+json.dumps(plan["targets"])+";",
        f"const ACTION={json.dumps(action,ensure_ascii=False)};",
        f"const LAYOUT={plan['layout']};",
        """let step=0,wrong=0;
        function draw(){
          document.getElementById('step').textContent=(step+1)+' / 3';
          let root=document.getElementById('choices');
          root.className='cards '+(LAYOUT===1?'linear':LAYOUT===2?'third':'');
          root.innerHTML='';
          for(let j=0;j<8;j++){
            const key=O[step][j];
            let box=document.createElement('section');box.className='card';
            let code=document.createElement('span');code.className='letter';
            code.textContent=String.fromCharCode(65+j);
            let title=document.createElement('span');title.className='text';
            title.textContent=L[key];
            let btn=document.createElement('button');btn.type='button';
            btn.className='pick';btn.dataset.key=String(key);
            btn.textContent=ACTION;
            btn.addEventListener('click',()=>{
              if(key!==T[step]){
                wrong++;document.body.dataset.wrong=String(wrong);
                document.getElementById('message').textContent='Incorrect choice (simulation)';
                return;
              }
              const previous=step;step++;
              document.body.dataset.lastCorrect=String(key);
              document.body.dataset.completedSteps=String(step);
              document.getElementById('message').textContent='';
              if(step===3){root.innerHTML='';document.getElementById('step').textContent='3 / 3';
                 document.body.dataset.taskFinished='true';return;}
              draw();
            });
            box.append(code,title,btn);root.appendChild(box);
          }
        }
        draw();""",
        '</script></body></html>'
    ]
    return "".join(body).encode("utf-8")

def make(out):
    out=Path(out)
    if out.exists(): raise FileExistsError("Original authored factory scenes create-only")
    planset=plans()
    if len(planset)!=96 or sum(x["lang"]=="ar" for x in planset)!=48:raise ValueError
    out.mkdir(parents=True)
    (out/"html").mkdir()
    receipts=[]
    for plan in planset:
        data=render(plan)
        path=out/"html"/(plan["id"]+".html")
        path.write_bytes(data)
        receipts.append(dict(id=plan["id"],sha256=digest(data)))
    (out/"plans.json").write_bytes(canon(planset))
    (out/"source.json").write_bytes(canon(dict(schema=VERSION,
          rights=RIGHTS,count=96,steps_per_task=3,
          original_html_sha256=digest(canon(receipts)),
          authored_plans_sha256=digest(canon(planset)),
          scenes=receipts,training_only=True,
          benchmark_eval_included=False,real_government_site=False,
          external_dataset_media=False,model_inference_executed=False)))
    print(json.dumps(dict(original_workflows=96,
               original_html=96,expected_train_screenshots=288,
               source_sha256=digest((out/"source.json").read_bytes()))))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    make(p.parse_args().output)
