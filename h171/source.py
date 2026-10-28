"""H171 fresh original licensed click-app TRAIN source; never consumed DEV/OOD."""
import hashlib,json,subprocess
from collections import Counter
from pathlib import Path
from h168.source import historical_image_shas

SCHEMA="hala.h171.fresh-two-original-click-app-train-only.v1"
COUNT=32
PINS={
    "flatpickr":("548330e6e18871d6ca1ca78746c0e89cb12a0303",
                "5fd7cb0b9a6f3e6a7d24f76e310ebe4e2edf050e82065933df98e4689f3e4cee",
                "912da5a92aea3447d39ab83d83706c004c999ee9b013a6267fab3d8710e8a7a8"),
    "awesomplete":("596bc5d5916266437970ecbf65434077d3950a3d",
                   "d3633eb827d8524f843a1f76913b6888e178e712e761d9f116cb362ef4ae98b9"),
}
PREFIXES=("Ja","ja","JA","Jav","jav","JAV","Java","java",
          "jAv","JAVa","JavaS","javas","Javas","javaSc",
          "JavaScri","javascri")

def sha(blob):return hashlib.sha256(blob).hexdigest()

def verify_sources(upstream_flatpickr,upstream_awesomplete,stage,package):
    roots={"flatpickr":Path(upstream_flatpickr),
           "awesomplete":Path(upstream_awesomplete)}
    proof={}
    for app,root in roots.items():
        pin=PINS[app][0]
        actual=subprocess.check_output(
            ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(
            ["git","-C",str(root),"status","--porcelain"],text=True).strip()
        license_file=root/("LICENSE.md" if app=="flatpickr" else "LICENSE")
        html=root/("index.template.html" if app=="flatpickr" else "index.html")
        if (actual!=pin or dirty
            or "MIT" not in license_file.read_text("utf8")[:70]
            or sha(html.read_bytes())!=PINS[app][1]):
            raise ValueError("H171 original donor Git/MIT/HTML identity invalid: "+app)
        proof[app]=dict(commit=pin,license="MIT",html_sha256=PINS[app][1])
    stage=Path(stage)
    if (sha(Path(package).read_bytes())!=PINS["flatpickr"][2]
        or sha((stage/"flatpickr-stage/package/index.template.html").read_bytes())
           !=PINS["flatpickr"][1]
        or not (stage/"flatpickr-stage/package/dist/flatpickr.js").is_file()
        or sha((stage/"awesomplete-upstream/index.html").read_bytes())
           !=PINS["awesomplete"][1]):
        raise ValueError("H171 published flatpickr runtime staging drifted")
    proof["flatpickr"]["npm_tarball_sha256"]=PINS["flatpickr"][2]
    return proof

def prior_hashes(owner,h168_train,h169_dev):
    existing=historical_image_shas(owner)
    items=[
      (Path(h168_train)/"manifest.json",
       "hala.h168.new-original-app-train-pilot.v1",32),
      (Path(h169_dev)/"manifest.json",
       "hala.h169.original-app-new-bilingual-dev-32.v1",32),
    ]
    for path,schema,n in items:
        doc=json.loads(path.read_text("utf8"))
        rows=doc.get("private_cases")
        if doc.get("schema")!=schema or not isinstance(rows,list) or len(rows)!=n:
            raise ValueError("H171 frozen old TRAIN/consumed DEV identity absent")
        for row in rows:
            digest=row.get("image_sha256")
            if not isinstance(digest,str) or len(digest)!=64:
                raise ValueError("H171 old training/DEV source SHA invalid")
            existing.add(digest)
    return existing

def cases():
    rows=[]
    for name in ("flatpickr","awesomplete"):
        for n in range(16):
            rows.append(dict(index=171000+(0 if name=="flatpickr" else 100)+n,
                split="TRAIN_PILOT",app=name,variant=n,action_type="click",
                task="choose_next_day" if name=="flatpickr"
                     else "choose_autocomplete_suggestion",
                language="en" if n%2==0 else "ar"))
    if len(rows)!=COUNT or len({r["index"] for r in rows})!=COUNT:
        raise ValueError("H171 fixed new click TRAIN source schedule invalid")
    return rows

def validate(doc,ref,prior):
    rows=doc.get("private_cases",[])
    actions=ref.get("owner_train_actions",[])
    if (doc.get("schema")!=SCHEMA or doc.get("split")!="TRAIN_PILOT"
        or doc.get("count")!=COUNT or len(rows)!=COUNT
        or len(actions)!=COUNT
        or doc.get("no_prior_dev_or_ood_images") is not True):
        raise ValueError("H171 fresh TRAIN only manifest invalid")
    ids=set();sha_set=set(prior)
    for expected,row,gold in zip(cases(),rows,actions):
        if (any(row.get(k)!=v for k,v in expected.items())
            or row.get("image_sha256") in sha_set
            or len(row.get("image_sha256",""))!=64
            or any(ch not in "0123456789abcdef"
                   for ch in row.get("image_sha256",""))
            or row.get("reference_native_success") is not True
            or not row.get("instruction")
            or any(k in row for k in ("training_supervision",
                                     "owner_gold","reference_point"))
            or gold.get("index")!=row["index"]
            or gold.get("native_success") is not True
            or gold.get("kind")!="click"
            or not isinstance(gold.get("point"),list)
            or len(gold["point"])!=2
            or any(type(v) is not int or not 0<=v<bound for v,bound in zip(
                   gold["point"],(960,720)))
            or row["index"] in ids):
            raise ValueError("H171 owner-only click TRAIN source or historical leakage")
        ids.add(row["index"])
        sha_set.add(row["image_sha256"])
    if Counter(row["app"] for row in rows)!={"flatpickr":16,"awesomplete":16}:
        raise ValueError("H171 original app distribution invalid")
    if Counter(row["language"] for row in rows)!={"en":16,"ar":16}:
        raise ValueError("H171 bilingual split not balanced")
    return dict(status="H171_TWO_NEW_ORIGINAL_CLICK_APPS_TRAIN_ADMITTED",
         cases=COUNT,distinct_original_app_families=2,
         arabic=16,english=16,click=32,
         owner_gold_separate=True,old_dev_ood_used=False,
         optimizer_steps=0,paid_gpu=False)
